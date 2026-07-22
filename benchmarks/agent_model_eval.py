from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import tempfile
import time
from typing import Any
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from openai import OpenAI

from benchmarks.agent_cases import BenchmarkCase, cases_for
from core.llm import _normalize_completion


STRUCTURED_SUITES = {"routing", "extraction"}
VALID_SUITES = {"routing", "tools", "extraction", "finalization", "recovery", "compact"}
CYRILLIC = re.compile(r"[а-яё]", re.IGNORECASE)
PARTIAL_WORDS = re.compile(
    r"частич|недостаточ|не удалось|нет данных|отсутств|пробел|"
    r"partial|insufficient|missing|no verified|cannot be determined",
    re.IGNORECASE,
)

FM_TOOL_SCHEMA = {
    "type": "object",
    "properties": {
        "call_tool": {"type": "boolean"},
        "tool_name": {"type": "string"},
        "query": {"type": "string"},
        "depth": {"type": "string"},
        "command": {"type": "string"},
        "response": {"type": "string"},
    },
    "required": ["call_tool", "tool_name", "query", "depth", "command", "response"],
    "additionalProperties": False,
}


@dataclass
class ModelReply:
    content: str
    tool_calls: list[dict[str, Any]]
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int
    reasoning_tokens: int
    latency: float
    schema_transport: str
    schema_note: str = ""


@dataclass
class Score:
    passed: bool
    earned: int
    possible: int
    notes: list[str]


def _norm(value: Any) -> str:
    text = str(value or "").casefold()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _contains_group(text: str, alternatives: list[str]) -> bool:
    normalized = _norm(text)
    return any(_norm(item) in normalized for item in alternatives)


def _json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
    stripped = re.sub(r"\s*```$", "", stripped)
    start, end = stripped.find("{"), stripped.rfind("}")
    if start < 0 or end < start:
        raise ValueError("JSON object not found")
    parsed = json.loads(stripped[start:end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("top-level JSON must be an object")
    return parsed


def _tool_calls(message: Any) -> list[dict[str, Any]]:
    calls = []
    for call in message.tool_calls or []:
        calls.append({
            "id": call.id,
            "name": call.function.name,
            "arguments": call.function.arguments or "{}",
        })
    return calls


class BenchmarkClient:
    def __init__(self, *, base_url: str, api_key: str, model: str,
                 structured_mode: str, temperature: float, timeout: float,
                 max_tokens_override: int | None = None,
                 schema_dialect: str = "auto", provider: str = "openai",
                 system_suffix: str = ""):
        self.provider = provider
        self.client = (
            OpenAI(base_url=base_url, api_key=api_key, timeout=timeout, max_retries=0)
            if provider in {"openai", "lmstudio"} else None
        )
        self.base_url = base_url
        self.system_suffix = system_suffix.strip()
        self.model = model
        self.structured_mode = structured_mode
        self.temperature = temperature
        self.timeout = timeout
        self.max_tokens_override = max_tokens_override
        if schema_dialect == "auto":
            schema_dialect = (
                "afm" if model in {"system", "pcc"} or ":1976/" in base_url
                else "standard"
            )
        self.schema_dialect = schema_dialect

    def _schema(self, source: dict[str, Any]) -> dict[str, Any]:
        schema = deepcopy(source)
        if self.schema_dialect != "afm":
            return schema

        def prepare(node: Any, title: str) -> None:
            if isinstance(node, list):
                for item in node:
                    prepare(item, title)
            elif isinstance(node, dict):
                if node.get("type") == "object" and isinstance(node.get("properties"), dict):
                    node["title"] = node.get("title") or title
                    node["x-order"] = list(node["properties"])
                    for name, value in node["properties"].items():
                        prepare(value, name.title().replace("_", ""))
                elif node.get("type") == "array":
                    prepare(node.get("items"), title + "Item")

        prepare(schema, "BenchmarkOutput")
        facts = schema.get("properties", {}).get("facts")
        if isinstance(facts, dict) and facts.get("type") == "array":
            item = facts.get("items")
            if isinstance(item, dict) and item.get("type") == "object":
                item["title"] = "Fact"
                schema["$defs"] = {"Fact": item}
                facts["items"] = {"$ref": "#/$defs/Fact"}
        return schema

    @staticmethod
    def _fm_schema(source: dict[str, Any]) -> dict[str, Any]:
        """Convert benchmark schemas to the stricter fm respond file dialect."""
        schema = deepcopy(source)

        def clean(node: Any, title: str) -> None:
            if not isinstance(node, dict):
                return
            node.pop("enum", None)
            if node.get("type") == "object":
                node["title"] = node.get("title") or title
                properties = node.get("properties", {})
                node["x-order"] = list(properties)
                for name, value in properties.items():
                    clean(value, name.title().replace("_", ""))
            elif node.get("type") == "array":
                clean(node.get("items"), title + "Item")

        clean(schema, "BenchmarkOutput")
        facts = schema.get("properties", {}).get("facts")
        if isinstance(facts, dict) and facts.get("type") == "array":
            item = facts.get("items")
            if isinstance(item, dict) and item.get("type") == "object":
                item["title"] = "Fact"
                schema["$defs"] = {"Fact": item}
                facts["items"] = {"$ref": "#/$defs/Fact"}
        return schema

    def _request(self, case: BenchmarkCase, *, native_schema: bool) -> ModelReply:
        if self.provider == "fm":
            return self._request_fm(case)
        messages = [dict(message) for message in case.messages]
        if self.system_suffix:
            system_message = next((message for message in messages if message.get("role") == "system"), None)
            if system_message is None:
                messages.insert(0, {"role": "system", "content": self.system_suffix})
            else:
                system_message["content"] = (
                    str(system_message.get("content", "")).rstrip()
                    + "\n"
                    + self.system_suffix
                )
        schema_transport = "none"
        response_format = None
        if case.response_schema:
            if native_schema:
                schema_transport = "native"
                response_format = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": f"benchmark_{case.suite}",
                        "strict": True,
                        "schema": self._schema(case.response_schema),
                    },
                }
            else:
                schema_transport = "prompt"
                messages.append({
                    "role": "user",
                    "content": (
                        "Return one JSON object matching this JSON Schema exactly. "
                        "Do not use a code fence.\n"
                        + json.dumps(case.response_schema, ensure_ascii=False)
                    ),
                })

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens_override or case.max_tokens,
        }
        if case.tools:
            kwargs["tools"] = case.tools
            kwargs["tool_choice"] = "auto"
        if response_format:
            kwargs["response_format"] = response_format

        started = time.perf_counter()
        response = self.client.chat.completions.create(**kwargs)  # type: ignore[union-attr]
        elapsed = time.perf_counter() - started
        completion = _normalize_completion(response, self.model)
        choice = completion.choices[0]
        usage = getattr(response, "usage", None)
        details = getattr(usage, "completion_tokens_details", None) if usage else None
        return ModelReply(
            content=choice.message.content or "",
            tool_calls=_tool_calls(choice.message),
            finish_reason=choice.finish_reason or "",
            prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
            reasoning_tokens=int(getattr(details, "reasoning_tokens", 0) or 0),
            latency=elapsed,
            schema_transport=schema_transport,
            schema_note="",
        )

    @staticmethod
    def _fm_messages(case: BenchmarkCase) -> tuple[str, str]:
        instructions = "\n\n".join(
            str(message.get("content", ""))
            for message in case.messages
            if message.get("role") == "system"
        )
        transcript: list[str] = []
        for message in case.messages:
            role = message.get("role")
            if role == "system":
                continue
            content = str(message.get("content", "") or "")
            calls = message.get("tool_calls") or []
            if calls:
                rendered = []
                for call in calls:
                    function = call.get("function", {})
                    rendered.append(
                        f"{function.get('name', '')}({function.get('arguments', '{}')})"
                    )
                content = (content + "\nTool call: " + "; ".join(rendered)).strip()
            transcript.append(f"[{role}] {content}")
        return instructions, "\n\n".join(transcript)

    def _request_fm(self, case: BenchmarkCase) -> ModelReply:
        instructions, prompt = self._fm_messages(case)
        schema = case.response_schema
        tool_emulation = bool(case.tools)
        if tool_emulation:
            schema = FM_TOOL_SCHEMA
            prompt += (
                "\n\nOffered tools:\n"
                + json.dumps(case.tools, ensure_ascii=False)
                + "\nChoose whether to call one tool. Put its exact name in tool_name. For "
                  "web_search fill query and depth; for execute_bash fill command. Leave unused "
                  "fields empty. If no tool is needed, set "
                  "call_tool=false and write the answer in response."
            )

        schema_path = None
        try:
            command = ["/usr/bin/fm", "respond", "--model", self.model, "--no-stream", "--greedy"]
            if instructions:
                command.extend(["--instructions", instructions])
            if schema:
                prepared = self._fm_schema(schema)
                handle = tempfile.NamedTemporaryFile(
                    mode="w", encoding="utf-8", suffix=".json", delete=False
                )
                with handle:
                    json.dump(prepared, handle, ensure_ascii=False)
                schema_path = handle.name
                command.extend(["--schema", schema_path])
            command.append(prompt)
            started = time.perf_counter()
            completed = subprocess.run(
                command, capture_output=True, text=True, timeout=self.timeout, check=False
            )
            latency = time.perf_counter() - started
            if completed.returncode != 0:
                raise RuntimeError(completed.stderr.strip() or completed.stdout.strip())
            content = completed.stdout.strip()
            calls: list[dict[str, Any]] = []
            if tool_emulation:
                data = _json_object(content)
                if data.get("call_tool"):
                    name = str(data.get("tool_name", ""))
                    arguments = (
                        {"query": data.get("query", ""), "depth": data.get("depth", "")}
                        if name == "web_search"
                        else {"command": data.get("command", "")}
                    )
                    calls.append({
                        "id": "fm-schema-tool-call",
                        "name": name,
                        "arguments": json.dumps(arguments, ensure_ascii=False),
                    })
                    content = ""
                else:
                    content = str(data.get("response", ""))
            return ModelReply(
                content=content,
                tool_calls=calls,
                finish_reason="stop",
                prompt_tokens=0,
                completion_tokens=0,
                reasoning_tokens=0,
                latency=latency,
                schema_transport="fm_schema_tool_emulation" if tool_emulation else (
                    "fm_schema" if schema else "none"
                ),
                schema_note="fm respond does not expose token usage or native function calls",
            )
        finally:
            if schema_path:
                Path(schema_path).unlink(missing_ok=True)

    def _request_lmstudio(self, case: BenchmarkCase) -> ModelReply:
        system = "\n\n".join(
            str(message.get("content", ""))
            for message in case.messages
            if message.get("role") == "system"
        )
        transcript = "\n\n".join(
            f"[{message.get('role')}] {message.get('content', '')}"
            for message in case.messages
            if message.get("role") != "system"
        )
        schema_transport = "none"
        if case.response_schema:
            schema_transport = "prompt"
            transcript += (
                "\n\nReturn one JSON object matching this JSON Schema exactly. "
                "Do not use a code fence.\n"
                + json.dumps(case.response_schema, ensure_ascii=False)
            )
        root = self.base_url.removesuffix("/v1").rstrip("/")
        payload = {
            "model": self.model,
            "input": transcript,
            "system_prompt": system,
            "reasoning": "off",
            "temperature": self.temperature,
            "max_output_tokens": self.max_tokens_override or case.max_tokens,
        }
        request = Request(
            root + "/api/v1/chat",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        started = time.perf_counter()
        with urlopen(request, timeout=self.timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
        latency = time.perf_counter() - started
        content = "\n".join(
            str(item.get("content", ""))
            for item in data.get("output", [])
            if item.get("type") == "message"
        ).strip()
        stats = data.get("stats") or {}
        return ModelReply(
            content=content,
            tool_calls=[],
            finish_reason="stop" if content else "",
            prompt_tokens=int(stats.get("input_tokens", 0) or 0),
            completion_tokens=int(stats.get("total_output_tokens", 0) or 0),
            reasoning_tokens=int(stats.get("reasoning_output_tokens", 0) or 0),
            latency=latency,
            schema_transport=schema_transport,
            schema_note="LM Studio native API with reasoning=off",
        )

    def run(self, case: BenchmarkCase) -> ModelReply:
        if self.provider == "fm":
            return self._request_fm(case)
        if self.provider == "lmstudio" and not case.tools:
            return self._request_lmstudio(case)
        if not case.response_schema:
            return self._request(case, native_schema=False)
        if self.structured_mode == "native":
            return self._request(case, native_schema=True)
        if self.structured_mode == "prompt":
            return self._request(case, native_schema=False)
        try:
            return self._request(case, native_schema=True)
        except Exception as error:
            fallback = self._request(case, native_schema=False)
            fallback.schema_note = f"native_failed:{type(error).__name__}:{error}"
            return fallback


def _score_groups(text: str, groups: list[list[str]], notes: list[str]) -> tuple[int, int]:
    earned = 0
    for alternatives in groups:
        if _contains_group(text, alternatives):
            earned += 1
        else:
            notes.append(f"missing:{'|'.join(alternatives)}")
    return earned, len(groups)


def _score_forbidden(text: str, forbidden: list[str] | list[list[str]], notes: list[str]) -> tuple[int, int]:
    if not forbidden:
        return 0, 0
    hits = []
    for item in forbidden:
        alternatives = item if isinstance(item, list) else [item]
        if _contains_group(text, alternatives):
            hits.append("|".join(alternatives))
    if hits:
        notes.append("forbidden:" + ",".join(hits))
        return 0, 1
    return 1, 1


def score_routing(case: BenchmarkCase, reply: ModelReply) -> Score:
    notes: list[str] = []
    try:
        data = _json_object(reply.content)
    except Exception as error:
        return Score(False, 0, 2 + len(case.expected["anchors"]), [f"invalid_json:{error}"])
    earned = 0
    possible = 1
    if data.get("action") == case.expected["action"]:
        earned += 1
    else:
        notes.append(f"action:{data.get('action')}!=expected:{case.expected['action']}")
    if case.expected["action"] == "web_search":
        possible += 1
        if data.get("depth") == case.expected["depth"]:
            earned += 1
        else:
            notes.append(f"depth:{data.get('depth')}!=expected:{case.expected['depth']}")
        target = data.get("query", "")
    elif case.expected["action"] == "execute_bash":
        target = data.get("command", "")
    else:
        target = reply.content
    group_score, group_total = _score_groups(target, case.expected["anchors"], notes)
    earned += group_score
    possible += group_total
    return Score(not notes, earned, possible, notes)


def _decoded_tool(reply: ModelReply) -> tuple[str | None, dict[str, Any], list[str]]:
    notes: list[str] = []
    if not reply.tool_calls:
        return None, {}, notes
    if len(reply.tool_calls) != 1:
        notes.append(f"tool_call_count:{len(reply.tool_calls)}")
    first = reply.tool_calls[0]
    try:
        arguments = json.loads(first["arguments"])
        if not isinstance(arguments, dict):
            raise ValueError("arguments are not an object")
    except Exception as error:
        notes.append(f"invalid_arguments:{error}")
        arguments = {}
    return first["name"], arguments, notes


def score_tool(case: BenchmarkCase, reply: ModelReply) -> Score:
    notes: list[str] = []
    name, arguments, parse_notes = _decoded_tool(reply)
    notes.extend(parse_notes)
    expected = case.expected
    if expected["tool"] is None:
        if name is not None:
            notes.append(f"unexpected_tool:{name}")
        forbidden_score, forbidden_total = _score_forbidden(
            json.dumps(arguments, ensure_ascii=False), expected.get("forbidden", []), notes
        )
        text_score, text_total = _score_groups(
            reply.content, expected.get("required_text", []), notes
        )
        possible = 1 + forbidden_total + text_total
        earned = int(name is None) + forbidden_score + text_score
        return Score(not notes, earned, possible, notes)
    earned = int(name == expected["tool"])
    possible = 1
    if name != expected["tool"]:
        notes.append(f"tool:{name}!=expected:{expected['tool']}")
    target = str(arguments.get(expected.get("arg", ""), ""))
    group_score, group_total = _score_groups(target, expected.get("anchors", []), notes)
    forbidden_score, forbidden_total = _score_forbidden(target, expected.get("forbidden", []), notes)
    earned += group_score + forbidden_score
    possible += group_total + forbidden_total
    for key, allowed in expected.get("argument_allowed", {}).items():
        possible += 1
        if arguments.get(key) in allowed:
            earned += 1
        else:
            notes.append(f"argument:{key}={arguments.get(key)} not in {allowed}")
    return Score(not notes, earned, possible, notes)


def score_extraction(case: BenchmarkCase, reply: ModelReply) -> Score:
    notes: list[str] = []
    expected = case.expected
    possible = 3 + len(expected["required"])
    try:
        data = _json_object(reply.content)
    except Exception as error:
        return Score(False, 0, possible, [f"invalid_json:{error}"])
    facts = data.get("facts")
    if not isinstance(facts, list):
        return Score(False, 0, possible, ["facts_not_array"])
    fact_text = "\n".join(
        f"{fact.get('claim', '')} {fact.get('evidence', '')} {fact.get('published_at', '')}"
        for fact in facts if isinstance(fact, dict)
    )
    claim_text = "\n".join(
        str(fact.get("claim", ""))
        for fact in facts if isinstance(fact, dict)
    )
    earned = 0
    if data.get("insufficient_information") is expected["insufficient"]:
        earned += 1
    else:
        notes.append("wrong_insufficient_information")
    if len(facts) >= expected["min_facts"] and (not expected["insufficient"] or not facts):
        earned += 1
    else:
        notes.append(f"fact_count:{len(facts)}")
    source = _norm(expected["source"])
    evidence_ok = True
    for fact in facts:
        evidence = _norm(fact.get("evidence", "")) if isinstance(fact, dict) else ""
        if not evidence or evidence not in source:
            evidence_ok = False
            break
    if evidence_ok:
        earned += 1
    else:
        notes.append("unsupported_evidence")
    group_score, _ = _score_groups(fact_text, expected["required"], notes)
    earned += group_score
    forbidden_score, forbidden_total = _score_forbidden(claim_text, expected["forbidden"], notes)
    earned += forbidden_score
    possible += forbidden_total
    return Score(not notes, earned, possible, notes)


def score_text(case: BenchmarkCase, reply: ModelReply) -> Score:
    notes: list[str] = []
    text = reply.content.strip()
    expected = case.expected
    earned = 0
    possible = 2
    if text and not text.startswith("```"):
        earned += 1
    else:
        notes.append("empty_or_code_fence")
    try:
        decoded = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        decoded = None
    if not isinstance(decoded, (dict, list)):
        earned += 1
    else:
        notes.append("raw_json")
    group_score, group_total = _score_groups(text, expected.get("required", []), notes)
    earned += group_score
    possible += group_total
    forbidden_score, forbidden_total = _score_forbidden(text, expected.get("forbidden", []), notes)
    earned += forbidden_score
    possible += forbidden_total
    if expected.get("language") == "ru":
        possible += 1
        if len(CYRILLIC.findall(text)) >= 5:
            earned += 1
        else:
            notes.append("language_not_russian")
    if expected.get("require_partial"):
        possible += 1
        if PARTIAL_WORDS.search(text):
            earned += 1
        else:
            notes.append("partial_gap_hidden")
    if "max_chars" in expected:
        possible += 1
        if len(text) <= expected["max_chars"]:
            earned += 1
        else:
            notes.append(f"too_long:{len(text)}>{expected['max_chars']}")
    return Score(not notes, earned, possible, notes)


def score_recovery(case: BenchmarkCase, reply: ModelReply) -> Score:
    expected = case.expected
    if expected["tool"] is None:
        pseudo = BenchmarkCase(
            id=case.id, suite=case.suite, messages=case.messages,
            expected={"tool": None, "forbidden": []},
        )
        tool_score = score_tool(pseudo, reply)
        text_case = BenchmarkCase(
            id=case.id, suite=case.suite, messages=case.messages,
            expected={"required": expected["required"]},
        )
        text_score = score_text(text_case, reply)
        notes = tool_score.notes + text_score.notes
        return Score(not notes, tool_score.earned + text_score.earned,
                     tool_score.possible + text_score.possible, notes)
    notes: list[str] = []
    name, arguments, parse_notes = _decoded_tool(reply)
    notes.extend(parse_notes)
    earned = int(name == expected["tool"])
    possible = 1
    if name != expected["tool"]:
        notes.append(f"tool:{name}!=expected:{expected['tool']}")
    target = str(arguments.get(expected["arg"], ""))
    group_score, group_total = _score_groups(target, expected.get("anchors", []), notes)
    earned += group_score
    possible += group_total
    if "allowed" in expected:
        possible += 1
        if target in expected["allowed"]:
            earned += 1
        else:
            notes.append(f"not_allowed:{target}")
    if "not_equal" in expected:
        possible += 1
        if target != expected["not_equal"]:
            earned += 1
        else:
            notes.append("identical_retry")
    forbidden_score, forbidden_total = _score_forbidden(target, expected.get("forbidden", []), notes)
    earned += forbidden_score
    possible += forbidden_total
    return Score(not notes, earned, possible, notes)


def score_case(case: BenchmarkCase, reply: ModelReply) -> Score:
    if case.suite == "routing":
        return score_routing(case, reply)
    if case.suite == "tools":
        return score_tool(case, reply)
    if case.suite == "extraction":
        return score_extraction(case, reply)
    if case.suite in {"finalization", "compact"}:
        return score_text(case, reply)
    if case.suite == "recovery":
        return score_recovery(case, reply)
    raise ValueError(f"Unknown suite: {case.suite}")


def _percentile(values: list[float], ratio: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(len(ordered) * ratio) - 1))
    return ordered[index]


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["suite"]].append(row)

    suites = {}
    for suite, items in grouped.items():
        valid = [item for item in items if not item.get("error")]
        latencies = [item["latency"] for item in valid]
        suites[suite] = {
            "passed": sum(bool(item["passed"]) for item in valid),
            "total": len(items),
            "errors": len(items) - len(valid),
            "pass_rate": round(sum(bool(item["passed"]) for item in valid) / max(1, len(items)), 4),
            "score": round(
                sum(item["earned"] for item in valid)
                / max(1, sum(item["possible"] for item in valid)), 4
            ),
            "p50_latency": round(statistics.median(latencies), 3) if latencies else 0,
            "p95_latency": round(_percentile(latencies, 0.95), 3),
            "completion_tokens": sum(item["completion_tokens"] for item in valid),
            "reasoning_tokens": sum(item["reasoning_tokens"] for item in valid),
            "native_schema": sum(item["schema_transport"] == "native" for item in valid),
            "prompt_schema_fallback": sum(item["schema_transport"] == "prompt" for item in valid),
        }
    valid_rows = [row for row in rows if not row.get("error")]
    thresholds = {
        "routing": 0.8,
        "tools": 0.75,
        "extraction": 0.7,
        "finalization": 0.75,
        "recovery": 0.75,
        "compact": 0.75,
    }
    gates = {
        suite: suites.get(suite, {}).get("pass_rate", 0) >= threshold
        for suite, threshold in thresholds.items()
    }
    return {
        "passed": sum(bool(row["passed"]) for row in valid_rows),
        "total": len(rows),
        "errors": len(rows) - len(valid_rows),
        "score": round(
            sum(row["earned"] for row in valid_rows)
            / max(1, sum(row["possible"] for row in valid_rows)), 4
        ),
        "agent_ready": all(gates.values()) and len(valid_rows) == len(rows),
        "quality_gates": gates,
        "suites": suites,
    }


def _safe_model_name(model: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", model).strip("_") or "model"


def _start_managed_fm_server(base_url: str) -> subprocess.Popen[str]:
    parsed = urlparse(base_url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 1976
    process = subprocess.Popen(
        ["/usr/bin/fm", "serve", "--host", host, "--port", str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
        text=True,
    )
    health_url = f"http://{host}:{port}/health"
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("managed fm serve exited before becoming ready")
        try:
            with urlopen(health_url, timeout=0.5) as response:
                if response.status == 200:
                    return process
        except Exception:
            time.sleep(0.1)
    process.terminate()
    process.wait(timeout=2)
    raise TimeoutError("managed fm serve did not become ready")


def _stop_managed_fm_server(process: subprocess.Popen[str] | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate any OpenAI-compatible model for this agent.")
    parser.add_argument("--base-url", default=os.getenv("LLM_BASE_URL", "http://127.0.0.1:1976/v1"))
    parser.add_argument("--api-key", default=os.getenv("LLM_API_KEY", "local"))
    parser.add_argument("--model", default=os.getenv("LOCAL_MODEL", "system"))
    parser.add_argument("--provider", choices=["openai", "fm", "lmstudio"], default="openai")
    parser.add_argument(
        "--managed-fm-server",
        action="store_true",
        help="Start a fresh fm serve process for every case and stop it afterwards.",
    )
    parser.add_argument("--suite", action="append", choices=sorted(VALID_SUITES), help="Repeat to select suites.")
    parser.add_argument("--structured-mode", choices=["auto", "native", "prompt"], default="auto")
    parser.add_argument("--schema-dialect", choices=["auto", "standard", "afm"], default="auto")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--system-suffix",
        default="",
        help="Append text to the system instruction of every benchmark case.",
    )
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-tokens", type=int, default=None, help="Override every case output budget.")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--limit-per-suite", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.repeat < 1:
        raise SystemExit("--repeat must be >= 1")
    selected = cases_for(set(args.suite) if args.suite else None)
    if args.limit_per_suite is not None:
        counts: dict[str, int] = defaultdict(int)
        limited = []
        for case in selected:
            if counts[case.suite] < args.limit_per_suite:
                limited.append(case)
                counts[case.suite] += 1
        selected = limited

    output = args.output
    if output is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output = Path("benchmark-results") / f"{_safe_model_name(args.model)}_{stamp}.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)

    def make_client() -> BenchmarkClient:
        return BenchmarkClient(
            base_url=args.base_url,
            api_key=args.api_key,
            model=args.model,
            structured_mode=args.structured_mode,
            temperature=args.temperature,
            timeout=args.timeout,
            max_tokens_override=args.max_tokens,
            schema_dialect=args.schema_dialect,
            provider=args.provider,
            system_suffix=args.system_suffix,
        )

    client = make_client()
    rows: list[dict[str, Any]] = []
    total = len(selected) * args.repeat
    index = 0
    for repeat in range(1, args.repeat + 1):
        for case in selected:
            index += 1
            server = None
            try:
                if args.managed_fm_server:
                    server = _start_managed_fm_server(args.base_url)
                    client = make_client()
                reply = client.run(case)
                score = score_case(case, reply)
                row = {
                    "case_id": case.id,
                    "suite": case.suite,
                    "repeat": repeat,
                    "passed": score.passed,
                    "earned": score.earned,
                    "possible": score.possible,
                    "notes": score.notes,
                    **asdict(reply),
                    "error": "",
                }
            except Exception as error:
                row = {
                    "case_id": case.id,
                    "suite": case.suite,
                    "repeat": repeat,
                    "passed": False,
                    "earned": 0,
                    "possible": 0,
                    "notes": [],
                    "content": "",
                    "tool_calls": [],
                    "finish_reason": "",
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "reasoning_tokens": 0,
                    "latency": 0.0,
                    "schema_transport": "none",
                    "schema_note": "",
                    "error": f"{type(error).__name__}: {error}",
                }
            finally:
                _stop_managed_fm_server(server)
            rows.append(row)
            with output.open("w", encoding="utf-8") as handle:
                for item in rows:
                    handle.write(json.dumps(item, ensure_ascii=False) + "\n")
            state = "PASS" if row["passed"] else "ERROR" if row["error"] else "FAIL"
            print(
                f"[{index:03d}/{total:03d}] {case.suite}/{case.id} {state} "
                f"{row['earned']}/{row['possible']} {row['latency']:.2f}s",
                flush=True,
            )

    summary = summarize(rows)
    summary_path = output.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps({
            "model": args.model,
            "provider": args.provider,
            "base_url": args.base_url,
            "structured_mode": args.structured_mode,
            "schema_dialect": client.schema_dialect,
            "temperature": args.temperature,
            "system_suffix": args.system_suffix,
            "repeat": args.repeat,
            "managed_fm_server": args.managed_fm_server,
            "created_at": datetime.now(timezone.utc).isoformat(),
            **summary,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Raw results: {output}")
    print(f"Summary: {summary_path}")
    return 0 if summary["errors"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
