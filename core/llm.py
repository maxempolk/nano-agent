import json

from openai import OpenAI
from openai.types.chat import ChatCompletion


def _completion(content: str, model: str, *, completion_id: str = "local-response",
                created: int = 0, finish_reason: str = "stop",
                tool_calls: list | None = None) -> ChatCompletion:
    message: dict = {
        "role": "assistant",
        "content": content,
    }
    if tool_calls:
        message["tool_calls"] = tool_calls

    return ChatCompletion.model_validate({
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "finish_reason": finish_reason,
            "message": message,
        }],
    })


def _text_completion(content: str, model: str) -> ChatCompletion:
    return _completion(content, model)


def _sse_completion(response: str, model: str) -> ChatCompletion | None:
    chunks: list[dict] = []
    for line in response.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if not data or data == "[DONE]":
            continue
        try:
            chunk = json.loads(data)
        except json.JSONDecodeError:
            continue
        if isinstance(chunk, dict):
            chunks.append(chunk)

    if not chunks:
        return None

    content_parts: list[str] = []
    tool_calls: dict[int, dict] = {}
    finish_reason = "stop"
    completion_id = "local-response"
    created = 0
    response_model = model

    for chunk in chunks:
        completion_id = chunk.get("id") or completion_id
        created = chunk.get("created") or created
        response_model = chunk.get("model") or response_model

        choices = chunk.get("choices")
        if not isinstance(choices, list) or not choices:
            continue
        choice = choices[0]
        if not isinstance(choice, dict):
            continue
        finish_reason = choice.get("finish_reason") or finish_reason
        delta = choice.get("delta")
        if not isinstance(delta, dict):
            continue

        content = delta.get("content")
        if isinstance(content, str):
            content_parts.append(content)

        for call in delta.get("tool_calls") or []:
            if not isinstance(call, dict):
                continue
            index = call.get("index", len(tool_calls))
            current = tool_calls.setdefault(index, {
                "id": "",
                "type": "function",
                "function": {"name": "", "arguments": ""},
            })
            if call.get("id"):
                current["id"] = call["id"]
            if call.get("type"):
                current["type"] = call["type"]
            function = call.get("function")
            if isinstance(function, dict):
                current["function"]["name"] += function.get("name") or ""
                current["function"]["arguments"] += function.get("arguments") or ""

    normalized_tool_calls = []
    for index in sorted(tool_calls):
        call = tool_calls[index]
        call["id"] = call["id"] or f"local-call-{index}"
        normalized_tool_calls.append(call)

    return _completion(
        "".join(content_parts),
        response_model,
        completion_id=completion_id,
        created=created,
        finish_reason=finish_reason,
        tool_calls=normalized_tool_calls,
    )


def _normalize_completion(response, model: str) -> ChatCompletion:
    if hasattr(response, "choices"):
        return response

    if isinstance(response, str):
        streamed = _sse_completion(response, model)
        if streamed is not None:
            return streamed
        try:
            decoded = json.loads(response)
        except json.JSONDecodeError:
            return _text_completion(response, model)
        if isinstance(decoded, str):
            return _text_completion(decoded, model)
        response = decoded

    if isinstance(response, dict):
        if "choices" in response:
            return ChatCompletion.model_validate(response)

        content = response.get("content") or response.get("response")
        message = response.get("message")
        if not content and isinstance(message, dict):
            content = message.get("content")
        if isinstance(content, str):
            return _text_completion(content, model)

    raise TypeError(f"Неподдерживаемый формат ответа LLM: {type(response).__name__}")


def call_llm(client: OpenAI, model: str, messages: list,
             tools: list | None = None,
             response_format: dict | None = None) -> ChatCompletion:
    kwargs: dict = {"model": model, "messages": messages}
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"
    if response_format:
        kwargs["response_format"] = response_format
    response = client.chat.completions.create(**kwargs)
    return _normalize_completion(response, model)
