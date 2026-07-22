#!/usr/bin/env python3
"""Test script for agent interaction."""

from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
from core.agent import Agent
from core.config import APPLE_BASE_URL, APPLE_LOCAL_MODEL
from core.tools.web_search import WebSearchTool
from core.tools import bash

# Create client
client = OpenAI(base_url=APPLE_BASE_URL, api_key="apple-local")

# Create web search tool
web_search = WebSearchTool(client, APPLE_LOCAL_MODEL, model_mini=APPLE_LOCAL_MODEL)

# Create agent
system = """You are an autonomous AI agent. Use the provided tools to complete tasks.
Reply in the user's language. Use Russian for Russian input.
Be concise."""

agent = Agent(
    client,
    APPLE_LOCAL_MODEL,
    system,
    max_tool_output=2000,
    extra_tools=[web_search],
)

print("Агент запущен!")
print("=" * 50)

# Test queries
queries = [
    "Привет!",
    "Кто ты?",
    "Какая сейчас погода в Москве?",
]

for query in queries:
    print(f"\nВы: {query}")

    def on_tool_call(name, args, result):
        print(f"  [инструмент] {name}")
        print(f"  [результат] {result.strip()[:200]}...")

    reply = agent.run_turn(query, on_tool_call=on_tool_call)
    print(f"Агент: {reply[:500]}")
    print("-" * 50)
