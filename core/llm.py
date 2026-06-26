from openai import OpenAI


def call_llm(client: OpenAI, model: str, messages: list, tools: list | None = None):
    kwargs: dict = {"model": model, "messages": messages}
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"
    return client.chat.completions.create(**kwargs)
