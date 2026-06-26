import trafilatura
from ddgs import DDGS
from openai import OpenAI

from core.llm import call_llm

MAX_RESULTS = 10
MAX_SCRAPE = 3
MAX_CONTENT_CHARS = 2000

SCHEMA = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web and return summarized content from the most relevant pages.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"}
            },
            "required": ["query"]
        }
    }
}


class WebSearchTool:
    SCHEMA = SCHEMA

    def __init__(self, client: OpenAI, model: str, model_mini: str | None = None):
        self.client = client
        self.model = model
        self.model_mini = model_mini or model

    def _optimize_query(self, query: str) -> str:
        response = call_llm(self.client, self.model_mini, [{"role": "user", "content":
            f"Convert to a short English search engine query (3-6 keywords, no punctuation).\n"
            f"Reply with ONLY the query in English, nothing else.\n"
            f"Input: {query}"
        }])
        return response.choices[0].message.content.strip() or query

    def _format_results(self, results: list[dict]) -> str:
        rows = ""
        for i, r in enumerate(results):
            rows += (
                f"[{i+1}] {r['title']}\n"
                f"    URL: {r['href']}\n"
                f"    Snippet: {r['body']}\n\n"
            )
        return rows

    def _pick_relevant(self, formatted: str, results: list[dict]) -> list[int]:
        prompt = (
            f"Select up to {MAX_SCRAPE} most informative sources from the list below.\n"
            f"Output ONLY their numbers, comma-separated. No other text.\n"
            f"Example output: 1, 3, 5\n\n"
            f"Sources ({len(results)} total):\n{formatted}"
        )
        response = call_llm(self.client, self.model_mini, [{"role": "user", "content": prompt}])
        raw = response.choices[0].message.content or ""

        # Валидация — извлекаем только цифры
        indices = []
        for part in raw.split(","):
            part = part.strip()
            if part.isdigit():
                idx = int(part)
                if 1 <= idx <= len(results):
                    indices.append(idx - 1)  # переводим в 0-based

        # Fallback — если модель сломала формат, берём первые MAX_SCRAPE
        if not indices:
            indices = list(range(min(MAX_SCRAPE, len(results))))

        return indices[:MAX_SCRAPE]

    def _scrape(self, url: str) -> str:
        try:
            downloaded = trafilatura.fetch_url(url)
            text = trafilatura.extract(downloaded) or ""
            if len(text) > MAX_CONTENT_CHARS:
                text = text[:MAX_CONTENT_CHARS] + f"\n... [обрезано]"
            return text or "Не удалось извлечь текст."
        except Exception as e:
            return f"Ошибка парсинга: {e}"

    def execute(self, query: str) -> str:
        # 1. Оптимизируем запрос
        self.last_query = self._optimize_query(query)
        query = self.last_query

        # 2. Поиск
        with DDGS() as ddg:
            results = ddg.text(query, max_results=MAX_RESULTS)
        if not results:
            return "Ничего не найдено."

        # 3. Форматируем и выбираем релевантные
        formatted = self._format_results(results)
        indices = self._pick_relevant(formatted, results)

        # 4. Парсим выбранные сайты
        parts = []
        for i in indices:
            url = results[i]["href"]
            content = self._scrape(url)
            parts.append(f"[Source: {url}]\n{content}")

        return "\n\n---\n\n".join(parts)

if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    from openai import OpenAI
    from core.config import MODEL, MODEL_MINI

    load_dotenv()
    client = OpenAI(base_url='https://api.groq.com/openai/v1', api_key=os.environ['API_TOKEN'])
    tool = WebSearchTool(client, MODEL, model_mini=MODEL_MINI)
    result = tool.execute('Что такое абоба?')
    print(result)