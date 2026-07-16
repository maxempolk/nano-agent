import trafilatura
from concurrent.futures import ThreadPoolExecutor, as_completed
from ddgs import DDGS
from openai import OpenAI

try:
    from newspaper import Article as NewspaperArticle
    _NEWSPAPER_OK = True
except ImportError:
    _NEWSPAPER_OK = False

from core.llm import call_llm

MAX_RESULTS = 10
MAX_SCRAPE = 3
MAX_CONTENT_CHARS = 4000
MAX_PARAGRAPHS = 30

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

    def _scrape_trafilatura(self, url: str) -> str:
        try:
            downloaded = trafilatura.fetch_url(url)
            return trafilatura.extract(downloaded) or ""
        except Exception:
            return ""

    def _scrape_newspaper(self, url: str) -> str:
        if not _NEWSPAPER_OK:
            return ""
        try:
            article = NewspaperArticle(url)
            article.download()
            article.parse()
            return article.text or ""
        except Exception:
            return ""

    def _smart_truncate(self, text: str, query: str) -> str:
        if len(text) <= MAX_CONTENT_CHARS:
            return text
        paragraphs = [p.strip() for p in text.split("\n") if len(p.strip()) > 40][:MAX_PARAGRAPHS]
        if not paragraphs:
            return text[:MAX_CONTENT_CHARS]
        query_words = set(query.lower().split())
        def score(p: str) -> int:
            return len(set(p.lower().split()) & query_words)
        ranked = sorted(range(len(paragraphs)), key=lambda i: score(paragraphs[i]), reverse=True)
        selected: set[int] = set()
        total = 0
        for i in ranked:
            if total + len(paragraphs[i]) > MAX_CONTENT_CHARS:
                break
            selected.add(i)
            total += len(paragraphs[i])
        return "\n\n".join(paragraphs[i] for i in sorted(selected))

    def _scrape(self, url: str, query: str = "") -> str:
        text = self._scrape_trafilatura(url)
        if len(text) < 200:
            text = self._scrape_newspaper(url) or text
        if not text:
            return "Не удалось извлечь текст."
        return self._smart_truncate(text, query)

    def _extract_facts(self, original_query: str, pages: list[dict]) -> str:
        sources = ""
        for i, p in enumerate(pages):
            sources += f"[{i+1}] {p['url']}\n{p['content']}\n\n"
        prompt = (
            f"Question: {original_query}\n\n"
            f"Extract key facts relevant to the question from each source below.\n"
            f"For each source output: [N] followed by 2-3 bullet points.\n"
            f"Be specific — numbers, dates, names. Skip irrelevant content.\n\n"
            f"{sources.strip()}"
        )
        response = call_llm(self.client, self.model_mini, [{"role": "user", "content": prompt}])
        return response.choices[0].message.content or ""

    def execute(self, query: str) -> str:
        original_query = query

        # 1. Оптимизируем запрос
        self.last_query = self._optimize_query(query)

        # 2. Поиск
        with DDGS() as ddg:
            results = ddg.text(self.last_query, max_results=MAX_RESULTS)
        if not results:
            return "Ничего не найдено."

        # 3. Выбираем релевантные источники
        formatted = self._format_results(results)
        indices = self._pick_relevant(formatted, results)

        # 4. Парсим выбранные сайты параллельно
        urls = [results[i]["href"] for i in indices]
        scraped: dict[str, str] = {}
        with ThreadPoolExecutor(max_workers=len(urls)) as ex:
            futures = {ex.submit(self._scrape, url, original_query): url for url in urls}
            for future in as_completed(futures):
                scraped[futures[future]] = future.result()
        pages = [{"url": url, "content": scraped[url]} for url in urls]

        # 5. Извлекаем ключевые факты батчем
        return self._extract_facts(original_query, pages)

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