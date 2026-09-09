# contract-rag

RAG-пайплайн для юридических договоров с обязательным цитированием пункта и страницы.

## Требования

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)

## Начало работы

```bash
uv sync
uv run pytest
uv run ruff check .
```

## Текущее состояние

Первый корпус публичных шаблонов договоров хранится в `corpus_raw/contracts/`.

Loader извлекает и очищает текст каждого PDF постранично, сохраняя номер страницы
и исходный путь:

```python
from contract_rag.loader import LoaderError, NoTextLayerError, load_pdf

try:
    pages = load_pdf("corpus_raw/contracts/contract_01.pdf")
except NoTextLayerError:
    # OCR намеренно не входит в текущий Loader.
    raise
except LoaderError as error:
    print(error.code, error.source_file)

for page in pages:
    print(page.page_number, page.source_file, page.text[:80])
```

`load_pdf()` удаляет повторяющиеся граничные строки и исправляет переносы слов.
Chunking, определение пунктов, OCR и HTTP API остаются за пределами Loader.

## Разбиение по пунктам

Chunker применяет фиксированный приоритет generic-стратегий без определения типа
договора: dotted numbering, verbose Article/Section, затем ненумерованные заголовки.

```python
from contract_rag.chunker import UnsupportedNumberingError, chunk_by_clause
from contract_rag.loader import load_pdf

pages = load_pdf("corpus_raw/contracts/contract_01.pdf")

try:
    chunks = chunk_by_clause(pages)
except UnsupportedNumberingError:
    # Автоматический token-based fallback намеренно отсутствует.
    raise

for chunk in chunks:
    print(chunk.clause_id, chunk.page_number, chunk.detected_strategy)
```

Преамбула сохраняется отдельным чанком с `clause_id=None`. Если пункт продолжается
на следующей странице, он остаётся одним чанком, привязанным к странице начала.

### Стратегии на эталонном корпусе

| Файл | Результат |
| --- | --- |
| `contract_01.pdf` | `dotted_numbering` |
| `contract_02.pdf` | `verbose_numbering` |
| `contract_03.pdf` | `heading_only` |
| `contract_04.pdf` | `UnsupportedNumberingError` — плоская нумерация вне текущего набора стратегий |
| `contract_05.pdf` | `heading_only` |

## Миграции

Схема базы живёт в `supabase/migrations/` и накатывается Supabase CLI — вручную
через SQL Editor ничего копировать не нужно.

```bash
brew install supabase/tap/supabase
supabase login
supabase link --project-ref <project-ref>
supabase db push
```

`supabase login` и `link` делаются один раз на машину: CLI спрашивает пароль базы
и хранит привязку в `supabase/.temp/`, который исключён из Git. Дальше
`supabase db push` применяет только те миграции, которых ещё нет в базе, и сам
ведёт их учёт.

Обе текущие миграции идемпотентны (`create table if not exists`,
`create or replace function`), поэтому `db push` на уже настроенную базу
безопасен и просто зафиксирует их как применённые.

Новая миграция создаётся так — CLI сам проставит таймстамп в имени, порядок
применения определяется именно им:

```bash
supabase migration new add_something
```

| Файл | Что делает |
| --- | --- |
| `*_create_contract_chunks.sql` | расширение pgvector, таблица `contract_chunks` |
| `*_create_match_documents.sql` | RPC-функция `match_documents` для top-k поиска |

## Эмбеддинги и Supabase

Индексатор использует `ai-forever/ru-en-RoSBERTa` и хранит 1024-мерные векторы
в таблице Supabase `contract_chunks`. Исходный текст и все метаданные чанка
записываются вместе с вектором.

Перед первым запуском:

1. Создайте или выберите проект Supabase.
2. Накатите миграции через Supabase CLI (см. «Миграции» ниже).
3. Создайте локальный `.env` — этот файл исключён из Git:

```bash
cp .env.example .env
```

Все переменные и их назначение описаны в `.env.example`.

Пример индексации уже подготовленного списка чанков:

```python
from contract_rag.embeddings import create_supabase_client_from_env, embed_and_index

supabase = create_supabase_client_from_env()
embed_and_index(chunks, supabase)
```

`embed_and_index()` сам добавляет `search_document: ` перед кодированием и
делает upsert по `id`, поэтому повторный запуск не создаёт дубликаты. В поле
`text` всегда остаётся полный текст. Если вход длиннее 512 токенов, только вход
модели автоматически обрезается, а лог содержит warning с исходным файлом и
номером пункта.

Базовый ID имеет формат `source_file::clause_id` или
`source_file::preamble`. В многоязычных документах номер пункта может повторяться.
Чтобы не перезаписать один язык другим, коллизии получают детерминированный суффикс
страницы и порядкового номера.

## Поиск по векторам

`retrieve()` кодирует запрос с префиксом `search_query: ` и вызывает Postgres-функцию
`match_documents` через `supabase.rpc()`. Обычный клиент PostgREST не умеет строить
`order by embedding <=> ...`, поэтому сортировка живёт в SQL, а Python только
передаёт вектор и `k`.

```python
from contract_rag.retriever import retrieve

for hit in retrieve("Куда передаются неразрешённые споры?", k=5):
    print(hit.similarity, hit.chunk.clause_id, hit.chunk.page_number, hit.chunk.source_file)
```

Возвращается `RetrievedChunk`: целый `Chunk` со всеми метаданными плюс `chunk_id`
из базы и `similarity`. Текст чанка не обрезается и не переформулируется.
Список отсортирован по убыванию `similarity`, длина — не больше `k`.
`k <= 0` — это `ValueError`; пустая таблица — пустой список, а не исключение.

Клиент можно внедрить явно — так же, как в индексаторе, и так же тестировать
фейком без сети:

```python
from contract_rag.retriever import SupabaseRetriever

hits = SupabaseRetriever(supabase_client).retrieve("payment schedule", k=3)
```

### Решения

- **Косинус (`<=>`).** Модель отдаёт нормализованные векторы — для них pgvector
  рекомендует именно косинусное расстояние. `similarity = 1 - distance`.
- **ANN-индекса нет.** На десятках чанков sequential scan даёт точный результат
  (recall 100%) и не требует обслуживания. HNSW окупается ближе к ~1 млн строк.
- **Reranking не входит в retrieval.** Он добавляется поверх, отдельным решением.

### Качество на mini eval-сете

Разметка и скрипт замера — в `eval/`. Метрика: доля вопросов, у которых ожидаемый
пункт попал в top-k. Замер на 12 размеченных вопросах (5 ru/kk, 7 en) по корпусу
из 210 проиндексированных чанков:

| Метрика | Результат |
| --- | --- |
| recall@1 | 8/12 = 0.67 |
| recall@3 | 11/12 = 0.92 |
| recall@5 | 12/12 = 1.00 |

Сет маленький и составлен по содержимому уже проиндексированного корпуса, поэтому
`recall@5 = 1.00` означает «на текущем масштабе retrieval не теряет ни одного
языка», а не «поиск решён». Число нужно как база для сравнения: любое будущее
изменение (reranking, другая модель, другой chunking) сверяется с ним.

## Генерация ответа

`generate_answer()` собирает ответ строго по переданным чанкам и обязан подкрепить
каждое утверждение цитатой на пункт и страницу.

```python
from contract_rag.generator import generate_answer
from contract_rag.retriever import retrieve

hits = retrieve("Куда передаются неразрешённые споры?", k=5)
answer = generate_answer("Куда передаются неразрешённые споры?", [hit.chunk for hit in hits])

print(answer.text)
for citation in answer.citations:
    print(citation.source_file, citation.clause_id, citation.page_number)
```

Ответ приходит как схема, а не как свободный текст: `Answer.text` плюс
`Answer.citations` из `Citation(clause_id, page_number, source_file)`. Если в
переданных чанках ответа нет, модель обязана сказать это прямо и вернуть пустой
список цитат — выдумывать запрещено промптом.

### Смена провайдера

Провайдер задаётся строкой модели в `GENERATION_MODEL`, вызов идёт через
`litellm.completion()` — один и тот же код работает с Ollama, Anthropic, OpenAI,
Gemini, Bedrock и десятками других. Переключение — это правка `.env`, а не кода:

| `GENERATION_MODEL` | Что нужно ещё |
| --- | --- |
| `ollama/qwen2.5:7b` | установленный Ollama и `ollama pull qwen2.5:7b` |
| `anthropic/claude-sonnet-4-6` | `ANTHROPIC_API_KEY` в `.env` |
| `openai/gpt-4o` | `OPENAI_API_KEY` в `.env` |

Значения по умолчанию у переменной **нет**: без неё код падает с явной ошибкой
конфигурации. Молчаливый фолбэк на локальную модель, которая может быть не
запущена, превратил бы ошибку настройки в непонятный таймаут.

Локальная модель — единственный вариант, при котором тексты договоров не покидают
машину. Перед переключением на платный API это стоит учитывать отдельно от цены.

### Что считается ошибкой

- Провайдер недоступен, вернул не-JSON или payload не по схеме → `GenerationError`.
  Отката на разбор свободного текста нет: он бы вернул те самые ответы без
  проверяемых цитат, ради которых всё и затевалось.
- **Цитата на пункт, которого модели не давали, → `GenerationError`.** Совпадение
  проверяется по тройке `source_file` + `clause_id` + `page_number`. Выдуманная
  ссылка на пункт договора хуже, чем отсутствие ответа.

### Качество

`eval/run_generation_eval.py` считает на тех же 12 вопросах: доля ответов с
цитатой, доля попаданий в ожидаемый пункт, доля отказов «не найдено» и доля
отклонённых ответов. Эти метрики считаются без judge-модели; оценка
faithfulness — отдельная задача.

```bash
uv run python eval/run_generation_eval.py
```
