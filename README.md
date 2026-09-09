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

Loader принимает PDF, DOCX и TXT. `load_document()` выбирает загрузчик по
расширению файла; кто знает формат заранее, может звать `load_pdf()`,
`load_docx()` или `load_txt()` напрямую.

```python
from contract_rag.loader import LoaderError, NoTextLayerError, load_document

try:
    pages = load_document("corpus_raw/contracts/contract_01.pdf")
except NoTextLayerError:
    # OCR намеренно не входит в текущий Loader.
    raise
except LoaderError as error:
    print(error.code, error.source_file)

for page in pages:
    print(page.page_number, page.source_file, page.text[:80])
```

| Формат | Страницы | `page_number` |
| --- | --- | --- |
| `.pdf` | реальные границы страниц из файла | `1, 2, 3, …` |
| `.docx` | весь документ — одна «страница» | `None` |
| `.txt` | весь файл — одна «страница» | `None` |

**У DOCX и TXT номера страницы не существует.** У DOCX разбивка появляется только
при рендеринге — она зависит от шрифтов и полей устройства и в файле не хранится;
у TXT понятия страницы нет вовсе. Поэтому `page_number` там `None`, а не выдуманное
число: **цитата по такому документу может содержать пункт, но не страницу.**
Интерфейсу стоит предупреждать об этом при загрузке не-PDF файла.

Расширение с неизвестным суффиксом — это `UnsupportedFormatError`; список
поддерживаемых лежит в `SUPPORTED_EXTENSIONS`.

Все форматы проходят одну и ту же очистку переносов (`join_hyphenation`), поэтому
чанкеру не нужно знать, откуда пришёл текст. Удаление повторяющихся колонтитулов
(`strip_boilerplate`) применяется только к PDF — ему нужно минимум две страницы.
Из DOCX читаются и параграфы, и таблицы в порядке документа: реквизиты сторон и
графики платежей обычно живут именно в таблицах.

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
| `*_allow_null_page_number.sql` | `page_number` становится nullable — для DOCX/TXT |

## Эмбеддинги и Supabase

Индексатор использует `ai-forever/ru-en-RoSBERTa` и хранит 1024-мерные векторы
в таблице Supabase `contract_chunks`. Исходный текст и все метаданные чанка
записываются вместе с вектором.

Перед первым запуском:

1. Создайте или выберите проект Supabase.
2. Накатите миграции через Supabase CLI (см. «Миграции» ниже).
3. Создайте локальный `.env` — этот файл исключён из Git:

```dotenv
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-project-key
```

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
