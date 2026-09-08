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
