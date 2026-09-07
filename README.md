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
