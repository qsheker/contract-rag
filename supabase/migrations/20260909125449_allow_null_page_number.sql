-- DOCX and TXT carry no page boundaries, so their chunks are stored with a
-- null page_number rather than an invented one (see loader.load_document).
--
-- The existing `check (page_number >= 1)` constraint needs no change: in
-- Postgres a CHECK that evaluates to NULL passes, so it keeps rejecting 0 and
-- negative numbers while allowing NULL.
alter table public.contract_chunks
    alter column page_number drop not null;
