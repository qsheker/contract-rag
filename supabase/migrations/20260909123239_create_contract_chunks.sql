create extension if not exists vector with schema extensions;

create table if not exists public.contract_chunks (
    id text primary key,
    text text not null,
    embedding extensions.vector(1024) not null,
    clause_id text null,
    page_number integer not null check (page_number >= 1),
    source_file text not null,
    detected_strategy text not null
);
