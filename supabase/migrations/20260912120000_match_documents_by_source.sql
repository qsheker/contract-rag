-- Scope top-k search to a chosen set of documents.
--
-- A chat is about the document that was uploaded into it, but retrieval ran
-- over the whole table: asking about a rented flat could return clauses of an
-- employment contract, and the citation was formally valid while pointing at a
-- document the reader never opened here.
--
-- Recreated rather than overloaded: adding a defaulted parameter to an existing
-- function creates a second signature, and PostgREST then has two candidates
-- for the same named call.
drop function if exists public.match_documents(extensions.vector, int);

create or replace function public.match_documents(
    query_embedding extensions.vector(1024),
    match_count int default 5,
    -- NULL means "every document", which is what a chat with no upload of its
    -- own still wants.
    source_files text[] default null
)
returns table (
    id text,
    text text,
    clause_id text,
    page_number int,
    source_file text,
    detected_strategy text,
    similarity float
)
language plpgsql
stable
-- 001 creates the vector type in the extensions schema; pin it so the operator
-- class resolves regardless of the caller's search_path.
set search_path = public, extensions
as $$
begin
    return query
    select contract_chunks.id,
           contract_chunks.text,
           contract_chunks.clause_id,
           contract_chunks.page_number,
           contract_chunks.source_file,
           contract_chunks.detected_strategy,
           -- The model emits normalised vectors, so cosine distance is the
           -- operator pgvector recommends; 1 - distance yields similarity.
           1 - (contract_chunks.embedding <=> query_embedding) as similarity
    from public.contract_chunks
    -- Filtered before the ordering, so k is k within the scope rather than
    -- whatever survives a filter applied to someone else's top-k.
    where source_files is null
       or contract_chunks.source_file = any(source_files)
    order by contract_chunks.embedding <=> query_embedding
    limit greatest(match_count, 0);
end;
$$;
