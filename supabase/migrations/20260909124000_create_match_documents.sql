-- Top-k cosine search over contract_chunks.
--
-- PostgREST cannot express "order by embedding <=> $1", so vector search is
-- exposed as a Postgres function and called from Python via supabase.rpc().
--
-- No ANN index on purpose: pgvector sequential scan returns exact (100% recall)
-- neighbours and needs no maintenance at the current corpus size. An HNSW index
-- becomes worthwhile only around ~1M rows.
create or replace function public.match_documents(
    query_embedding extensions.vector(1024),
    match_count int default 5
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
    order by contract_chunks.embedding <=> query_embedding
    limit greatest(match_count, 0);
end;
$$;
