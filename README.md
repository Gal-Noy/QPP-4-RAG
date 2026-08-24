# QPP-4-RAG: Llama reproduction/extension

This fork keeps the paper's 31 checked-in query variants, BM25 and Cohere dense
retrieval, Top-5 contexts, Nuggetizer evaluation, and QPP/oracle analysis. The
only experimental substitution is the answer generator:
`meta-llama/Llama-3.2-3B-Instruct` running locally through Hugging Face.

The 31 files in `querygym/queries/` are experiment inputs: one original plus
six reformulation methods with five trials each, all with 56 topics. **Do not
run the query-generation scripts for this reproduction.** Verify the inputs at
any time with:

```bash
python3 scripts/verify_llama_pipeline.py --stage queries
```

## Pipeline

1. `querygym/retrieve_all_queries.py`: paper BM25 (`k1=0.9`, `b=0.4`, top 100).
2. `querygym/retrieve_all_queries_cohere.py`: the original
   `Cohere/trec-rag-2024-index`, now batching query embeddings and FAISS search.
3. `scripts/convert_to_ragnarok_format.py`: materialize exactly ranks 1--5 in
   Ragnarok request format.
4. `querygym/run_llama_generator.py`: local Llama 3.2 3B generation with the
   prepared query and five references preserved verbatim/in order.
5. `querygym/run_rag_nuggetizer.py`: the vendored GPT-4o Nuggetizer assigner;
   this evaluator is deliberately not replaced by Llama. It emits Nugget-All
   (`all_score`) and Nugget-Strict (`strict_vital_score`), plus the other
   original metrics.
6. Existing QPP, retrieval evaluation, consolidation, correlation, oracle, and
   Utility-Gap analysis scripts consume the new Llama score filenames.

Every expensive stage resumes by default. Retrieval checkpoints complete qid
blocks, preparation skips complete JSON files, Llama checkpoints after every
batch, and Nuggetizer resumes completed qids.

## Important `query.text` provenance ambiguity

The public repository is inconsistent: its old preparation commands supplied
the original TREC question, while the paper describes generation conditioned
on each reformulated query. This fork does not choose between those claims.
Preparation requires an explicit `--query-text-source matching` or
`--query-text-source original`. The Llama generator then copies the prepared
`query.text` without reloading or rewriting it. Use one declared choice for
both retrievers and record it with the run.

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for environment setup, one-query
smokes, the full 56 x 31 commands, data/API requirements, and verification.
All active pipeline credentials can be placed in a repository-local `.env` by
copying `.env.example`; exported variables remain supported and take priority.

## Key paths

```text
querygym/queries/                 immutable published query variants
querygym/retrieval/               BM25 top-100 runs
querygym/retrieval_cohere/        Cohere dense top-100 runs
querygym/rag_prepared/            Ragnarok Top-5 requests
querygym/rag_results/             local Llama answers
querygym/rag_nuggetized_eval/     GPT Nuggetizer assignments and scores
querygym/qpp/                     existing QPP outputs and scripts
```

The upstream paper is [Can QPP Choose the Right Query Variant?](https://arxiv.org/html/2604.22661).
