# QPP-4-RAG: Llama reproduction/extension

This fork keeps the paper's 31 checked-in query variants, BM25 and Cohere dense
retrieval, Top-5 contexts, Nuggetizer evaluation, and QPP/oracle analysis. The
answer generator remains `meta-llama/Llama-3.2-3B-Instruct` running locally
through Hugging Face. Only the Nuggetizer assignment judge is changed from
GPT-4o to the local `Qwen/Qwen3-4B-Instruct-2507`; its scores are therefore not
paper-consistent evaluator results.

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
5. `querygym/run_rag_nuggetizer.py`: the vendored Nuggetizer with a local Qwen3
   4B judge. It emits Nugget-All (`all_score`) and Nugget-Strict
   (`strict_vital_score`), plus the other original metrics.
6. Existing QPP, retrieval evaluation, consolidation, correlation, oracle, and
   Utility-Gap analysis scripts consume the new Llama score filenames.

Every expensive stage resumes by default. Retrieval checkpoints complete qid
blocks, preparation skips complete JSON files, Llama checkpoints after every
batch, and Nuggetizer resumes completed qids.

## Official TREC 2024 data

Download the two official NIST inputs and derive the grouped Nuggetizer input:

```bash
mkdir -p data
curl -L https://trec.nist.gov/data/rag/2024-retrieval-qrels.txt \
  -o data/2024-retrieval-qrels.txt
curl -L https://trec.nist.gov/data/rag/nugget_assignment.20241218.jsonl \
  -o data/nugget_assignment.20241218.jsonl
python3 scripts/prepare_trec2024_data.py
```

- `data/2024-retrieval-qrels.txt` is the official retrieval-assessment file
  used by retrieval evaluation. It contains 86 official topics and covers all
  56 experiment qids.
- `data/nugget_assignment.20241218.jsonl` is the official per-run nugget
  assignment export and is the source for conversion only.
- `data/hr_scored_nist_nuggets_20241218_rag24.test_qrels_nist.jsonl` is derived
  locally from that export. Its 56 grouped `{qid, query, nuggets}` records are
  consumed by the Qwen Nuggetizer judge. These fixed NIST-derived nuggets are
  never generated or rewritten by the judge.

## Main experiment: matching `query.text`

The main reproduction uses `--query-text-source matching`. This follows the
methodology described in the paper and the repository's `convert_all_batch.py`:
each reformulated retrieval run is paired with its corresponding reformulated
query file.

```bash
python3 scripts/convert_to_ragnarok_format.py \
  --retrieval-dir querygym/retrieval \
  --output-dir querygym/rag_prepared/retrieval \
  --query-text-source matching --k 5
```

Use the same `matching` setting for Cohere preparation. The Llama generator
then copies the prepared `query.text` without reloading or rewriting it.
`--query-text-source original` is retained only for an optional sensitivity or
ablation run and should write to separate prepared/result directories.

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
querygym/rag_nuggetized_eval/     local Qwen judge assignments and scores
querygym/qpp/                     existing QPP outputs and scripts
```

The upstream paper is [Can QPP Choose the Right Query Variant?](https://arxiv.org/html/2604.22661).
