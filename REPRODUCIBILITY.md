# Llama reproduction guide

Run commands from the repository root. Python 3.10 or 3.11 and Java 21 are
recommended.

## 1. Environment and required inputs

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

For a 4 GB NVIDIA GPU (including the verified RTX 3050 Ti), install
`bitsandbytes` and use the 4-bit command shown below:

```bash
python3 -m pip install bitsandbytes
```

Required credentials:

```bash
export COHERE_API_KEY='...'       # CO_API_KEY is also accepted
export OPENAI_API_KEY='...'       # only the unchanged GPT-4o Nuggetizer
export HF_TOKEN='...'             # if Hugging Face requests authentication
```

Accept the Meta model terms if Hugging Face requires it, then make sure
`meta-llama/Llama-3.2-3B-Instruct` can be downloaded. The first BM25 use pulls
Pyserini's `msmarco-v2.1-doc-segmented` prebuilt index. The first Cohere use
downloads `Cohere/trec-rag-2024-index` and corpus shards into `index_cache/`;
budget substantial disk space (the vector index alone is about 15 GB).

The repository does not redistribute these official TREC 2024 assessment
files. Place them at the shown paths, or pass another path explicitly:

```text
data/hr_scored_nist_nuggets_20241218_rag24.test_qrels_nist.jsonl
data/qrels.rag24.raggy-dev.txt
```

The first is required for Nuggetizer. The second is required for retrieval
metrics used by consolidation/oracle analysis. Java discovery normally works;
if needed, set `JAVA_HOME` to the JDK root.

Do not regenerate query variants. Confirm all 31 x 56 checked-in inputs:

```bash
python3 scripts/verify_llama_pipeline.py --stage queries
```

## 2. Incremental one-query smoke

Use the same qid for both retrievers:

```bash
SMOKE_QID=2024-105741
mkdir -p querygym/smoke/retrieval querygym/smoke/retrieval_cohere

python3 querygym/retrieve_all_queries.py \
  --test-file topics.original.txt --qid "$SMOKE_QID" --k 100 \
  --output-dir querygym/smoke/retrieval

python3 querygym/retrieve_all_queries_cohere.py \
  --test-file topics.original.txt --qid "$SMOKE_QID" --k 100 --batch-size 32 \
  --cache-dir index_cache --output-dir querygym/smoke/retrieval_cohere
```

For this original-query smoke, `matching` and `original` are identical. Create
both exact Top-5 inputs:

```bash
python3 scripts/convert_to_ragnarok_format.py \
  --run-file querygym/smoke/retrieval/run.original.txt \
  --output-dir querygym/smoke/rag_prepared/retrieval \
  --query-text-source matching --qid "$SMOKE_QID" --k 5

python3 scripts/convert_to_ragnarok_format.py \
  --run-file querygym/smoke/retrieval_cohere/run.original.txt \
  --output-dir querygym/smoke/rag_prepared/retrieval_cohere \
  --query-text-source matching --qid "$SMOKE_QID" --k 5
```

Validate without loading a model, then generate one answer with the required
3B model:

```bash
python3 querygym/run_llama_generator.py \
  --input-dirs querygym/smoke/rag_prepared/retrieval \
               querygym/smoke/rag_prepared/retrieval_cohere \
  --output-dirs querygym/smoke/rag_results/retrieval \
                querygym/smoke/rag_results/retrieval_cohere \
  --validate-only

python3 querygym/run_llama_generator.py \
  --input-dirs querygym/smoke/rag_prepared/retrieval \
  --output-dirs querygym/smoke/rag_results/retrieval \
  --model meta-llama/Llama-3.2-3B-Instruct \
  --qid "$SMOKE_QID" --max-files 1 --max-records 1 \
  --device auto --dtype float16 --load-in-4bit --batch-size 1
```

Evaluate the generated smoke answer with the unchanged evaluator (the nugget
file may contain all 56 qids; only the present answer is processed):

```bash
python3 querygym/run_rag_nuggetizer.py \
  --rag-results-dir querygym/smoke/rag_results/retrieval \
  --nugget-file data/hr_scored_nist_nuggets_20241218_rag24.test_qrels_nist.jsonl \
  --output-dir querygym/smoke/rag_nuggetized_eval/retrieval \
  --model gpt-4o --max-files 1
```

## 3. Full 56 x 31 pipeline

### Retrieval

Both commands resume complete qids. `--no-resume` deliberately starts the
selected outputs over. Cohere groups up to 32 queries into each embed request
instead of making 1,736 individual requests.

```bash
python3 querygym/retrieve_all_queries.py --k 100

python3 querygym/retrieve_all_queries_cohere.py \
  --k 100 --batch-size 32 --cache-dir index_cache

python3 scripts/verify_llama_pipeline.py --stage retrieval
```

### Explicit provenance decision and Top-5 preparation

Set this only after deciding which interpretation of the public provenance
conflict the run represents. `matching` writes each variant's reformulation;
`original` writes the original TREC question into every prepared variant.

```bash
export QUERY_TEXT_SOURCE=matching   # or: original

python3 scripts/convert_to_ragnarok_format.py \
  --retrieval-dir querygym/retrieval \
  --output-dir querygym/rag_prepared/retrieval \
  --query-text-source "$QUERY_TEXT_SOURCE" --k 5

python3 scripts/convert_to_ragnarok_format.py \
  --retrieval-dir querygym/retrieval_cohere \
  --output-dir querygym/rag_prepared/retrieval_cohere \
  --query-text-source "$QUERY_TEXT_SOURCE" --k 5

python3 scripts/verify_llama_pipeline.py --stage prepared
```

Preparation copies the stored MS MARCO passage object and the exact first five
ranked doc IDs. It neither reranks nor changes candidate order. Completed files
are skipped; use `--overwrite` only when intentionally changing provenance.

### Local Llama generation

```bash
python3 querygym/run_llama_generator.py \
  --input-dirs querygym/rag_prepared/retrieval \
               querygym/rag_prepared/retrieval_cohere \
  --output-dirs querygym/rag_results/retrieval \
                querygym/rag_results/retrieval_cohere \
  --model meta-llama/Llama-3.2-3B-Instruct \
  --device auto --dtype float16 --load-in-4bit --batch-size 1 \
  --context-size 8192 --max-new-tokens 1024

python3 scripts/verify_llama_pipeline.py --stage generated
```

The shown 4-bit configuration is the appropriate starting point for 4 GB VRAM.
On a larger GPU, omit `--load-in-4bit` and use `--dtype float16` or
`--dtype bfloat16`. Increase `--batch-size` only if memory permits. Prompts are
never silently truncated: a request exceeding `--context-size` fails with its
qid. Output is checkpointed after every generation batch; rerunning skips
existing topic IDs.

### Paper-consistent Nuggetizer

```bash
python3 querygym/run_rag_nuggetizer.py \
  --rag-results-dir querygym/rag_results \
  --nugget-file data/hr_scored_nist_nuggets_20241218_rag24.test_qrels_nist.jsonl \
  --output-dir querygym/rag_nuggetized_eval \
  --model gpt-4o

python3 scripts/verify_llama_pipeline.py --stage scores
```

This is an expensive OpenAI stage and resumes by qid. Do not pass a Llama model:
the wrapper rejects it. Score files contain per-qid and aggregate `all_score`
and `strict_vital_score` values.

### Retrieval metrics, QPP, oracle, and Utility-Gap analysis

The existing analysis remains in place; rerun it on the new retrieval/generation
artifacts as follows:

```bash
python3 querygym/evaluate_retrieval_per_query.py \
  --qrels data/qrels.rag24.raggy-dev.txt

python3 querygym/qpp/run_pre_retrieval_verbose.py \
  --queries-dir querygym/queries --output-dir querygym/qpp \
  --index msmarco-v2.1-doc-segmented

python3 querygym/qpp/run_qpp_querygym.py \
  --mode post --queries_dir querygym/queries \
  --retrieval_dirs querygym/retrieval querygym/retrieval_cohere \
  --output_dir querygym/qpp --index_path msmarco-v2.1-doc-segmented \
  --k_top 100

python3 querygym/consolidate_query_data.py \
  --rag-score-tag llama_3_2_3b_instruct_top5
python3 querygym/analyze_qpp_correlations.py
python3 querygym/analyze_qpp_oracle_performance.py
```

Optional BERT-QPP/QSDQPP inputs are loaded from their existing locations when
present; missing optional predictors remain absent rather than blocking the
base analysis. The consolidation keeps retrieval metrics, QPP values, oracle
selection, and downstream Utility-Gap inputs while switching only the answer
score tag.

Final structural audit:

```bash
python3 scripts/verify_llama_pipeline.py --stage all
```

Success means two retrievers x 31 variants x 56 topics, top-100 retrieval,
exact Top-5 preparation/references, preserved `query.text`, and both requested
nugget score fields for every topic.

## 4. Offline developer checks

These do not download indexes or call paid APIs:

```bash
python3 -m unittest discover -s tests -v
python3 querygym/run_llama_generator.py \
  --input-dirs tests/fixtures/rag_prepared/retrieval \
  --output-dirs /tmp/qpp4rag-validate --validate-only
python3 nuggetizer/scripts/calculate_metrics.py \
  --input_file tests/fixtures/nuggetizer/smoke_assignments.jsonl \
  --output_file /tmp/qpp4rag-smoke-scores.jsonl
```

The unit tests use fake retrievers to confirm BM25 checkpoint behavior and that
multiple Cohere queries use one embed call. They also assert exact Top-5/order,
query provenance, Llama result shape, citation bounds, and nugget metrics.
