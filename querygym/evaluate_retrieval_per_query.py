#!/usr/bin/env python3
"""
Evaluate retrieval results per query.
Computes NDCG@10 and Recall@100 for each query in each run file.
"""

import sys
from pathlib import Path
import argparse
import json
from tqdm import tqdm

_REPO = Path(__file__).resolve().parent.parent

try:
    import pytrec_eval
except ImportError:
    print("❌ pytrec_eval not found. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pytrec_eval"])
    import pytrec_eval


def load_qrels(qrels_file):
    """Load qrels file."""
    qrels = {}
    with open(qrels_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split()
                if len(parts) >= 4:
                    query_id, iter_num, doc_id, relevance = parts[0], parts[1], parts[2], parts[3]
                    if query_id not in qrels:
                        qrels[query_id] = {}
                    qrels[query_id][doc_id] = int(relevance)
    return qrels


def load_run_file(run_file):
    """Load run file in TREC format."""
    run = {}
    with open(run_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split()
                if len(parts) >= 6:
                    query_id, _, doc_id, rank, score, run_name = parts[0], parts[1], parts[2], parts[3], parts[4], parts[5]
                    if query_id not in run:
                        run[query_id] = {}
                    run[query_id][doc_id] = float(score)
    return run


def evaluate_per_query(run_file, qrels, metrics=None):
    """Evaluate a single run file and return per-query metrics."""
    if metrics is None:
        metrics = ['ndcg_cut_5', 'ndcg_cut_10', 'recall_100']
    
    try:
        run = load_run_file(run_file)
        
        # Create evaluator
        evaluator = pytrec_eval.RelevanceEvaluator(qrels, set(metrics))
        
        # Evaluate
        results = evaluator.evaluate(run)
        
        # Convert to list of records
        per_query_results = []
        for qid in sorted(results.keys()):
            record = {'qid': qid}
            for metric in metrics:
                if metric in results[qid]:
                    record[metric] = results[qid][metric]
                else:
                    record[metric] = 0.0
            per_query_results.append(record)
        
        return per_query_results
    except Exception as e:
        print(f"   ⚠️  Error evaluating {run_file.name}: {e}")
        return None


def get_run_name_from_file(run_file):
    """Extract run name from filename (e.g., run.genqr_trial1.txt -> genqr_trial1)."""
    name = run_file.stem  # Remove .txt
    if name.startswith('run.'):
        return name[4:]  # Remove 'run.' prefix
    return name


def main():
    parser = argparse.ArgumentParser(description='Evaluate retrieval results per query')
    parser.add_argument('--qrels', type=str,
                       default=str(_REPO / 'data/2024-retrieval-qrels.txt'),
                       help='Path to qrels file')
    parser.add_argument('--retrieval-dir', type=str,
                       default=str(_REPO / 'querygym/retrieval'),
                       help='Directory with BM25 (pyserini) run files')
    parser.add_argument('--cohere-dir', type=str,
                       default=str(_REPO / 'querygym/retrieval_cohere'),
                       help='Directory with Cohere run files')
    parser.add_argument('--output-dir', type=str,
                       default=str(_REPO / 'querygym/retrieval_eval'),
                       help='Output directory for evaluation results')
    
    args = parser.parse_args()
    
    # Load qrels file
    print("📖 Loading qrels file...")
    qrels_file = Path(args.qrels)
    
    if not qrels_file.exists():
        print(f"❌ Qrels file not found: {qrels_file}")
        return
    
    qrels = load_qrels(qrels_file)
    print(f"✅ Loaded qrels: {len(qrels)} queries")
    
    # Find all run files
    retrieval_dir = Path(args.retrieval_dir)
    cohere_dir = Path(args.cohere_dir)
    output_dir = Path(args.output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not retrieval_dir.exists():
        print(f"❌ Retrieval directory not found: {retrieval_dir}")
        return
    
    if not cohere_dir.exists():
        print(f"❌ Cohere directory not found: {cohere_dir}")
        return
    
    retrieval_files = sorted(retrieval_dir.glob("run.*.txt"))
    cohere_files = sorted(cohere_dir.glob("run.*.txt"))
    
    print(f"📁 Found {len(retrieval_files)} BM25 run files")
    print(f"📁 Found {len(cohere_files)} Cohere run files")
    
    # Create mapping from run name to files
    retrieval_map = {get_run_name_from_file(f): f for f in retrieval_files}
    cohere_map = {get_run_name_from_file(f): f for f in cohere_files}
    
    # Get all unique run names
    all_run_names = sorted(set(list(retrieval_map.keys()) + list(cohere_map.keys())))
    
    print(f"\n📊 Evaluating {len(all_run_names)} run files per query...")
    print("=" * 80)
    
    # Evaluate each run file
    successful = 0
    failed = 0
    
    for run_name in tqdm(all_run_names, desc="Evaluating runs", unit="run"):
        print(f"\n📄 Evaluating: {run_name}")
        
        # Evaluate BM25
        if run_name in retrieval_map:
            print(f"   BM25: {retrieval_map[run_name].name}")
            bm25_results = evaluate_per_query(retrieval_map[run_name], qrels)
            if bm25_results:
                # Save to JSONL file
                output_file = output_dir / f"retrieval_{run_name}_per_query.jsonl"
                with open(output_file, 'w') as f:
                    for record in bm25_results:
                        f.write(json.dumps(record) + '\n')
                
                # Calculate averages for display
                avg_ndcg5 = sum(r['ndcg_cut_5'] for r in bm25_results) / len(bm25_results) if bm25_results else 0.0
                avg_ndcg10 = sum(r['ndcg_cut_10'] for r in bm25_results) / len(bm25_results) if bm25_results else 0.0
                avg_recall = sum(r['recall_100'] for r in bm25_results) / len(bm25_results) if bm25_results else 0.0
                print(f"      ✅ Saved {len(bm25_results)} queries | Avg NDCG@5: {avg_ndcg5:.6f}, Avg NDCG@10: {avg_ndcg10:.6f}, Avg Recall@100: {avg_recall:.6f}")
                print(f"      📄 Saved to: {output_file.name}")
                successful += 1
            else:
                print(f"      ❌ Evaluation failed")
                failed += 1
        else:
            print(f"   ⚠️  BM25 file not found")
        
        # Evaluate Cohere
        if run_name in cohere_map:
            print(f"   Cohere: {cohere_map[run_name].name}")
            cohere_results = evaluate_per_query(cohere_map[run_name], qrels)
            if cohere_results:
                # Save to JSONL file
                output_file = output_dir / f"retrieval_cohere_{run_name}_per_query.jsonl"
                with open(output_file, 'w') as f:
                    for record in cohere_results:
                        f.write(json.dumps(record) + '\n')
                
                # Calculate averages for display
                avg_ndcg5 = sum(r['ndcg_cut_5'] for r in cohere_results) / len(cohere_results) if cohere_results else 0.0
                avg_ndcg10 = sum(r['ndcg_cut_10'] for r in cohere_results) / len(cohere_results) if cohere_results else 0.0
                avg_recall = sum(r['recall_100'] for r in cohere_results) / len(cohere_results) if cohere_results else 0.0
                print(f"      ✅ Saved {len(cohere_results)} queries | Avg NDCG@5: {avg_ndcg5:.6f}, Avg NDCG@10: {avg_ndcg10:.6f}, Avg Recall@100: {avg_recall:.6f}")
                print(f"      📄 Saved to: {output_file.name}")
                successful += 1
            else:
                print(f"      ❌ Evaluation failed")
                failed += 1
        else:
            print(f"   ⚠️  Cohere file not found")
    
    print("\n" + "=" * 80)
    print("📊 EVALUATION SUMMARY")
    print("=" * 80)
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"📁 Results saved to: {output_dir}")
    print("\n✅ Evaluation complete!")
    print(f"📄 Each file contains per-query metrics with 'qid', 'ndcg_cut_5', 'ndcg_cut_10', and 'recall_100' fields")


if __name__ == "__main__":
    main()
