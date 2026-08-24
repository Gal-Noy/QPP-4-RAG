#!/usr/bin/env python3
"""
Evaluate retrieval results from both BM25 (pyserini) and Cohere methods.
Computes NDCG@10 for each run file and outputs a summary.
"""

import sys
from pathlib import Path
import argparse

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


def evaluate_run_file(run_file, qrels, metrics=None):
    """Evaluate a single run file and return metrics."""
    if metrics is None:
        metrics = ['ndcg_cut_10', 'recall_100']
    
    try:
        run = load_run_file(run_file)
        
        # Create evaluator
        evaluator = pytrec_eval.RelevanceEvaluator(qrels, set(metrics))
        
        # Evaluate
        results = evaluator.evaluate(run)
        
        # Get average for each metric
        avg_metrics = {}
        for metric in metrics:
            metric_scores = [results[qid][metric] for qid in results if metric in results[qid]]
            avg_metrics[metric] = sum(metric_scores) / len(metric_scores) if metric_scores else 0.0
        
        return avg_metrics
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
    parser = argparse.ArgumentParser(description='Evaluate retrieval results from BM25 and Cohere')
    parser.add_argument('--raggy-qrels', type=str,
                       default=str(_REPO / 'data/2024-retrieval-qrels.txt'),
                       help='Path to official TREC 2024 retrieval qrels file')
    parser.add_argument('--umbrella-qrels', type=str,
                       default=str(_REPO / 'data/qrels.rag24.umbrella.txt'),
                       help='Path to umbrella qrels file')
    parser.add_argument('--retrieval-dir', type=str,
                       default=str(_REPO / 'querygym/retrieval'),
                       help='Directory with BM25 (pyserini) run files')
    parser.add_argument('--cohere-dir', type=str,
                       default=str(_REPO / 'querygym/retrieval_cohere'),
                       help='Directory with Cohere run files')
    parser.add_argument('--output', type=str,
                       default=str(_REPO / 'querygym/evaluation_results.txt'),
                       help='Output file for evaluation results')
    
    args = parser.parse_args()
    
    # Load both qrels files
    print("📖 Loading qrels files...")
    raggy_qrels_file = Path(args.raggy_qrels)
    umbrella_qrels_file = Path(args.umbrella_qrels)
    
    if not raggy_qrels_file.exists():
        print(f"❌ Raggy qrels file not found: {raggy_qrels_file}")
        return
    
    if not umbrella_qrels_file.exists():
        print(f"❌ Umbrella qrels file not found: {umbrella_qrels_file}")
        return
    
    raggy_qrels = load_qrels(raggy_qrels_file)
    umbrella_qrels = load_qrels(umbrella_qrels_file)
    print(f"✅ Loaded raggy qrels: {len(raggy_qrels)} queries")
    print(f"✅ Loaded umbrella qrels: {len(umbrella_qrels)} queries")
    
    # Find all run files
    retrieval_dir = Path(args.retrieval_dir)
    cohere_dir = Path(args.cohere_dir)
    
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
    
    print(f"\n📊 Evaluating {len(all_run_names)} run files...")
    print("=" * 80)
    
    # Evaluate each run file with both qrels
    results = []
    
    for run_name in all_run_names:
        print(f"\n📄 Evaluating: {run_name}")
        
        result = {'run_name': run_name}
        
        # Evaluate BM25 with raggy qrels
        if run_name in retrieval_map:
            print(f"   BM25 (raggy): {retrieval_map[run_name].name}")
            bm25_raggy = evaluate_run_file(retrieval_map[run_name], raggy_qrels)
            if bm25_raggy:
                result['bm25_raggy_ndcg10'] = bm25_raggy.get('ndcg_cut_10', None)
                result['bm25_raggy_recall100'] = bm25_raggy.get('recall_100', None)
                print(f"      NDCG@10: {result['bm25_raggy_ndcg10']:.6f}, Recall@100: {result['bm25_raggy_recall100']:.6f}")
            else:
                result['bm25_raggy_ndcg10'] = None
                result['bm25_raggy_recall100'] = None
                print(f"   ❌ BM25 (raggy) evaluation failed")
        else:
            result['bm25_raggy_ndcg10'] = None
            result['bm25_raggy_recall100'] = None
            print(f"   ⚠️  BM25 file not found")
        
        # Evaluate BM25 with umbrella qrels
        if run_name in retrieval_map:
            print(f"   BM25 (umbrella): {retrieval_map[run_name].name}")
            bm25_umbrella = evaluate_run_file(retrieval_map[run_name], umbrella_qrels)
            if bm25_umbrella:
                result['bm25_umbrella_ndcg10'] = bm25_umbrella.get('ndcg_cut_10', None)
                result['bm25_umbrella_recall100'] = bm25_umbrella.get('recall_100', None)
                print(f"      NDCG@10: {result['bm25_umbrella_ndcg10']:.6f}, Recall@100: {result['bm25_umbrella_recall100']:.6f}")
            else:
                result['bm25_umbrella_ndcg10'] = None
                result['bm25_umbrella_recall100'] = None
                print(f"   ❌ BM25 (umbrella) evaluation failed")
        else:
            result['bm25_umbrella_ndcg10'] = None
            result['bm25_umbrella_recall100'] = None
        
        # Evaluate Cohere with raggy qrels
        if run_name in cohere_map:
            print(f"   Cohere (raggy): {cohere_map[run_name].name}")
            cohere_raggy = evaluate_run_file(cohere_map[run_name], raggy_qrels)
            if cohere_raggy:
                result['cohere_raggy_ndcg10'] = cohere_raggy.get('ndcg_cut_10', None)
                result['cohere_raggy_recall100'] = cohere_raggy.get('recall_100', None)
                print(f"      NDCG@10: {result['cohere_raggy_ndcg10']:.6f}, Recall@100: {result['cohere_raggy_recall100']:.6f}")
            else:
                result['cohere_raggy_ndcg10'] = None
                result['cohere_raggy_recall100'] = None
                print(f"   ❌ Cohere (raggy) evaluation failed")
        else:
            result['cohere_raggy_ndcg10'] = None
            result['cohere_raggy_recall100'] = None
            print(f"   ⚠️  Cohere file not found")
        
        # Evaluate Cohere with umbrella qrels
        if run_name in cohere_map:
            print(f"   Cohere (umbrella): {cohere_map[run_name].name}")
            cohere_umbrella = evaluate_run_file(cohere_map[run_name], umbrella_qrels)
            if cohere_umbrella:
                result['cohere_umbrella_ndcg10'] = cohere_umbrella.get('ndcg_cut_10', None)
                result['cohere_umbrella_recall100'] = cohere_umbrella.get('recall_100', None)
                print(f"      NDCG@10: {result['cohere_umbrella_ndcg10']:.6f}, Recall@100: {result['cohere_umbrella_recall100']:.6f}")
            else:
                result['cohere_umbrella_ndcg10'] = None
                result['cohere_umbrella_recall100'] = None
                print(f"   ❌ Cohere (umbrella) evaluation failed")
        else:
            result['cohere_umbrella_ndcg10'] = None
            result['cohere_umbrella_recall100'] = None
        
        results.append(result)
    
    # Write results
    output_file = Path(args.output)
    print(f"\n💾 Writing results to: {output_file}")
    
    with open(output_file, 'w') as f:
        # Header - 8 metrics per run file
        f.write("run_name\t")
        f.write("bm25_raggy_ndcg10\tbm25_raggy_recall100\t")
        f.write("cohere_raggy_ndcg10\tcohere_raggy_recall100\t")
        f.write("bm25_umbrella_ndcg10\tbm25_umbrella_recall100\t")
        f.write("cohere_umbrella_ndcg10\tcohere_umbrella_recall100\n")
        
        # Results
        for result in results:
            f.write(f"{result['run_name']}\t")
            # BM25 raggy
            f.write(f"{result['bm25_raggy_ndcg10']:.6f}\t" if result['bm25_raggy_ndcg10'] is not None else "N/A\t")
            f.write(f"{result['bm25_raggy_recall100']:.6f}\t" if result['bm25_raggy_recall100'] is not None else "N/A\t")
            # Cohere raggy
            f.write(f"{result['cohere_raggy_ndcg10']:.6f}\t" if result['cohere_raggy_ndcg10'] is not None else "N/A\t")
            f.write(f"{result['cohere_raggy_recall100']:.6f}\t" if result['cohere_raggy_recall100'] is not None else "N/A\t")
            # BM25 umbrella
            f.write(f"{result['bm25_umbrella_ndcg10']:.6f}\t" if result['bm25_umbrella_ndcg10'] is not None else "N/A\t")
            f.write(f"{result['bm25_umbrella_recall100']:.6f}\t" if result['bm25_umbrella_recall100'] is not None else "N/A\t")
            # Cohere umbrella
            f.write(f"{result['cohere_umbrella_ndcg10']:.6f}\t" if result['cohere_umbrella_ndcg10'] is not None else "N/A\t")
            f.write(f"{result['cohere_umbrella_recall100']:.6f}\n" if result['cohere_umbrella_recall100'] is not None else "N/A\n")
    
    # Print summary
    print("\n" + "=" * 120)
    print("📊 EVALUATION SUMMARY")
    print("=" * 120)
    print(f"{'Run Name':<25} {'BM25 Raggy':<20} {'Cohere Raggy':<20} {'BM25 Umbrella':<20} {'Cohere Umbrella':<20}")
    print(f"{'':<25} {'NDCG@10':<10} {'R@100':<10} {'NDCG@10':<10} {'R@100':<10} {'NDCG@10':<10} {'R@100':<10} {'NDCG@10':<10} {'R@100':<10}")
    print("-" * 120)
    
    for result in results:
        print(f"{result['run_name']:<25}", end="")
        # BM25 raggy
        bm25_r_ndcg = f"{result['bm25_raggy_ndcg10']:.6f}" if result['bm25_raggy_ndcg10'] is not None else "N/A"
        bm25_r_rec = f"{result['bm25_raggy_recall100']:.6f}" if result['bm25_raggy_recall100'] is not None else "N/A"
        print(f" {bm25_r_ndcg:<9} {bm25_r_rec:<10}", end="")
        # Cohere raggy
        cohere_r_ndcg = f"{result['cohere_raggy_ndcg10']:.6f}" if result['cohere_raggy_ndcg10'] is not None else "N/A"
        cohere_r_rec = f"{result['cohere_raggy_recall100']:.6f}" if result['cohere_raggy_recall100'] is not None else "N/A"
        print(f" {cohere_r_ndcg:<9} {cohere_r_rec:<10}", end="")
        # BM25 umbrella
        bm25_u_ndcg = f"{result['bm25_umbrella_ndcg10']:.6f}" if result['bm25_umbrella_ndcg10'] is not None else "N/A"
        bm25_u_rec = f"{result['bm25_umbrella_recall100']:.6f}" if result['bm25_umbrella_recall100'] is not None else "N/A"
        print(f" {bm25_u_ndcg:<9} {bm25_u_rec:<10}", end="")
        # Cohere umbrella
        cohere_u_ndcg = f"{result['cohere_umbrella_ndcg10']:.6f}" if result['cohere_umbrella_ndcg10'] is not None else "N/A"
        cohere_u_rec = f"{result['cohere_umbrella_recall100']:.6f}" if result['cohere_umbrella_recall100'] is not None else "N/A"
        print(f" {cohere_u_ndcg:<9} {cohere_u_rec:<10}")
    
    print("\n✅ Evaluation complete!")
    print(f"📄 Results saved to: {output_file}")


if __name__ == "__main__":
    main()
