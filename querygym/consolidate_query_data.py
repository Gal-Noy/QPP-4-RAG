#!/usr/bin/env python3
"""
Consolidate query reformulation data, nugget evaluation scores, and QPP metrics
into a single JSON file.
"""

import json
import csv
import argparse
from pathlib import Path
from collections import defaultdict

def load_original_queries(queries_file):
    """Load original queries from topics file."""
    queries = {}
    with open(queries_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t', 1)
            if len(parts) == 2:
                query_id, query_text = parts
                queries[query_id] = query_text
    return queries

def load_nugget_scores(jsonl_file):
    """Load nugget evaluation scores from JSONL file."""
    scores = {}
    if not Path(jsonl_file).exists():
        return scores
    
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            qid = data.get('qid')
            if qid and qid != 'all':  # Skip aggregate 'all' entry
                scores[qid] = {
                    'strict_vital_score': data.get('strict_vital_score'),
                    'strict_all_score': data.get('strict_all_score'),
                    'vital_score': data.get('vital_score'),
                    'all_score': data.get('all_score')
                }
    return scores

def load_qpp_metrics(csv_file):
    """Load QPP metrics from CSV file."""
    metrics = {}
    if not Path(csv_file).exists():
        return metrics
    
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            query_id = row.get('query_id')
            if query_id:
                # Convert all values to float where possible
                qpp_data = {}
                for key, value in row.items():
                    if key != 'query_id':
                        try:
                            qpp_data[key] = float(value) if value else None
                        except (ValueError, TypeError):
                            qpp_data[key] = value
                metrics[query_id] = qpp_data
    return metrics

def load_reformulated_queries(queries_file):
    """Load reformulated queries from topics file."""
    queries = {}
    if not Path(queries_file).exists():
        return queries
    
    with open(queries_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t', 1)
            if len(parts) == 2:
                query_id, query_text = parts
                queries[query_id] = query_text
    return queries

def load_retrieval_metrics(jsonl_file):
    """Load per-query NDCG@5, NDCG@10, and Recall@100 metrics."""
    metrics = {}
    if not Path(jsonl_file).exists():
        return metrics
    
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            qid = data.get('qid')
            if qid and qid != 'all':  # Skip aggregate 'all' entry
                metrics[qid] = {
                    'ndcg@5': data.get('ndcg_cut_5'),
                    'ndcg@10': data.get('ndcg_cut_10'),
                    'recall@100': data.get('recall_100')
                }
    return metrics

def get_reformulation_methods_and_trials():
    """Get all reformulation methods and trials."""
    methods = ['original', 'genqr', 'genqr_ensemble', 'mugi', 'qa_expand', 'query2doc', 'query2e']
    trials = {}
    for method in methods:
        if method == 'original':
            trials[method] = [None]  # Original has no trial
        else:
            trials[method] = [1, 2, 3, 4, 5]
    return methods, trials

def load_bert_qpp_scores(json_file):
    """Load BERT-QPP scores from JSON file, organized by query, method, and retrieval method."""
    bert_scores = {}
    if not Path(json_file).exists():
        return bert_scores
    
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for key, value in data.items():
            qid = value.get('qid')
            method = value.get('method')
            retrieval_method = value.get('retrieval_method', 'pyserini')  # Default to pyserini for backward compatibility
            
            if qid and method:
                # Create key: method_trial format
                if method == 'original':
                    method_key = 'original_None'
                else:
                    # Extract trial from method name (e.g., "genqr_trial1" -> ("genqr", 1))
                    parts = method.split('_trial')
                    if len(parts) == 2:
                        method_key = f"{parts[0]}_{parts[1]}"
                    else:
                        method_key = f"{method}_None"
                
                if qid not in bert_scores:
                    bert_scores[qid] = {}
                if method_key not in bert_scores[qid]:
                    bert_scores[qid][method_key] = {}
                
                bert_scores[qid][method_key][retrieval_method] = {
                    'bert_qpp_cross_encoder': value.get('bert_qpp_cross_encoder'),
                    'bert_qpp_bi_encoder': value.get('bert_qpp_bi_encoder')
                }
    return bert_scores

def load_rv_scores(json_file):
    """Load RV scores from JSON file, organized by query and method."""
    rv_scores = {}
    if not Path(json_file).exists():
        return rv_scores
    
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for qid, reform_scores in data.items():
            rv_scores[qid] = {}
            for method_key, scores in reform_scores.items():
                rv_scores[qid][method_key] = {
                    'rv_bert': scores.get('rv_bert'),
                    'rv_e5': scores.get('rv_e5')
                }
    return rv_scores

def load_qsdqpp_scores(qsdqpp_dir):
    """Load QSDQPP predicted nDCG scores from QSDQPP directory, organized by query and method."""
    qsdqpp_scores = {}
    qsdqpp_path = Path(qsdqpp_dir)
    
    if not qsdqpp_path.exists():
        return qsdqpp_scores
    
    # Find all predicted_ndcg files (pre-retrieval)
    qsdqpp_files = list(qsdqpp_path.glob("topics.*_predicted_ndcg.txt"))
    
    for qsdqpp_file in qsdqpp_files:
        # Extract method name from filename
        # Format: topics.{method}_predicted_ndcg.txt
        filename = qsdqpp_file.name
        method_name = filename.replace('topics.', '').replace('_predicted_ndcg.txt', '')
        
        # Load scores from file
        with open(qsdqpp_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            # Skip header
            for line in lines[1:]:
                line = line.strip()
                if not line:
                    continue
                parts = line.split('\t')
                if len(parts) >= 2:
                    query_id = parts[0]
                    try:
                        predicted_ndcg = float(parts[1])
                    except (ValueError, TypeError):
                        continue
                    
                    if query_id not in qsdqpp_scores:
                        qsdqpp_scores[query_id] = {}
                    qsdqpp_scores[query_id][method_name] = predicted_ndcg
    
    return qsdqpp_scores

def load_qsd_post_scores(qsd_post_dir):
    """Load post-retrieval QSDQPP scores from qsd_post directory, organized by query, method, and retrieval method."""
    qsd_post_scores = {}
    qsd_post_path = Path(qsd_post_dir)
    
    if not qsd_post_path.exists():
        return qsd_post_scores
    
    # Find all qsd_post files
    qsd_post_files = list(qsd_post_path.glob("qsd_post.*.txt"))
    
    for qsd_post_file in qsd_post_files:
        # Extract method name and retrieval method from filename
        # Format: qsd_post.{method}.{retrieval_method}.txt
        # retrieval_method is either 'bm25' (pyserini) or 'cohere'
        filename = qsd_post_file.name
        # Remove 'qsd_post.' prefix and '.txt' suffix
        name_part = filename.replace('qsd_post.', '').replace('.txt', '')
        
        # Split by last dot to separate method and retrieval_method
        parts = name_part.rsplit('.', 1)
        if len(parts) == 2:
            method_name = parts[0]
            retrieval_method_str = parts[1]
            
            # Map bm25 to pyserini
            if retrieval_method_str == 'bm25':
                retrieval_method = 'pyserini'
            elif retrieval_method_str == 'cohere':
                retrieval_method = 'cohere'
            else:
                continue
            
            # Load scores from file (no header, just query_id \t score)
            with open(qsd_post_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        query_id = parts[0]
                        try:
                            score = float(parts[1])
                        except (ValueError, TypeError):
                            continue
                        
                        if query_id not in qsd_post_scores:
                            qsd_post_scores[query_id] = {}
                        if method_name not in qsd_post_scores[query_id]:
                            qsd_post_scores[query_id][method_name] = {}
                        qsd_post_scores[query_id][method_name][retrieval_method] = score
    
    return qsd_post_scores

def consolidate_data(
    queries_file,
    queries_dir,
    retrieval_scores_dir,
    retrieval_cohere_scores_dir,
    qpp_dir,
    retrieval_eval_dir,
    bert_qpp_file,
    qsdqpp_dir,
    generationonly_scores_dir,
    output_file,
    rag_score_tag
):
    """Consolidate all data into a single JSON file."""
    
    print("📖 Loading original queries...")
    original_queries = load_original_queries(queries_file)
    print(f"✅ Loaded {len(original_queries)} queries")
    
    print("📖 Loading BERT-QPP scores...")
    bert_qpp_scores = load_bert_qpp_scores(bert_qpp_file)
    print(f"✅ Loaded BERT-QPP scores for {len(bert_qpp_scores)} queries")
    
    print("📖 Loading RV scores...")
    rv_scores_file = qpp_dir / "rv_scores.json"
    rv_scores = load_rv_scores(rv_scores_file)
    print(f"✅ Loaded RV scores for {len(rv_scores)} queries")
    
    print("📖 Loading QSDQPP scores (pre-retrieval)...")
    qsdqpp_scores = load_qsdqpp_scores(qsdqpp_dir)
    print(f"✅ Loaded QSDQPP pre-retrieval scores for {len(qsdqpp_scores)} queries")
    
    print("📖 Loading QSDQPP post-retrieval scores...")
    qsd_post_dir = Path(qsdqpp_dir) / "qsd_post"
    qsd_post_scores = load_qsd_post_scores(qsd_post_dir)
    print(f"✅ Loaded QSDQPP post-retrieval scores for {len(qsd_post_scores)} queries")
    
    # Initialize consolidated data structure
    consolidated = {}
    for query_id, query_text in original_queries.items():
        consolidated[query_id] = {
            'query': query_text,
            'reformulations': []
        }
    
    methods, trials_dict = get_reformulation_methods_and_trials()
    
    print("\n🔄 Processing reformulations...")
    total_combinations = sum(len(trials) for trials in trials_dict.values())
    processed = 0
    
    for method in methods:
        for trial in trials_dict[method]:
            processed += 1
            print(f"  [{processed}/{total_combinations}] Processing {method}" + (f"_trial{trial}" if trial else ""))
            
            # Determine file naming pattern
            if method == 'original':
                query_file = queries_dir / "topics.original.txt"
                nugget_file_retrieval = retrieval_scores_dir / f"rag_results_run.original_{rag_score_tag}_scores.jsonl"
                nugget_file_cohere = retrieval_cohere_scores_dir / f"rag_results_run.original_{rag_score_tag}_scores.jsonl"
                nugget_file_generationonly = generationonly_scores_dir / f"rag_results_run.original_gpt_4o_mini_top3_top0_scores.jsonl"
                qpp_file_pyserini = qpp_dir / f"post_retrieval_original_original_pyserini_qpp_metrics.csv"
                qpp_file_cohere = qpp_dir / f"post_retrieval_original_original_cohere_qpp_metrics.csv"
                pre_qpp_file = qpp_dir / f"pre_retrieval_original_qpp_metrics.csv"
                retrieval_eval_file_pyserini = retrieval_eval_dir / "retrieval_original_per_query.jsonl"
                retrieval_eval_file_cohere = retrieval_eval_dir / "retrieval_cohere_original_per_query.jsonl"
            else:
                query_file = queries_dir / f"topics.{method}_trial{trial}.txt"
                nugget_file_retrieval = retrieval_scores_dir / f"rag_results_run.{method}_trial{trial}_{rag_score_tag}_scores.jsonl"
                nugget_file_cohere = retrieval_cohere_scores_dir / f"rag_results_run.{method}_trial{trial}_{rag_score_tag}_scores.jsonl"
                nugget_file_generationonly = generationonly_scores_dir / f"rag_results_run.{method}_trial{trial}_gpt_4o_mini_top3_top0_scores.jsonl"
                qpp_file_pyserini = qpp_dir / f"post_retrieval_{method}_trial{trial}_{method}_trial{trial}_pyserini_qpp_metrics.csv"
                qpp_file_cohere = qpp_dir / f"post_retrieval_{method}_trial{trial}_{method}_trial{trial}_cohere_qpp_metrics.csv"
                pre_qpp_file = qpp_dir / f"pre_retrieval_{method}_trial{trial}_qpp_metrics.csv"
                retrieval_eval_file_pyserini = retrieval_eval_dir / f"retrieval_{method}_trial{trial}_per_query.jsonl"
                retrieval_eval_file_cohere = retrieval_eval_dir / f"retrieval_cohere_{method}_trial{trial}_per_query.jsonl"
            
            # Load reformulated queries
            reformulated_queries = load_reformulated_queries(query_file)
            
            # Load nugget scores
            nugget_scores_retrieval = load_nugget_scores(nugget_file_retrieval)
            nugget_scores_cohere = load_nugget_scores(nugget_file_cohere)
            nugget_scores_generationonly = load_nugget_scores(nugget_file_generationonly)
            
            # Load QPP metrics (post-retrieval)
            qpp_metrics_pyserini = load_qpp_metrics(qpp_file_pyserini)
            qpp_metrics_cohere = load_qpp_metrics(qpp_file_cohere)
            
            # Load pre-retrieval QPP metrics
            pre_qpp_metrics = load_qpp_metrics(pre_qpp_file)
            
            # Load retrieval evaluation metrics
            retrieval_metrics_pyserini = load_retrieval_metrics(retrieval_eval_file_pyserini)
            retrieval_metrics_cohere = load_retrieval_metrics(retrieval_eval_file_cohere)
            
            # Add data for each query
            for query_id in original_queries.keys():
                # Get BERT-QPP scores for this method/trial
                if method == 'original':
                    bert_key = 'original_None'
                    rv_key = 'original_None'
                    qsdqpp_key = 'original'
                else:
                    bert_key = f"{method}_{trial}"
                    rv_key = f"{method}_{trial}"
                    qsdqpp_key = f"{method}_trial{trial}"
                
                bert_scores_dict = bert_qpp_scores.get(query_id, {}).get(bert_key, {})
                
                # Get QSDQPP scores for this method/trial
                qsdqpp_score = qsdqpp_scores.get(query_id, {}).get(qsdqpp_key)  # Pre-retrieval
                qsd_post_pyserini = qsd_post_scores.get(query_id, {}).get(qsdqpp_key, {}).get('pyserini')  # Post-retrieval
                qsd_post_cohere = qsd_post_scores.get(query_id, {}).get(qsdqpp_key, {}).get('cohere')  # Post-retrieval
                
                # Build qpp_metrics with post-retrieval metrics including BERT-QPP and QSD post-retrieval
                qpp_metrics_pyserini_dict = qpp_metrics_pyserini.get(query_id, {}).copy()
                qpp_metrics_cohere_dict = qpp_metrics_cohere.get(query_id, {}).copy()
                
                # Add BERT-QPP metrics to each retrieval method's qpp_metrics
                bert_pyserini = bert_scores_dict.get('pyserini', {})
                if bert_pyserini:
                    qpp_metrics_pyserini_dict['bert_qpp_cross_encoder'] = bert_pyserini.get('bert_qpp_cross_encoder')
                    qpp_metrics_pyserini_dict['bert_qpp_bi_encoder'] = bert_pyserini.get('bert_qpp_bi_encoder')
                
                bert_cohere = bert_scores_dict.get('cohere', {})
                if bert_cohere:
                    qpp_metrics_cohere_dict['bert_qpp_cross_encoder'] = bert_cohere.get('bert_qpp_cross_encoder')
                    qpp_metrics_cohere_dict['bert_qpp_bi_encoder'] = bert_cohere.get('bert_qpp_bi_encoder')
                
                # Add post-retrieval QSDQPP scores
                if qsd_post_pyserini is not None:
                    qpp_metrics_pyserini_dict['qsd_post_predicted_ndcg'] = qsd_post_pyserini
                if qsd_post_cohere is not None:
                    qpp_metrics_cohere_dict['qsd_post_predicted_ndcg'] = qsd_post_cohere
                
                qpp_metrics = {
                    'pyserini': qpp_metrics_pyserini_dict,
                    'cohere': qpp_metrics_cohere_dict
                }
                
                # Build pre-retrieval QPP metrics including RV and QSDQPP
                pre_retrieval_metrics = {
                    **pre_qpp_metrics.get(query_id, {}),
                    **rv_scores.get(query_id, {}).get(rv_key, {})
                }
                # Add QSDQPP score if available
                if qsdqpp_score is not None:
                    pre_retrieval_metrics['qsdqpp_predicted_ndcg'] = qsdqpp_score
                
                reformulation_data = {
                    'method': method,
                    'trial': trial,
                    'reformulated_query': reformulated_queries.get(query_id, original_queries.get(query_id, '')),
                    'nugget_scores': {
                        'retrieval': nugget_scores_retrieval.get(query_id, {}),
                        'retrieval_cohere': nugget_scores_cohere.get(query_id, {})
                    },
                    'generationonly_nugget_scores': nugget_scores_generationonly.get(query_id, {}),
                    'qpp_metrics': qpp_metrics,
                    'pre_retrieval_qpp_metrics': pre_retrieval_metrics,
                    'retrieval_metrics': {
                        'pyserini': retrieval_metrics_pyserini.get(query_id, {}),
                        'cohere': retrieval_metrics_cohere.get(query_id, {})
                    }
                }
                consolidated[query_id]['reformulations'].append(reformulation_data)
    
    print(f"\n💾 Saving consolidated data to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(consolidated, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Saved {len(consolidated)} queries with reformulation data")
    print(f"📊 Total reformulations per query: {len(consolidated[list(consolidated.keys())[0]]['reformulations'])}")
    
    return consolidated

def main():
    base_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--rag-score-tag',
        default='llama_3_2_3b_instruct_top5',
        help='Generator tag embedded in Nuggetizer score filenames'
    )
    parser.add_argument(
        '--output-file', type=Path,
        default=base_dir / 'consolidated_query_data.json'
    )
    args = parser.parse_args()
    
    queries_file = base_dir / "queries" / "topics.original.txt"
    queries_dir = base_dir / "queries"
    retrieval_scores_dir = base_dir / "rag_nuggetized_eval" / "retrieval" / "scores"
    retrieval_cohere_scores_dir = base_dir / "rag_nuggetized_eval" / "retrieval_cohere" / "scores"
    generationonly_scores_dir = base_dir / "rag_nuggetized_eval_o" / "scores"
    qpp_dir = base_dir / "qpp"
    retrieval_eval_dir = base_dir / "retrieval_eval"
    bert_qpp_file = base_dir / "qpp" / "bert_qpp_results" / "bert_qpp_scores.json"
    qsdqpp_dir = base_dir / "qpp" / "QSDQPP"
    output_file = args.output_file
    
    consolidate_data(
        queries_file,
        queries_dir,
        retrieval_scores_dir,
        retrieval_cohere_scores_dir,
        qpp_dir,
        retrieval_eval_dir,
        bert_qpp_file,
        qsdqpp_dir,
        generationonly_scores_dir,
        output_file,
        args.rag_score_tag
    )
    
    print(f"\n🎉 Done! Output saved to: {output_file}")

if __name__ == "__main__":
    main()
