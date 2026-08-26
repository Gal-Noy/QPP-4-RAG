#!/usr/bin/env python3
"""
Analyze oracle performance: For each QPP metric, select the variation with highest QPP value
for each query, then report the aggregated nugget scores and retrieval metrics.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

_REPO = Path(__file__).resolve().parent.parent


def _generation_only_scores(reformulation):
    """Read either historical spelling of the optional generation-only field."""
    return (
        reformulation.get('generationonly_nugget_scores')
        or reformulation.get('generation_only_nugget_scores')
        or {}
    )


def _append_available_row(frame, row):
    """Do not add meaningless all-empty oracle rows for unavailable metrics."""
    if not row.get('n_queries'):
        print(f"    Skipped {row.get('qpp_metric')}: no source values found")
        return frame
    return pd.concat([frame, pd.DataFrame([row])], ignore_index=True)

def load_consolidated_data(json_file):
    """Load the consolidated query data."""
    with open(json_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_all_qpp_metrics(data):
    """Get all unique QPP metric names from the data (both post and pre-retrieval, and BERT-QPP)."""
    qpp_metrics = set()
    for query_id, query_data in data.items():
        for reformulation in query_data['reformulations']:
            qpp_metrics_dict = reformulation.get('qpp_metrics', {})
            
            # Check both pyserini and cohere QPP metrics (post-retrieval)
            pyserini_qpp = qpp_metrics_dict.get('pyserini', {})
            cohere_qpp = qpp_metrics_dict.get('cohere', {})
            qpp_metrics.update(pyserini_qpp.keys())
            qpp_metrics.update(cohere_qpp.keys())
            
            # Check pre-retrieval QPP metrics
            pre_qpp = reformulation.get('pre_retrieval_qpp_metrics', {})
            qpp_metrics.update(pre_qpp.keys())
    return sorted(list(qpp_metrics))

def analyze_qpp_oracle_performance(data, retrieval_method='pyserini'):
    """
    For each QPP metric, select the variation with highest QPP value for each query,
    then aggregate the performance metrics.
    
    Args:
        data: Consolidated query data
        retrieval_method: 'pyserini' or 'cohere'
    
    Returns:
        DataFrame with one row per QPP metric showing aggregated performance
    """
    results = []
    
    # Get all QPP metrics
    all_qpp_metrics = get_all_qpp_metrics(data)
    
    # Filter QPP metrics - check both post-retrieval, pre-retrieval, and BERT-QPP
    qpp_metrics_to_analyze = []
    for qpp_metric in all_qpp_metrics:
        # Check if this metric exists (post-retrieval for specified method, pre-retrieval, or BERT-QPP)
        found = False
        is_pre_retrieval = False
        is_bert_qpp = False
        
        for query_id, query_data in data.items():
            for reformulation in query_data['reformulations']:
                qpp_metrics_dict = reformulation.get('qpp_metrics', {})
                
                # Check BERT-QPP metrics (now inside retrieval method's qpp_metrics)
                if qpp_metric.startswith('bert_qpp_'):
                    retrieval_qpp = qpp_metrics_dict.get(retrieval_method, {})
                    if qpp_metric in retrieval_qpp and retrieval_qpp.get(qpp_metric) is not None:
                        found = True
                        is_bert_qpp = True
                        break
                
                # Check post-retrieval metrics
                qpp_metrics = qpp_metrics_dict.get(retrieval_method, {})
                if qpp_metric in qpp_metrics:
                    found = True
                    break
                
                # Check pre-retrieval metrics
                pre_qpp_metrics = reformulation.get('pre_retrieval_qpp_metrics', {})
                if qpp_metric in pre_qpp_metrics:
                    found = True
                    is_pre_retrieval = True
                    break
            if found:
                break
        if found:
            qpp_metrics_to_analyze.append((qpp_metric, is_pre_retrieval, is_bert_qpp))
    
    print(f"📊 Analyzing {len(qpp_metrics_to_analyze)} QPP metrics for {retrieval_method}...")
    
    for qpp_metric, is_pre_retrieval, is_bert_qpp in qpp_metrics_to_analyze:
        if is_bert_qpp:
            metric_label = qpp_metric  # Already has bert_qpp_ prefix
        elif is_pre_retrieval:
            metric_label = f"pre_{qpp_metric}"
        else:
            metric_label = qpp_metric
        print(f"  Processing {metric_label}...")
        
        # For each query, find the reformulation with highest QPP value
        selected_performances = {
            'nugget_strict_vital_score': [],
            'nugget_strict_all_score': [],
            'nugget_vital_score': [],
            'nugget_all_score': [],
            'genonly_strict_vital_score': [],
            'genonly_strict_all_score': [],
            'genonly_vital_score': [],
            'genonly_all_score': [],
            'retrieval_ndcg@5': [],
            'retrieval_ndcg@10': [],
            'retrieval_recall@100': []
        }
        
        queries_processed = 0
        
        for query_id, query_data in data.items():
            best_reformulation = None
            best_qpp_value = None
            
            # Find reformulation with highest QPP value for this metric
            for reformulation in query_data['reformulations']:
                if is_bert_qpp:
                    # Use BERT-QPP metrics (now inside retrieval method's qpp_metrics)
                    qpp_metrics_dict = reformulation.get('qpp_metrics', {})
                    retrieval_qpp = qpp_metrics_dict.get(retrieval_method, {})
                    qpp_value = retrieval_qpp.get(qpp_metric)
                elif is_pre_retrieval:
                    # Use pre-retrieval QPP metrics
                    qpp_metrics = reformulation.get('pre_retrieval_qpp_metrics', {})
                    qpp_value = qpp_metrics.get(qpp_metric)
                else:
                    # Use post-retrieval QPP metrics
                    qpp_metrics = reformulation.get('qpp_metrics', {}).get(retrieval_method, {})
                    qpp_value = qpp_metrics.get(qpp_metric)
                
                # Only consider numeric values
                if qpp_value is not None:
                    try:
                        qpp_value_float = float(qpp_value)
                        if not np.isnan(qpp_value_float):
                            if best_qpp_value is None or qpp_value_float > best_qpp_value:
                                best_qpp_value = qpp_value_float
                                best_reformulation = reformulation
                    except (ValueError, TypeError):
                        # Skip non-numeric values
                        pass
            
            # If we found a reformulation, collect its performance metrics
            if best_reformulation is not None:
                queries_processed += 1
                
                # Get nugget scores (use the same retrieval method)
                if retrieval_method == 'pyserini':
                    nugget_scores = best_reformulation.get('nugget_scores', {}).get('retrieval', {})
                else:  # cohere
                    nugget_scores = best_reformulation.get('nugget_scores', {}).get('retrieval_cohere', {})
                
                # Get generation-only nugget scores
                generation_only_scores = _generation_only_scores(best_reformulation)
                
                # Get retrieval metrics
                retrieval_metrics = best_reformulation.get('retrieval_metrics', {}).get(retrieval_method, {})
                
                # Collect retrieval-based nugget metrics
                if nugget_scores:
                    selected_performances['nugget_strict_vital_score'].append(nugget_scores.get('strict_vital_score'))
                    selected_performances['nugget_strict_all_score'].append(nugget_scores.get('strict_all_score'))
                    selected_performances['nugget_vital_score'].append(nugget_scores.get('vital_score'))
                    selected_performances['nugget_all_score'].append(nugget_scores.get('all_score'))
                
                # Collect generation-only nugget metrics
                if generation_only_scores:
                    selected_performances['genonly_strict_vital_score'].append(generation_only_scores.get('strict_vital_score'))
                    selected_performances['genonly_strict_all_score'].append(generation_only_scores.get('strict_all_score'))
                    selected_performances['genonly_vital_score'].append(generation_only_scores.get('vital_score'))
                    selected_performances['genonly_all_score'].append(generation_only_scores.get('all_score'))
                
                if retrieval_metrics:
                    selected_performances['retrieval_ndcg@5'].append(retrieval_metrics.get('ndcg@5'))
                    selected_performances['retrieval_ndcg@10'].append(retrieval_metrics.get('ndcg@10'))
                    selected_performances['retrieval_recall@100'].append(retrieval_metrics.get('recall@100'))
        
        # Calculate averages (ignoring None/NaN values)
        result_row = {
            'qpp_metric': metric_label,
            'retrieval_method': retrieval_method,
            'n_queries': queries_processed
        }
        
        for metric_name, values in selected_performances.items():
            valid_values = [v for v in values if v is not None and not np.isnan(v)]
            if valid_values:
                result_row[f'{metric_name}_mean'] = np.mean(valid_values)
                result_row[f'{metric_name}_std'] = np.std(valid_values)
                result_row[f'{metric_name}_n'] = len(valid_values)
            else:
                result_row[f'{metric_name}_mean'] = None
                result_row[f'{metric_name}_std'] = None
                result_row[f'{metric_name}_n'] = 0
        
        results.append(result_row)
    
    return pd.DataFrame(results)

def calculate_original_performance(data, retrieval_method='pyserini'):
    """Calculate performance when using original queries only."""
    selected_performances = {
        'nugget_strict_vital_score': [],
        'nugget_strict_all_score': [],
        'nugget_vital_score': [],
        'nugget_all_score': [],
        'genonly_strict_vital_score': [],
        'genonly_strict_all_score': [],
        'genonly_vital_score': [],
        'genonly_all_score': [],
        'retrieval_ndcg@5': [],
        'retrieval_ndcg@10': [],
        'retrieval_recall@100': []
    }
    
    queries_processed = 0
    
    for query_id, query_data in data.items():
        # Find the original query reformulation
        for reformulation in query_data['reformulations']:
            if reformulation.get('method') == 'original':
                queries_processed += 1
                
                # Get nugget scores
                if retrieval_method == 'pyserini':
                    nugget_scores = reformulation.get('nugget_scores', {}).get('retrieval', {})
                else:  # cohere
                    nugget_scores = reformulation.get('nugget_scores', {}).get('retrieval_cohere', {})
                
                # Get generation-only scores
                generation_only_scores = _generation_only_scores(reformulation)
                
                # Get retrieval metrics
                retrieval_metrics = reformulation.get('retrieval_metrics', {}).get(retrieval_method, {})
                
                # Collect retrieval-based nugget metrics
                if nugget_scores:
                    selected_performances['nugget_strict_vital_score'].append(nugget_scores.get('strict_vital_score'))
                    selected_performances['nugget_strict_all_score'].append(nugget_scores.get('strict_all_score'))
                    selected_performances['nugget_vital_score'].append(nugget_scores.get('vital_score'))
                    selected_performances['nugget_all_score'].append(nugget_scores.get('all_score'))
                
                # Collect generation-only nugget metrics
                if generation_only_scores:
                    selected_performances['genonly_strict_vital_score'].append(generation_only_scores.get('strict_vital_score'))
                    selected_performances['genonly_strict_all_score'].append(generation_only_scores.get('strict_all_score'))
                    selected_performances['genonly_vital_score'].append(generation_only_scores.get('vital_score'))
                    selected_performances['genonly_all_score'].append(generation_only_scores.get('all_score'))
                
                if retrieval_metrics:
                    selected_performances['retrieval_ndcg@5'].append(retrieval_metrics.get('ndcg@5'))
                    selected_performances['retrieval_ndcg@10'].append(retrieval_metrics.get('ndcg@10'))
                    selected_performances['retrieval_recall@100'].append(retrieval_metrics.get('recall@100'))
                break
    
    # Calculate averages
    result_row = {
        'qpp_metric': 'original',
        'retrieval_method': retrieval_method,
        'n_queries': queries_processed
    }
    
    for metric_name, values in selected_performances.items():
        valid_values = [v for v in values if v is not None and not np.isnan(v)]
        if valid_values:
            result_row[f'{metric_name}_mean'] = np.mean(valid_values)
            result_row[f'{metric_name}_std'] = np.std(valid_values)
            result_row[f'{metric_name}_n'] = len(valid_values)
        else:
            result_row[f'{metric_name}_mean'] = None
            result_row[f'{metric_name}_std'] = None
            result_row[f'{metric_name}_n'] = 0
    
    return result_row

def calculate_max_oracle_performance(data, retrieval_method='pyserini', performance_metric='nugget_all_score', use_generation_only=False):
    """
    Calculate maximum oracle performance: for each query, select the variation
    with the highest actual performance (not based on QPP).
    
    Args:
        performance_metric: Which metric to use for selecting best variation
                           ('nugget_all_score', 'nugget_strict_all_score', 'retrieval_ndcg@10', etc.)
        use_generation_only: If True, select based on generation-only scores instead of retrieval scores
    """
    selected_performances = {
        'nugget_strict_vital_score': [],
        'nugget_strict_all_score': [],
        'nugget_vital_score': [],
        'nugget_all_score': [],
        'genonly_strict_vital_score': [],
        'genonly_strict_all_score': [],
        'genonly_vital_score': [],
        'genonly_all_score': [],
        'retrieval_ndcg@5': [],
        'retrieval_ndcg@10': [],
        'retrieval_recall@100': []
    }
    
    queries_processed = 0
    
    for query_id, query_data in data.items():
        best_reformulation = None
        best_performance_value = None
        
        # Find reformulation with highest performance for the specified metric
        for reformulation in query_data['reformulations']:
            # Determine which scores to use for selection
            if use_generation_only:
                # Use generation-only scores for selection
                generation_only_scores = _generation_only_scores(reformulation)
                if performance_metric.startswith('nugget_'):
                    metric_value = generation_only_scores.get(performance_metric.replace('nugget_', ''))
                else:
                    metric_value = None
            else:
                # Use retrieval-based scores for selection
                if retrieval_method == 'pyserini':
                    nugget_scores = reformulation.get('nugget_scores', {}).get('retrieval', {})
                else:  # cohere
                    nugget_scores = reformulation.get('nugget_scores', {}).get('retrieval_cohere', {})
                
                retrieval_metrics = reformulation.get('retrieval_metrics', {}).get(retrieval_method, {})
                
                if performance_metric.startswith('nugget_'):
                    metric_value = nugget_scores.get(performance_metric.replace('nugget_', ''))
                elif performance_metric.startswith('retrieval_'):
                    metric_value = retrieval_metrics.get(performance_metric.replace('retrieval_', ''))
                else:
                    metric_value = None
            
            if metric_value is not None and not np.isnan(metric_value):
                if best_performance_value is None or metric_value > best_performance_value:
                    best_performance_value = metric_value
                    best_reformulation = reformulation
        
        # Collect metrics from best reformulation
        if best_reformulation is not None:
            queries_processed += 1
            
            # Get nugget scores
            if retrieval_method == 'pyserini':
                nugget_scores = best_reformulation.get('nugget_scores', {}).get('retrieval', {})
            else:  # cohere
                nugget_scores = best_reformulation.get('nugget_scores', {}).get('retrieval_cohere', {})
            
            # Get generation-only scores
            generation_only_scores = _generation_only_scores(best_reformulation)
            
            # Get retrieval metrics
            retrieval_metrics = best_reformulation.get('retrieval_metrics', {}).get(retrieval_method, {})
            
            # Collect retrieval-based nugget metrics
            if nugget_scores:
                selected_performances['nugget_strict_vital_score'].append(nugget_scores.get('strict_vital_score'))
                selected_performances['nugget_strict_all_score'].append(nugget_scores.get('strict_all_score'))
                selected_performances['nugget_vital_score'].append(nugget_scores.get('vital_score'))
                selected_performances['nugget_all_score'].append(nugget_scores.get('all_score'))
            
            # Collect generation-only nugget metrics
            if generation_only_scores:
                selected_performances['genonly_strict_vital_score'].append(generation_only_scores.get('strict_vital_score'))
                selected_performances['genonly_strict_all_score'].append(generation_only_scores.get('strict_all_score'))
                selected_performances['genonly_vital_score'].append(generation_only_scores.get('vital_score'))
                selected_performances['genonly_all_score'].append(generation_only_scores.get('all_score'))
            
            if retrieval_metrics:
                selected_performances['retrieval_ndcg@5'].append(retrieval_metrics.get('ndcg@5'))
                selected_performances['retrieval_ndcg@10'].append(retrieval_metrics.get('ndcg@10'))
                selected_performances['retrieval_recall@100'].append(retrieval_metrics.get('recall@100'))
    
    # Calculate averages
    if use_generation_only:
        metric_prefix = 'max_oracle_genonly_'
    else:
        metric_prefix = 'max_oracle_'
    
    result_row = {
        'qpp_metric': f'{metric_prefix}{performance_metric}',
        'retrieval_method': retrieval_method,
        'n_queries': queries_processed
    }
    
    for metric_name, values in selected_performances.items():
        valid_values = [v for v in values if v is not None and not np.isnan(v)]
        if valid_values:
            result_row[f'{metric_name}_mean'] = np.mean(valid_values)
            result_row[f'{metric_name}_std'] = np.std(valid_values)
            result_row[f'{metric_name}_n'] = len(valid_values)
        else:
            result_row[f'{metric_name}_mean'] = None
            result_row[f'{metric_name}_std'] = None
            result_row[f'{metric_name}_n'] = 0
    
    return result_row

def calculate_method_average_performance(data, retrieval_method='pyserini', method_name='genqr'):
    """
    Calculate average performance across all trials for a specific reformulation method.
    For each query, average the performance metrics across all trials of the specified method.
    
    Args:
        data: Consolidated query data
        retrieval_method: 'pyserini' or 'cohere'
        method_name: Name of the reformulation method (e.g., 'genqr', 'genqr_ensemble', etc.)
    
    Returns:
        Dictionary with aggregated performance metrics
    """
    selected_performances = {
        'nugget_strict_vital_score': [],
        'nugget_strict_all_score': [],
        'nugget_vital_score': [],
        'nugget_all_score': [],
        'genonly_strict_vital_score': [],
        'genonly_strict_all_score': [],
        'genonly_vital_score': [],
        'genonly_all_score': [],
        'retrieval_ndcg@5': [],
        'retrieval_ndcg@10': [],
        'retrieval_recall@100': []
    }
    
    queries_processed = 0
    
    for query_id, query_data in data.items():
        # Find all reformulations with the specified method
        method_reformulations = [
            reform for reform in query_data['reformulations']
            if reform.get('method') == method_name
        ]
        
        if not method_reformulations:
            continue
        
        queries_processed += 1
        
        # Average metrics across all trials for this method
        query_nugget_strict_vital = []
        query_nugget_strict_all = []
        query_nugget_vital = []
        query_nugget_all = []
        query_genonly_strict_vital = []
        query_genonly_strict_all = []
        query_genonly_vital = []
        query_genonly_all = []
        query_ndcg5 = []
        query_ndcg = []
        query_recall = []
        
        for reformulation in method_reformulations:
            # Get nugget scores
            if retrieval_method == 'pyserini':
                nugget_scores = reformulation.get('nugget_scores', {}).get('retrieval', {})
            else:  # cohere
                nugget_scores = reformulation.get('nugget_scores', {}).get('retrieval_cohere', {})
            
            # Get generation-only scores
            generation_only_scores = _generation_only_scores(reformulation)
            
            # Get retrieval metrics
            retrieval_metrics = reformulation.get('retrieval_metrics', {}).get(retrieval_method, {})
            
            # Collect retrieval-based nugget metrics for this trial
            if nugget_scores:
                if nugget_scores.get('strict_vital_score') is not None:
                    query_nugget_strict_vital.append(nugget_scores.get('strict_vital_score'))
                if nugget_scores.get('strict_all_score') is not None:
                    query_nugget_strict_all.append(nugget_scores.get('strict_all_score'))
                if nugget_scores.get('vital_score') is not None:
                    query_nugget_vital.append(nugget_scores.get('vital_score'))
                if nugget_scores.get('all_score') is not None:
                    query_nugget_all.append(nugget_scores.get('all_score'))
            
            # Collect generation-only nugget metrics for this trial
            if generation_only_scores:
                if generation_only_scores.get('strict_vital_score') is not None:
                    query_genonly_strict_vital.append(generation_only_scores.get('strict_vital_score'))
                if generation_only_scores.get('strict_all_score') is not None:
                    query_genonly_strict_all.append(generation_only_scores.get('strict_all_score'))
                if generation_only_scores.get('vital_score') is not None:
                    query_genonly_vital.append(generation_only_scores.get('vital_score'))
                if generation_only_scores.get('all_score') is not None:
                    query_genonly_all.append(generation_only_scores.get('all_score'))
            
            if retrieval_metrics:
                if retrieval_metrics.get('ndcg@5') is not None:
                    query_ndcg5.append(retrieval_metrics.get('ndcg@5'))
                if retrieval_metrics.get('ndcg@10') is not None:
                    query_ndcg.append(retrieval_metrics.get('ndcg@10'))
                if retrieval_metrics.get('recall@100') is not None:
                    query_recall.append(retrieval_metrics.get('recall@100'))
        
        # Average across trials for this query, then add to overall list
        if query_nugget_strict_vital:
            selected_performances['nugget_strict_vital_score'].append(np.mean(query_nugget_strict_vital))
        if query_nugget_strict_all:
            selected_performances['nugget_strict_all_score'].append(np.mean(query_nugget_strict_all))
        if query_nugget_vital:
            selected_performances['nugget_vital_score'].append(np.mean(query_nugget_vital))
        if query_nugget_all:
            selected_performances['nugget_all_score'].append(np.mean(query_nugget_all))
        if query_genonly_strict_vital:
            selected_performances['genonly_strict_vital_score'].append(np.mean(query_genonly_strict_vital))
        if query_genonly_strict_all:
            selected_performances['genonly_strict_all_score'].append(np.mean(query_genonly_strict_all))
        if query_genonly_vital:
            selected_performances['genonly_vital_score'].append(np.mean(query_genonly_vital))
        if query_genonly_all:
            selected_performances['genonly_all_score'].append(np.mean(query_genonly_all))
        if query_ndcg5:
            selected_performances['retrieval_ndcg@5'].append(np.mean(query_ndcg5))
        if query_ndcg:
            selected_performances['retrieval_ndcg@10'].append(np.mean(query_ndcg))
        if query_recall:
            selected_performances['retrieval_recall@100'].append(np.mean(query_recall))
    
    # Calculate averages across all queries
    result_row = {
        'qpp_metric': f'avg_{method_name}',
        'retrieval_method': retrieval_method,
        'n_queries': queries_processed
    }
    
    for metric_name, values in selected_performances.items():
        valid_values = [v for v in values if v is not None and not np.isnan(v)]
        if valid_values:
            result_row[f'{metric_name}_mean'] = np.mean(valid_values)
            result_row[f'{metric_name}_std'] = np.std(valid_values)
            result_row[f'{metric_name}_n'] = len(valid_values)
        else:
            result_row[f'{metric_name}_mean'] = None
            result_row[f'{metric_name}_std'] = None
            result_row[f'{metric_name}_n'] = 0
    
    return result_row

def analyze_generationonly_oracle_performance(data):
    """
    Analyze oracle performance for generation-only: For each pre-retrieval QPP metric, 
    select the variation with highest QPP value for each query, then report the aggregated 
    generation-only nugget scores.
    
    Args:
        data: Consolidated query data
    
    Returns:
        DataFrame with one row per QPP metric showing aggregated performance
    """
    results = []
    
    # Get all pre-retrieval QPP metrics
    pre_retrieval_metrics = set()
    for query_id, query_data in data.items():
        for reformulation in query_data['reformulations']:
            pre_qpp = reformulation.get('pre_retrieval_qpp_metrics', {})
            pre_retrieval_metrics.update(pre_qpp.keys())
    
    pre_retrieval_metrics = sorted(list(pre_retrieval_metrics))
    print(f"📊 Analyzing {len(pre_retrieval_metrics)} pre-retrieval QPP metrics for generation-only...")
    
    for qpp_metric_name in pre_retrieval_metrics:
        metric_label = f"pre_{qpp_metric_name}"
        print(f"  Processing {metric_label}...")
        
        selected_performances = {
            'generationonly_nugget_strict_vital_score': [],
            'generationonly_nugget_strict_all_score': [],
            'generationonly_nugget_vital_score': [],
            'generationonly_nugget_all_score': []
        }
        
        queries_processed = 0
        
        for query_id, query_data in data.items():
            best_reformulation = None
            best_qpp_value = None
            
            # Find reformulation with highest QPP value for this metric
            for reformulation in query_data['reformulations']:
                pre_qpp_metrics = reformulation.get('pre_retrieval_qpp_metrics', {})
                qpp_value = pre_qpp_metrics.get(qpp_metric_name)
                
                if qpp_value is not None:
                    try:
                        qpp_float = float(qpp_value)
                        if not np.isnan(qpp_float):
                            if best_qpp_value is None or qpp_float > best_qpp_value:
                                best_qpp_value = qpp_float
                                best_reformulation = reformulation
                    except (ValueError, TypeError):
                        pass
            
            # If we found a reformulation, collect its generation-only nugget scores
            if best_reformulation is not None:
                queries_processed += 1
                
                # Get generation-only nugget scores
                generationonly_nugget_scores = _generation_only_scores(best_reformulation)
                
                # Collect metrics
                if generationonly_nugget_scores:
                    selected_performances['generationonly_nugget_strict_vital_score'].append(
                        generationonly_nugget_scores.get('strict_vital_score'))
                    selected_performances['generationonly_nugget_strict_all_score'].append(
                        generationonly_nugget_scores.get('strict_all_score'))
                    selected_performances['generationonly_nugget_vital_score'].append(
                        generationonly_nugget_scores.get('vital_score'))
                    selected_performances['generationonly_nugget_all_score'].append(
                        generationonly_nugget_scores.get('all_score'))
        
        # Calculate averages (ignoring None/NaN values)
        result_row = {
            'qpp_metric': metric_label,
            'retrieval_method': 'generationonly',
            'n_queries': queries_processed
        }
        
        for metric_name, values in selected_performances.items():
            valid_values = [v for v in values if v is not None and not np.isnan(v)]
            if valid_values:
                result_row[f'{metric_name}_mean'] = np.mean(valid_values)
                result_row[f'{metric_name}_std'] = np.std(valid_values)
                result_row[f'{metric_name}_n'] = len(valid_values)
            else:
                result_row[f'{metric_name}_mean'] = None
                result_row[f'{metric_name}_std'] = None
                result_row[f'{metric_name}_n'] = 0
        
        results.append(result_row)
    
    return pd.DataFrame(results)

def main():
    base_dir = _REPO / "querygym"
    json_file = base_dir / "consolidated_query_data.json"
    output_dir = base_dir / "qpp_oracle_analysis"
    output_dir.mkdir(exist_ok=True)
    
    print("📖 Loading consolidated data...")
    data = load_consolidated_data(json_file)
    print(f"✅ Loaded data for {len(data)} queries")
    
    # Analyze for both retrieval methods
    print("\n" + "="*80)
    print("ANALYZING PYSERINI RETRIEVAL")
    print("="*80)
    df_pyserini = analyze_qpp_oracle_performance(data, retrieval_method='pyserini')
    
    # Add original and max oracle rows
    print("\n  Adding original query performance...")
    original_row_pyserini = calculate_original_performance(data, retrieval_method='pyserini')
    df_pyserini = pd.concat([df_pyserini, pd.DataFrame([original_row_pyserini])], ignore_index=True)
    
    print("  Adding maximum oracle performance (based on nugget_all_score)...")
    max_oracle_row_pyserini = calculate_max_oracle_performance(data, retrieval_method='pyserini', performance_metric='nugget_all_score')
    max_oracle_row_pyserini['qpp_metric'] = 'max_oracle_nugget_all'
    df_pyserini = pd.concat([df_pyserini, pd.DataFrame([max_oracle_row_pyserini])], ignore_index=True)
    
    print("  Adding maximum oracle performance (based on retrieval_ndcg@5)...")
    max_oracle_ndcg5_pyserini = calculate_max_oracle_performance(data, retrieval_method='pyserini', performance_metric='retrieval_ndcg@5')
    max_oracle_ndcg5_pyserini['qpp_metric'] = 'max_oracle_ndcg@5'
    df_pyserini = _append_available_row(df_pyserini, max_oracle_ndcg5_pyserini)
    
    print("  Adding maximum oracle performance (based on retrieval_ndcg@10)...")
    max_oracle_ndcg_pyserini = calculate_max_oracle_performance(data, retrieval_method='pyserini', performance_metric='retrieval_ndcg@10')
    max_oracle_ndcg_pyserini['qpp_metric'] = 'max_oracle_ndcg@10'
    df_pyserini = pd.concat([df_pyserini, pd.DataFrame([max_oracle_ndcg_pyserini])], ignore_index=True)
    
    print("  Adding maximum oracle performance (based on retrieval_recall@100)...")
    max_oracle_recall_pyserini = calculate_max_oracle_performance(data, retrieval_method='pyserini', performance_metric='retrieval_recall@100')
    max_oracle_recall_pyserini['qpp_metric'] = 'max_oracle_recall@100'
    df_pyserini = pd.concat([df_pyserini, pd.DataFrame([max_oracle_recall_pyserini])], ignore_index=True)
    
    print("  Adding maximum oracle performance (based on nugget_strict_all_score)...")
    max_oracle_strict_all_pyserini = calculate_max_oracle_performance(data, retrieval_method='pyserini', performance_metric='nugget_strict_all_score')
    max_oracle_strict_all_pyserini['qpp_metric'] = 'max_oracle_strict_all_score'
    df_pyserini = pd.concat([df_pyserini, pd.DataFrame([max_oracle_strict_all_pyserini])], ignore_index=True)
    
    print("  Adding maximum oracle performance (based on nugget_strict_vital_score)...")
    max_oracle_strict_vital_pyserini = calculate_max_oracle_performance(data, retrieval_method='pyserini', performance_metric='nugget_strict_vital_score')
    max_oracle_strict_vital_pyserini['qpp_metric'] = 'max_oracle_strict_vital_score'
    df_pyserini = pd.concat([df_pyserini, pd.DataFrame([max_oracle_strict_vital_pyserini])], ignore_index=True)
    
    print("  Adding maximum oracle performance (based on nugget_vital_score)...")
    max_oracle_vital_pyserini = calculate_max_oracle_performance(data, retrieval_method='pyserini', performance_metric='nugget_vital_score')
    max_oracle_vital_pyserini['qpp_metric'] = 'max_oracle_vital_score'
    df_pyserini = pd.concat([df_pyserini, pd.DataFrame([max_oracle_vital_pyserini])], ignore_index=True)
    
    # Add generation-only oracle rows (selecting based on generation-only scores)
    print("  Adding generation-only oracle performance (based on genonly nugget_all_score)...")
    max_oracle_genonly_all_pyserini = calculate_max_oracle_performance(data, retrieval_method='pyserini', performance_metric='nugget_all_score', use_generation_only=True)
    df_pyserini = _append_available_row(df_pyserini, max_oracle_genonly_all_pyserini)
    
    print("  Adding generation-only oracle performance (based on genonly nugget_strict_all_score)...")
    max_oracle_genonly_strict_all_pyserini = calculate_max_oracle_performance(data, retrieval_method='pyserini', performance_metric='nugget_strict_all_score', use_generation_only=True)
    df_pyserini = _append_available_row(df_pyserini, max_oracle_genonly_strict_all_pyserini)
    
    print("  Adding generation-only oracle performance (based on genonly nugget_strict_vital_score)...")
    max_oracle_genonly_strict_vital_pyserini = calculate_max_oracle_performance(data, retrieval_method='pyserini', performance_metric='nugget_strict_vital_score', use_generation_only=True)
    df_pyserini = _append_available_row(df_pyserini, max_oracle_genonly_strict_vital_pyserini)
    
    print("  Adding generation-only oracle performance (based on genonly nugget_vital_score)...")
    max_oracle_genonly_vital_pyserini = calculate_max_oracle_performance(data, retrieval_method='pyserini', performance_metric='nugget_vital_score', use_generation_only=True)
    df_pyserini = _append_available_row(df_pyserini, max_oracle_genonly_vital_pyserini)
    
    # Add average performance across trials for each reformulation method
    print("\n  Adding average performance across trials for each reformulation method...")
    methods = ['genqr', 'genqr_ensemble', 'mugi', 'qa_expand', 'query2doc', 'query2e']
    for method in methods:
        print(f"    Processing {method}...")
        method_avg_pyserini = calculate_method_average_performance(data, retrieval_method='pyserini', method_name=method)
        df_pyserini = pd.concat([df_pyserini, pd.DataFrame([method_avg_pyserini])], ignore_index=True)
    
    output_csv_pyserini = output_dir / "qpp_oracle_performance_pyserini.csv"
    df_pyserini.to_csv(output_csv_pyserini, index=False)
    print(f"\n✅ Saved Pyserini results: {output_csv_pyserini}")
    
    # Create simplified version with all nugget metrics
    simple_columns = [
        'qpp_metric', 'retrieval_method', 'n_queries',
        'nugget_strict_vital_score_mean',
        'nugget_strict_all_score_mean',
        'nugget_vital_score_mean',
        'nugget_all_score_mean',
        'retrieval_ndcg@10_mean',
        'retrieval_recall@100_mean'
    ]
    df_pyserini_simple = df_pyserini[simple_columns].copy()
    # Round to 4 decimals
    for col in simple_columns[3:]:
        if col in df_pyserini_simple.columns:
            df_pyserini_simple[col] = pd.to_numeric(
                df_pyserini_simple[col], errors='coerce'
            ).round(4)
    output_csv_pyserini_simple = output_dir / "qpp_oracle_performance_pyserini_simple.csv"
    df_pyserini_simple.to_csv(output_csv_pyserini_simple, index=False)
    print(f"✅ Saved Pyserini simplified results: {output_csv_pyserini_simple}")
    
    print("\n" + "="*80)
    print("ANALYZING COHERE RETRIEVAL")
    print("="*80)
    df_cohere = analyze_qpp_oracle_performance(data, retrieval_method='cohere')
    
    # Add original and max oracle rows
    print("\n  Adding original query performance...")
    original_row_cohere = calculate_original_performance(data, retrieval_method='cohere')
    df_cohere = pd.concat([df_cohere, pd.DataFrame([original_row_cohere])], ignore_index=True)
    
    print("  Adding maximum oracle performance (based on nugget_all_score)...")
    max_oracle_row_cohere = calculate_max_oracle_performance(data, retrieval_method='cohere', performance_metric='nugget_all_score')
    max_oracle_row_cohere['qpp_metric'] = 'max_oracle_nugget_all'
    df_cohere = pd.concat([df_cohere, pd.DataFrame([max_oracle_row_cohere])], ignore_index=True)
    
    print("  Adding maximum oracle performance (based on retrieval_ndcg@5)...")
    max_oracle_ndcg5_cohere = calculate_max_oracle_performance(data, retrieval_method='cohere', performance_metric='retrieval_ndcg@5')
    max_oracle_ndcg5_cohere['qpp_metric'] = 'max_oracle_ndcg@5'
    df_cohere = _append_available_row(df_cohere, max_oracle_ndcg5_cohere)
    
    print("  Adding maximum oracle performance (based on retrieval_ndcg@10)...")
    max_oracle_ndcg_cohere = calculate_max_oracle_performance(data, retrieval_method='cohere', performance_metric='retrieval_ndcg@10')
    max_oracle_ndcg_cohere['qpp_metric'] = 'max_oracle_ndcg@10'
    df_cohere = pd.concat([df_cohere, pd.DataFrame([max_oracle_ndcg_cohere])], ignore_index=True)
    
    print("  Adding maximum oracle performance (based on retrieval_recall@100)...")
    max_oracle_recall_cohere = calculate_max_oracle_performance(data, retrieval_method='cohere', performance_metric='retrieval_recall@100')
    max_oracle_recall_cohere['qpp_metric'] = 'max_oracle_recall@100'
    df_cohere = pd.concat([df_cohere, pd.DataFrame([max_oracle_recall_cohere])], ignore_index=True)
    
    print("  Adding maximum oracle performance (based on nugget_strict_all_score)...")
    max_oracle_strict_all_cohere = calculate_max_oracle_performance(data, retrieval_method='cohere', performance_metric='nugget_strict_all_score')
    max_oracle_strict_all_cohere['qpp_metric'] = 'max_oracle_strict_all_score'
    df_cohere = pd.concat([df_cohere, pd.DataFrame([max_oracle_strict_all_cohere])], ignore_index=True)
    
    print("  Adding maximum oracle performance (based on nugget_strict_vital_score)...")
    max_oracle_strict_vital_cohere = calculate_max_oracle_performance(data, retrieval_method='cohere', performance_metric='nugget_strict_vital_score')
    max_oracle_strict_vital_cohere['qpp_metric'] = 'max_oracle_strict_vital_score'
    df_cohere = pd.concat([df_cohere, pd.DataFrame([max_oracle_strict_vital_cohere])], ignore_index=True)
    
    print("  Adding maximum oracle performance (based on nugget_vital_score)...")
    max_oracle_vital_cohere = calculate_max_oracle_performance(data, retrieval_method='cohere', performance_metric='nugget_vital_score')
    max_oracle_vital_cohere['qpp_metric'] = 'max_oracle_vital_score'
    df_cohere = pd.concat([df_cohere, pd.DataFrame([max_oracle_vital_cohere])], ignore_index=True)
    
    # Add generation-only oracle rows (selecting based on generation-only scores)
    print("  Adding generation-only oracle performance (based on genonly nugget_all_score)...")
    max_oracle_genonly_all_cohere = calculate_max_oracle_performance(data, retrieval_method='cohere', performance_metric='nugget_all_score', use_generation_only=True)
    df_cohere = _append_available_row(df_cohere, max_oracle_genonly_all_cohere)
    
    print("  Adding generation-only oracle performance (based on genonly nugget_strict_all_score)...")
    max_oracle_genonly_strict_all_cohere = calculate_max_oracle_performance(data, retrieval_method='cohere', performance_metric='nugget_strict_all_score', use_generation_only=True)
    df_cohere = _append_available_row(df_cohere, max_oracle_genonly_strict_all_cohere)
    
    print("  Adding generation-only oracle performance (based on genonly nugget_strict_vital_score)...")
    max_oracle_genonly_strict_vital_cohere = calculate_max_oracle_performance(data, retrieval_method='cohere', performance_metric='nugget_strict_vital_score', use_generation_only=True)
    df_cohere = _append_available_row(df_cohere, max_oracle_genonly_strict_vital_cohere)
    
    print("  Adding generation-only oracle performance (based on genonly nugget_vital_score)...")
    max_oracle_genonly_vital_cohere = calculate_max_oracle_performance(data, retrieval_method='cohere', performance_metric='nugget_vital_score', use_generation_only=True)
    df_cohere = _append_available_row(df_cohere, max_oracle_genonly_vital_cohere)
    
    # Add average performance across trials for each reformulation method
    print("\n  Adding average performance across trials for each reformulation method...")
    methods = ['genqr', 'genqr_ensemble', 'mugi', 'qa_expand', 'query2doc', 'query2e']
    for method in methods:
        print(f"    Processing {method}...")
        method_avg_cohere = calculate_method_average_performance(data, retrieval_method='cohere', method_name=method)
        df_cohere = pd.concat([df_cohere, pd.DataFrame([method_avg_cohere])], ignore_index=True)
    
    output_csv_cohere = output_dir / "qpp_oracle_performance_cohere.csv"
    df_cohere.to_csv(output_csv_cohere, index=False)
    print(f"\n✅ Saved Cohere results: {output_csv_cohere}")
    
    # Create simplified version with all nugget metrics
    simple_columns = [
        'qpp_metric', 'retrieval_method', 'n_queries',
        'nugget_strict_vital_score_mean',
        'nugget_strict_all_score_mean',
        'nugget_vital_score_mean',
        'nugget_all_score_mean',
        'retrieval_ndcg@10_mean',
        'retrieval_recall@100_mean'
    ]
    df_cohere_simple = df_cohere[simple_columns].copy()
    # Round to 4 decimals
    for col in simple_columns[3:]:
        if col in df_cohere_simple.columns:
            df_cohere_simple[col] = pd.to_numeric(
                df_cohere_simple[col], errors='coerce'
            ).round(4)
    output_csv_cohere_simple = output_dir / "qpp_oracle_performance_cohere_simple.csv"
    df_cohere_simple.to_csv(output_csv_cohere_simple, index=False)
    print(f"✅ Saved Cohere simplified results: {output_csv_cohere_simple}")
    
    # Create combined summary
    print("\n📊 Summary Statistics:")
    print("\n" + "="*80)
    print("PY SERINI - Top 10 QPP Metrics by Nugget All Score:")
    print("="*80)
    top_pyserini = df_pyserini.nlargest(10, 'nugget_all_score_mean')[
        ['qpp_metric', 'nugget_all_score_mean', 'nugget_strict_all_score_mean', 
         'retrieval_ndcg@10_mean', 'retrieval_recall@100_mean']
    ]
    print(top_pyserini.to_string(index=False))
    
    print("\n" + "="*80)
    print("COHERE - Top 10 QPP Metrics by Nugget All Score:")
    print("="*80)
    top_cohere = df_cohere.nlargest(10, 'nugget_all_score_mean')[
        ['qpp_metric', 'nugget_all_score_mean', 'nugget_strict_all_score_mean',
         'retrieval_ndcg@10_mean', 'retrieval_recall@100_mean']
    ]
    print(top_cohere.to_string(index=False))
    
    print(f"\n🎉 Analysis complete! Results saved to: {output_dir}")

if __name__ == "__main__":
    main()
