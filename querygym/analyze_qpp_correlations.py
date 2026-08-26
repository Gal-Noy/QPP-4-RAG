#!/usr/bin/env python3
"""
Analyze correlations between QPP metrics and actual performance metrics.

For each QPP method (pre-retrieval, post-retrieval pyserini, post-retrieval cohere),
calculate Pearson and Kendall Tau correlations with:
- Retrieval metrics (nDCG@10, Recall@100)
- Nugget scores (strict_vital_score, strict_all_score, vital_score, all_score)

Correlations are calculated per-query across all 31 variations, then averaged.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr, kendalltau
from collections import defaultdict

_REPO = Path(__file__).resolve().parent.parent


def _numeric_array(values):
    """Return float data with missing/non-numeric values represented as NaN."""
    converted = []
    for value in values:
        try:
            converted.append(float(value) if value is not None else np.nan)
        except (TypeError, ValueError):
            converted.append(np.nan)
    return np.asarray(converted, dtype=float)


def _generation_only_scores(reformulation):
    """Read either historical spelling of the optional generation-only field."""
    return (
        reformulation.get('generationonly_nugget_scores')
        or reformulation.get('generation_only_nugget_scores')
        or {}
    )


def _has_generation_only_scores(data):
    return any(
        any(value is not None for value in _generation_only_scores(reformulation).values())
        for query_data in data.values()
        for reformulation in query_data['reformulations']
    )


def _append_pair_correlation(destination, qpp_values, performance_values):
    """Append one valid Pearson/Kendall pair, if both inputs vary."""
    qpp_array = _numeric_array(qpp_values)
    performance_array = _numeric_array(performance_values)
    if qpp_array.shape != performance_array.shape:
        raise ValueError("QPP and performance arrays are not aligned")

    valid = np.isfinite(qpp_array) & np.isfinite(performance_array)
    if valid.sum() < 3:
        return
    qpp_valid = qpp_array[valid]
    performance_valid = performance_array[valid]
    if np.ptp(qpp_valid) == 0 or np.ptp(performance_valid) == 0:
        return

    pearson_r, _ = pearsonr(qpp_valid, performance_valid)
    kendall_tau, _ = kendalltau(qpp_valid, performance_valid)
    pearson_value = pearson_r if np.isfinite(pearson_r) else None
    kendall_value = kendall_tau if np.isfinite(kendall_tau) else None
    if pearson_value is not None or kendall_value is not None:
        destination.append({'pearson': pearson_value, 'kendall': kendall_value})

def load_consolidated_data(json_file):
    """Load the consolidated query data."""
    with open(json_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_all_qpp_metrics(data):
    """Get all unique QPP metric names from the data."""
    qpp_metrics = set()
    for query_id, query_data in data.items():
        for reformulation in query_data['reformulations']:
            # Post-retrieval metrics
            qpp_metrics_dict = reformulation.get('qpp_metrics', {})
            pyserini_qpp = qpp_metrics_dict.get('pyserini', {})
            cohere_qpp = qpp_metrics_dict.get('cohere', {})
            qpp_metrics.update([(m, 'post_pyserini') for m in pyserini_qpp.keys()])
            qpp_metrics.update([(m, 'post_cohere') for m in cohere_qpp.keys()])
            
            # Pre-retrieval metrics - analyze with both retrieval methods
            pre_qpp = reformulation.get('pre_retrieval_qpp_metrics', {})
            qpp_metrics.update([(m, 'pre_pyserini') for m in pre_qpp.keys()])
            qpp_metrics.update([(m, 'pre_cohere') for m in pre_qpp.keys()])
    
    return sorted(list(qpp_metrics))

def calculate_correlations(data, qpp_metric_name, qpp_type):
    """
    Calculate correlations for a specific QPP metric.
    
    Args:
        data: Consolidated query data
        qpp_metric_name: Name of the QPP metric
        qpp_type: 'pre', 'post_pyserini', or 'post_cohere'
    
    Returns:
        Dictionary with correlation results
    """
    # Determine which retrieval method to use for performance metrics
    if qpp_type == 'post_pyserini' or qpp_type == 'pre_pyserini':
        retrieval_method = 'pyserini'
        nugget_key = 'retrieval'
    else:  # post_cohere or pre_cohere
        retrieval_method = 'cohere'
        nugget_key = 'retrieval_cohere'
    
    # Collect data across all queries and variations
    correlations_per_query = {
        'nugget_strict_vital_score': [],
        'nugget_strict_all_score': [],
        'nugget_vital_score': [],
        'nugget_all_score': [],
        'retrieval_ndcg@10': [],
        'retrieval_recall@100': []
    }
    
    for query_id, query_data in data.items():
        qpp_values = []
        performance_values = {
            'nugget_strict_vital_score': [],
            'nugget_strict_all_score': [],
            'nugget_vital_score': [],
            'nugget_all_score': [],
            'retrieval_ndcg@10': [],
            'retrieval_recall@100': []
        }
        
        # Collect QPP values and performance metrics for all variations
        for reformulation in query_data['reformulations']:
            # Get QPP value
            if qpp_type == 'pre_pyserini' or qpp_type == 'pre_cohere':
                qpp_metrics = reformulation.get('pre_retrieval_qpp_metrics', {})
                qpp_value = qpp_metrics.get(qpp_metric_name)
            elif qpp_type == 'post_pyserini':
                qpp_metrics = reformulation.get('qpp_metrics', {}).get('pyserini', {})
                qpp_value = qpp_metrics.get(qpp_metric_name)
            else:  # post_cohere
                qpp_metrics = reformulation.get('qpp_metrics', {}).get('cohere', {})
                qpp_value = qpp_metrics.get(qpp_metric_name)
            
            # Get performance metrics
            nugget_scores = reformulation.get('nugget_scores', {}).get(nugget_key, {})
            retrieval_metrics = reformulation.get('retrieval_metrics', {}).get(retrieval_method, {})
            
            # Only include if QPP value is valid
            if qpp_value is not None:
                try:
                    qpp_float = float(qpp_value)
                    if not np.isnan(qpp_float):
                        qpp_values.append(qpp_float)
                        
                        # Collect performance values
                        performance_values['nugget_strict_vital_score'].append(
                            nugget_scores.get('strict_vital_score'))
                        performance_values['nugget_strict_all_score'].append(
                            nugget_scores.get('strict_all_score'))
                        performance_values['nugget_vital_score'].append(
                            nugget_scores.get('vital_score'))
                        performance_values['nugget_all_score'].append(
                            nugget_scores.get('all_score'))
                        performance_values['retrieval_ndcg@10'].append(
                            retrieval_metrics.get('ndcg@10'))
                        performance_values['retrieval_recall@100'].append(
                            retrieval_metrics.get('recall@100'))
                except (ValueError, TypeError):
                    pass
        
        # Calculate correlations for this query
        if len(qpp_values) >= 3:  # Need at least 3 points for correlation
            for perf_metric in correlations_per_query:
                _append_pair_correlation(
                    correlations_per_query[perf_metric],
                    qpp_values,
                    performance_values[perf_metric],
                )
    
    # Calculate average correlations across all queries
    result = {
        'qpp_metric': qpp_metric_name,
        'qpp_type': qpp_type,
        'retrieval_method': retrieval_method
    }
    
    for perf_metric in correlations_per_query.keys():
        correlations = correlations_per_query[perf_metric]
        
        if correlations:
            pearson_values = [c['pearson'] for c in correlations if c['pearson'] is not None]
            kendall_values = [c['kendall'] for c in correlations if c['kendall'] is not None]
            
            result[f'{perf_metric}_pearson_mean'] = np.mean(pearson_values) if pearson_values else np.nan
            result[f'{perf_metric}_pearson_n'] = len(pearson_values)
            
            result[f'{perf_metric}_kendall_mean'] = np.mean(kendall_values) if kendall_values else np.nan
            result[f'{perf_metric}_kendall_n'] = len(kendall_values)
        else:
            result[f'{perf_metric}_pearson_mean'] = np.nan
            result[f'{perf_metric}_pearson_n'] = 0
            result[f'{perf_metric}_kendall_mean'] = np.nan
            result[f'{perf_metric}_kendall_n'] = 0
    
    return result

def calculate_generationonly_correlations(data, qpp_metric_name):
    """
    Calculate correlations for generation-only nugget scores with pre-retrieval QPP metrics.
    
    Args:
        data: Consolidated query data
        qpp_metric_name: Name of the QPP metric (must be pre-retrieval)
    
    Returns:
        Dictionary with correlation results
    """
    # Collect data across all queries and variations
    correlations_per_query = {
        'generationonly_nugget_strict_vital_score': [],
        'generationonly_nugget_strict_all_score': [],
        'generationonly_nugget_vital_score': [],
        'generationonly_nugget_all_score': []
    }
    
    for query_id, query_data in data.items():
        qpp_values = []
        performance_values = {
            'generationonly_nugget_strict_vital_score': [],
            'generationonly_nugget_strict_all_score': [],
            'generationonly_nugget_vital_score': [],
            'generationonly_nugget_all_score': []
        }
        
        # Collect QPP values and performance metrics for all variations
        for reformulation in query_data['reformulations']:
            # Get QPP value (only pre-retrieval metrics)
            qpp_metrics = reformulation.get('pre_retrieval_qpp_metrics', {})
            qpp_value = qpp_metrics.get(qpp_metric_name)
            
            # Get generation-only nugget scores
            generationonly_nugget_scores = _generation_only_scores(reformulation)
            
            # Only include if QPP value is valid
            if qpp_value is not None:
                try:
                    qpp_float = float(qpp_value)
                    if not np.isnan(qpp_float):
                        qpp_values.append(qpp_float)
                        
                        # Collect performance values
                        performance_values['generationonly_nugget_strict_vital_score'].append(
                            generationonly_nugget_scores.get('strict_vital_score'))
                        performance_values['generationonly_nugget_strict_all_score'].append(
                            generationonly_nugget_scores.get('strict_all_score'))
                        performance_values['generationonly_nugget_vital_score'].append(
                            generationonly_nugget_scores.get('vital_score'))
                        performance_values['generationonly_nugget_all_score'].append(
                            generationonly_nugget_scores.get('all_score'))
                except (ValueError, TypeError):
                    pass
        
        # Calculate correlations for this query
        if len(qpp_values) >= 3:  # Need at least 3 points for correlation
            for perf_metric in correlations_per_query:
                _append_pair_correlation(
                    correlations_per_query[perf_metric],
                    qpp_values,
                    performance_values[perf_metric],
                )
    
    # Calculate average correlations across all queries
    result = {
        'qpp_metric': qpp_metric_name,
        'qpp_type': 'pre_generationonly',
        'retrieval_method': 'generationonly'
    }
    
    for perf_metric in correlations_per_query.keys():
        correlations = correlations_per_query[perf_metric]
        
        if correlations:
            pearson_values = [c['pearson'] for c in correlations if c['pearson'] is not None]
            kendall_values = [c['kendall'] for c in correlations if c['kendall'] is not None]
            
            result[f'{perf_metric}_pearson_mean'] = np.mean(pearson_values) if pearson_values else np.nan
            result[f'{perf_metric}_pearson_n'] = len(pearson_values)
            
            result[f'{perf_metric}_kendall_mean'] = np.mean(kendall_values) if kendall_values else np.nan
            result[f'{perf_metric}_kendall_n'] = len(kendall_values)
        else:
            result[f'{perf_metric}_pearson_mean'] = np.nan
            result[f'{perf_metric}_pearson_n'] = 0
            result[f'{perf_metric}_kendall_mean'] = np.nan
            result[f'{perf_metric}_kendall_n'] = 0
    
    return result

def main():
    base_dir = _REPO / "querygym"
    json_file = base_dir / "consolidated_query_data.json"
    output_dir = base_dir / "qpp_oracle_analysis"
    output_dir.mkdir(exist_ok=True)
    
    print("="*80)
    print("QPP CORRELATION ANALYSIS")
    print("="*80)
    
    print("\n📖 Loading consolidated data...")
    data = load_consolidated_data(json_file)
    print(f"✅ Loaded data for {len(data)} queries")
    
    # Get all QPP metrics
    all_qpp_metrics = get_all_qpp_metrics(data)
    print(f"\n📊 Found {len(all_qpp_metrics)} QPP metrics to analyze")
    
    # Calculate correlations for each QPP metric
    results = []
    
    print("\n🔄 Calculating correlations...")
    print("   (This may take a few minutes...)")
    
    for qpp_metric_name, qpp_type in all_qpp_metrics:
        print(f"  Processing {qpp_type}:{qpp_metric_name}...")
        result = calculate_correlations(data, qpp_metric_name, qpp_type)
        results.append(result)
    
    # Calculate generation-only correlations (only with pre-retrieval metrics)
    if _has_generation_only_scores(data):
        print("\n🔄 Calculating generation-only correlations (pre-retrieval metrics only)...")
        pre_retrieval_metrics = set()
        for query_id, query_data in data.items():
            for reformulation in query_data['reformulations']:
                pre_qpp = reformulation.get('pre_retrieval_qpp_metrics', {})
                pre_retrieval_metrics.update(pre_qpp.keys())

        for qpp_metric_name in sorted(pre_retrieval_metrics):
            print(f"  Processing generation-only: {qpp_metric_name}...")
            result = calculate_generationonly_correlations(data, qpp_metric_name)
            results.append(result)
    else:
        print("\n⏭️  Skipping generation-only correlations: no generation-only scores found")
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Split by retrieval method
    df_pyserini = df[df['retrieval_method'] == 'pyserini'].copy()
    df_cohere = df[df['retrieval_method'] == 'cohere'].copy()
    df_generationonly = df[df['retrieval_method'] == 'generationonly'].copy()
    
    # Create full output files for each retrieval method
    output_csv_pyserini_full = output_dir / "qpp_correlations_pyserini_full.csv"
    df_pyserini.to_csv(output_csv_pyserini_full, index=False)
    print(f"\n✅ Saved Pyserini full results: {output_csv_pyserini_full}")
    
    output_csv_cohere_full = output_dir / "qpp_correlations_cohere_full.csv"
    df_cohere.to_csv(output_csv_cohere_full, index=False)
    print(f"✅ Saved Cohere full results: {output_csv_cohere_full}")
    
    if len(df_generationonly) > 0:
        output_csv_generationonly_full = output_dir / "qpp_correlations_generationonly_full.csv"
        df_generationonly.to_csv(output_csv_generationonly_full, index=False)
        print(f"✅ Saved Generation-only full results: {output_csv_generationonly_full}")
    
    # Create simplified output with key metrics (all nugget metrics)
    key_columns = [
        'qpp_metric', 'qpp_type',
        'nugget_strict_vital_score_pearson_mean',
        'nugget_strict_vital_score_kendall_mean',
        'nugget_strict_all_score_pearson_mean',
        'nugget_strict_all_score_kendall_mean',
        'nugget_vital_score_pearson_mean',
        'nugget_vital_score_kendall_mean',
        'nugget_all_score_pearson_mean',
        'nugget_all_score_kendall_mean',
        'retrieval_ndcg@10_pearson_mean',
        'retrieval_ndcg@10_kendall_mean',
        'retrieval_recall@100_pearson_mean',
        'retrieval_recall@100_kendall_mean'
    ]
    
    df_pyserini_simple = df_pyserini[key_columns].copy()
    df_cohere_simple = df_cohere[key_columns].copy()
    
    # Round to 4 decimals
    for col in df_pyserini_simple.columns[2:]:
        df_pyserini_simple[col] = df_pyserini_simple[col].round(4)
        df_cohere_simple[col] = df_cohere_simple[col].round(4)
    
    output_csv_pyserini_simple = output_dir / "qpp_correlations_pyserini_simple.csv"
    df_pyserini_simple.to_csv(output_csv_pyserini_simple, index=False)
    print(f"✅ Saved Pyserini simplified results: {output_csv_pyserini_simple}")
    
    output_csv_cohere_simple = output_dir / "qpp_correlations_cohere_simple.csv"
    df_cohere_simple.to_csv(output_csv_cohere_simple, index=False)
    print(f"✅ Saved Cohere simplified results: {output_csv_cohere_simple}")
    
    # Print summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS - PYSERINI")
    print("="*80)
    
    print("\n📊 Top 10 QPP Metrics by Pearson Correlation (nugget_all_score):")
    print("-" * 80)
    top_pearson_pyserini = df_pyserini_simple.nlargest(10, 'nugget_all_score_pearson_mean')[
        ['qpp_metric', 'qpp_type', 'nugget_all_score_pearson_mean', 
         'nugget_all_score_kendall_mean', 'retrieval_ndcg@10_pearson_mean']
    ]
    print(top_pearson_pyserini.to_string(index=False))
    
    print("\n📊 Top 10 QPP Metrics by Pearson Correlation (retrieval_ndcg@10):")
    print("-" * 80)
    top_ndcg_pyserini = df_pyserini_simple.nlargest(10, 'retrieval_ndcg@10_pearson_mean')[
        ['qpp_metric', 'qpp_type', 'nugget_all_score_pearson_mean',
         'retrieval_ndcg@10_pearson_mean', 'retrieval_ndcg@10_kendall_mean']
    ]
    print(top_ndcg_pyserini.to_string(index=False))
    
    print("\n" + "="*80)
    print("SUMMARY STATISTICS - COHERE")
    print("="*80)
    
    print("\n📊 Top 10 QPP Metrics by Pearson Correlation (nugget_all_score):")
    print("-" * 80)
    top_pearson_cohere = df_cohere_simple.nlargest(10, 'nugget_all_score_pearson_mean')[
        ['qpp_metric', 'qpp_type', 'nugget_all_score_pearson_mean', 
         'nugget_all_score_kendall_mean', 'retrieval_ndcg@10_pearson_mean']
    ]
    print(top_pearson_cohere.to_string(index=False))
    
    print("\n📊 Top 10 QPP Metrics by Pearson Correlation (retrieval_ndcg@10):")
    print("-" * 80)
    top_ndcg_cohere = df_cohere_simple.nlargest(10, 'retrieval_ndcg@10_pearson_mean')[
        ['qpp_metric', 'qpp_type', 'nugget_all_score_pearson_mean',
         'retrieval_ndcg@10_pearson_mean', 'retrieval_ndcg@10_kendall_mean']
    ]
    print(top_ndcg_cohere.to_string(index=False))
    
    # Show QSDQPP specifically
    print("\n" + "="*80)
    print("QSDQPP CORRELATION RESULTS")
    print("="*80)
    qsdqpp_pyserini = df_pyserini_simple[df_pyserini_simple['qpp_metric'] == 'qsdqpp_predicted_ndcg']
    qsdqpp_cohere = df_cohere_simple[df_cohere_simple['qpp_metric'] == 'qsdqpp_predicted_ndcg']
    
    if not qsdqpp_pyserini.empty:
        print("\nPYSERINI:")
        print(qsdqpp_pyserini.to_string(index=False))
    if not qsdqpp_cohere.empty:
        print("\nCOHERE:")
        print(qsdqpp_cohere.to_string(index=False))
    if qsdqpp_pyserini.empty and qsdqpp_cohere.empty:
        print("QSDQPP not found in results")
    
    print(f"\n🎉 Analysis complete! Results saved to: {output_dir}")
    print(f"   - Pyserini full: {output_csv_pyserini_full.name}")
    print(f"   - Pyserini simple: {output_csv_pyserini_simple.name}")
    print(f"   - Cohere full: {output_csv_cohere_full.name}")
    print(f"   - Cohere simple: {output_csv_cohere_simple.name}")

if __name__ == "__main__":
    main()
