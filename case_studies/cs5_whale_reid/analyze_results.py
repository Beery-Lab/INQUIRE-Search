#!/usr/bin/env python3
"""
Utility script to analyze matching results.

Provides various analyses of the matching results including:
- Overall accuracy metrics (if ground truth available)
- Distribution of distances
- Per-rank statistics
- Visualization of example matches
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_results(csv_path):
    """Load results CSV."""
    if not Path(csv_path).exists():
        raise FileNotFoundError(f"Results file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} matches from {csv_path}")
    return df

def compute_accuracy_metrics(df):
    """
    Compute accuracy metrics if labels are available.
    
    Args:
        df: Results DataFrame
        
    Returns:
        dict: Accuracy metrics
    """
    # Check if we have labels
    if 'query_label' not in df.columns or 'neighbor_label' not in df.columns:
        print("Warning: Labels not available in results.")
        return None
    
    # Remove rows with missing labels
    df_labeled = df.dropna(subset=['query_label', 'neighbor_label'])
    
    if len(df_labeled) == 0:
        print("Warning: No labeled matches found.")
        return None
    
    print(f"\nAnalyzing {len(df_labeled)} labeled matches...")
    
    # Add correctness column
    df_labeled['is_correct'] = df_labeled['query_label'] == df_labeled['neighbor_label']
    
    # Overall accuracy
    total_queries = df_labeled['query_annot'].nunique()
    
    # Top-1 accuracy
    top1_df = df_labeled[df_labeled['rank'] == 1]
    top1_correct = top1_df['is_correct'].sum()
    top1_acc = top1_correct / len(top1_df) if len(top1_df) > 0 else 0
    
    # Top-3 accuracy (any correct in top-3)
    queries_with_correct = df_labeled[df_labeled['is_correct']].groupby('query_annot').size()
    top3_correct = len(queries_with_correct)
    top3_acc = top3_correct / total_queries if total_queries > 0 else 0
    
    metrics = {
        'total_queries': total_queries,
        'top1_accuracy': top1_acc,
        'top1_correct': top1_correct,
        'top1_total': len(top1_df),
        'top3_accuracy': top3_acc,
        'top3_correct': top3_correct,
        'top3_total': total_queries,
    }
    
    return metrics

def print_accuracy_metrics(metrics):
    """Print accuracy metrics."""
    if metrics is None:
        return
    
    print(f"\n{'='*60}")
    print("ACCURACY METRICS")
    print(f"{'='*60}")
    print(f"Total queries:          {metrics['total_queries']}")
    print(f"\nTop-1 Accuracy:         {metrics['top1_accuracy']:.2%} ({metrics['top1_correct']}/{metrics['top1_total']})")
    print(f"Top-3 Accuracy:         {metrics['top3_accuracy']:.2%} ({metrics['top3_correct']}/{metrics['top3_total']})")
    print(f"{'='*60}")

def analyze_distances(df):
    """Analyze distance distributions."""
    print(f"\n{'='*60}")
    print("DISTANCE STATISTICS")
    print(f"{'='*60}")
    
    for rank in sorted(df['rank'].unique()):
        rank_df = df[df['rank'] == rank]
        distances = rank_df['distance']
        
        print(f"\nRank {rank}:")
        print(f"  Mean distance:   {distances.mean():.4f}")
        print(f"  Median distance: {distances.median():.4f}")
        print(f"  Min distance:    {distances.min():.4f}")
        print(f"  Max distance:    {distances.max():.4f}")
    
    print(f"{'='*60}")

def plot_distance_distribution(df, output_path="distance_distribution.png"):
    """Plot distance distribution by rank."""
    ranks = sorted(df['rank'].unique())
    
    fig, axes = plt.subplots(1, len(ranks), figsize=(5*len(ranks), 4))
    if len(ranks) == 1:
        axes = [axes]
    
    for idx, rank in enumerate(ranks):
        rank_df = df[df['rank'] == rank]
        axes[idx].hist(rank_df['distance'], bins=30, edgecolor='black')
        axes[idx].set_title(f'Rank {rank}')
        axes[idx].set_xlabel('Distance')
        axes[idx].set_ylabel('Count')
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved distance distribution plot to: {output_path}")
    plt.close()

def show_example_matches(df, n_examples=5, save_path=None):
    """Show example matches."""
    print(f"\n{'='*60}")
    print(f"EXAMPLE MATCHES (showing {n_examples})")
    print(f"{'='*60}")
    
    # Get unique queries
    queries = df['query_annot'].unique()[:n_examples]
    
    for query in queries:
        query_matches = df[df['query_annot'] == query].sort_values('rank')
        
        print(f"\nQuery: {query}")
        if 'query_label' in df.columns:
            q_label = query_matches.iloc[0]['query_label']
            print(f"  Label: {q_label}")
        if 'query_url' in df.columns:
            q_url = query_matches.iloc[0]['query_url']
            print(f"  URL: {q_url}")
        
        print(f"  Top matches:")
        for _, match in query_matches.iterrows():
            is_correct = ""
            if 'query_label' in df.columns and 'neighbor_label' in df.columns:
                if pd.notna(match['query_label']) and pd.notna(match['neighbor_label']):
                    is_correct = " ✓" if match['query_label'] == match['neighbor_label'] else " ✗"
            
            print(f"    {match['rank']}. {match['neighbor_annot']} "
                  f"(dist={match['distance']:.4f}){is_correct}")
            if pd.notna(match['neighbor_label']):
                print(f"       Label: {match['neighbor_label']}")
    
    print(f"{'='*60}")

def main():
    parser = argparse.ArgumentParser(description="Analyze whale re-identification results")
    parser.add_argument("--results", type=str, default="matches.csv",
                        help="Path to results CSV")
    parser.add_argument("--plot", action="store_true",
                        help="Generate plots")
    parser.add_argument("--examples", type=int, default=5,
                        help="Number of example matches to show")
    parser.add_argument("--output-dir", type=str, default="analysis",
                        help="Directory for output plots")
    
    args = parser.parse_args()
    
    # Load results
    df = load_results(args.results)
    
    # Compute accuracy metrics
    metrics = compute_accuracy_metrics(df)
    print_accuracy_metrics(metrics)
    
    # Analyze distances
    analyze_distances(df)
    
    # Show examples
    show_example_matches(df, n_examples=args.examples)
    
    # Generate plots
    if args.plot:
        import os
        os.makedirs(args.output_dir, exist_ok=True)
        
        plot_path = os.path.join(args.output_dir, "distance_distribution.png")
        plot_distance_distribution(df, plot_path)
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()
