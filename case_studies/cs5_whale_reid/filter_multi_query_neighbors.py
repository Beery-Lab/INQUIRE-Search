#!/usr/bin/env python3
"""
Filter query_to_neighbors.csv to keep only entries where the same neighbor (label or annot)
appears for multiple different queries.

Usage:
    python filter_multi_query_neighbors.py --csv query_to_neighbors.csv --out filtered_neighbors.csv --by label
    python filter_multi_query_neighbors.py --csv query_to_neighbors.csv --out filtered_neighbors.csv --by annot
"""

import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Filter neighbors that match multiple queries"
    )
    parser.add_argument(
        "--csv",
        default="query_to_neighbors.csv",
        help="Input CSV path (default: query_to_neighbors.csv)"
    )
    parser.add_argument(
        "--out",
        default="filtered_neighbors.csv",
        help="Output CSV path (default: filtered_neighbors.csv)"
    )
    parser.add_argument(
        "--by",
        choices=["label", "annot"],
        default="annot",
        help="Group by neighbor_label or neighbor_annot (default: annot, since labels are empty)"
    )
    parser.add_argument(
        "--min-queries",
        type=int,
        default=2,
        help="Minimum number of unique queries a neighbor must match (default: 2)"
    )
    args = parser.parse_args()

    # Load CSV
    df = pd.read_csv(args.csv)
    
    print(f"Original CSV: {len(df)} rows, {df['query_annot'].nunique()} unique queries")
    
    # Choose grouping column
    if args.by == "label":
        group_col = "neighbor_label"
        # Filter out rows with empty/null labels
        df_valid = df[df[group_col].notna() & (df[group_col] != "")]
        if len(df_valid) == 0:
            print(f"\nWARNING: No non-empty {group_col} values found!")
            print("Falling back to grouping by neighbor_annot instead.")
            group_col = "neighbor_annot"
            df_valid = df
    else:
        group_col = "neighbor_annot"
        df_valid = df
    
    # Count how many unique queries each neighbor appears in
    neighbor_query_counts = (
        df_valid.groupby(group_col)['query_annot']
        .nunique()
        .reset_index()
        .rename(columns={'query_annot': 'num_queries'})
    )
    
    # Keep only neighbors that match >= min_queries different queries
    multi_query_neighbors = neighbor_query_counts[
        neighbor_query_counts['num_queries'] >= args.min_queries
    ][group_col].tolist()
    
    print(f"\nFound {len(multi_query_neighbors)} neighbors appearing in {args.min_queries}+ queries")
    
    # Filter original dataframe
    filtered_df = df_valid[df_valid[group_col].isin(multi_query_neighbors)]
    
    # Sort by neighbor and then by query for readability
    filtered_df = filtered_df.sort_values([group_col, 'query_annot', 'rank'])
    
    # Save
    filtered_df.to_csv(args.out, index=False)
    
    print(f"\nFiltered CSV: {len(filtered_df)} rows, {filtered_df['query_annot'].nunique()} unique queries")
    print(f"Saved to {args.out}")
    
    # Show summary stats
    print("\nTop neighbors by number of queries matched:")
    top_neighbors = (
        filtered_df.groupby(group_col)['query_annot']
        .nunique()
        .sort_values(ascending=False)
        .head(10)
    )
    for neighbor, count in top_neighbors.items():
        print(f"  {neighbor}: {count} queries")


if __name__ == '__main__':
    main()