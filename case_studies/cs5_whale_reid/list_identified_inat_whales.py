#!/usr/bin/env python3
"""
Generate a list of iNaturalist photos that correspond to individually identifiable whales.

This script reads query_to_neighbors.csv and extracts all unique query_url entries
where query_label is not empty (indicating the whale has been individually identified).

Usage:
    python list_identified_inat_whales.py --csv query_to_neighbors.csv --out identified_whales.txt
"""

import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="List iNaturalist photos of individually identifiable whales"
    )
    parser.add_argument(
        "--csv",
        default="query_to_neighbors.csv",
        help="Path to query_to_neighbors.csv (default: query_to_neighbors.csv)"
    )
    parser.add_argument(
        "--out",
        default="identified_whales.txt",
        help="Output file path (default: identified_whales.txt)"
    )
    parser.add_argument(
        "--format",
        choices=["txt", "csv"],
        default="txt",
        help="Output format: txt (URLs only) or csv (with labels) (default: txt)"
    )
    args = parser.parse_args()

    # Load CSV
    df = pd.read_csv(args.csv)
    
    # Filter to queries with non-null labels (individually identifiable)
    # Group by query to get unique queries
    queries = df[['query_annot', 'query_label', 'query_url']].drop_duplicates()
    identified = queries[queries['query_label'].notna() & (queries['query_label'] != '')]
    
    print(f"Total queries: {len(queries)}")
    print(f"Identified whales (with labels): {len(identified)}")
    print(f"Unidentified whales: {len(queries) - len(identified)}")
    
    if args.format == "txt":
        # Write URLs only
        urls = identified['query_url'].dropna().tolist()
        with open(args.out, 'w') as f:
            for url in urls:
                f.write(f"{url}\n")
        print(f"\nWrote {len(urls)} iNaturalist URLs to {args.out}")
    else:
        # Write CSV with annotations, labels, and URLs
        identified.to_csv(args.out, index=False)
        print(f"\nWrote {len(identified)} entries to {args.out}")
        print("Columns: query_annot, query_label, query_url")


if __name__ == '__main__':
    main()