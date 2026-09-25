"""Diagnostic script for analyzing ByT5 truncation lengths and exact field cut locations."""

import argparse
import json
import logging
from collections import Counter
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer

from src.neural_adapters.base import PAIR_TEXT_FIELDS, serialize_pair_fields

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def main():
    parser = argparse.ArgumentParser(description="Analyze truncation lengths for ByT5 serialization")
    parser.add_argument("--pair-manifest", type=str, required=True, help="Path to pair manifest JSON")
    parser.add_argument("--pair-dir", type=str, required=True, help="Path to pair parquet directory")
    parser.add_argument("--checkpoint", type=str, default="google/byt5-small", help="Tokenizer checkpoint")
    parser.add_argument("--max-length", type=int, default=512, help="Max byte length for ByT5")
    parser.add_argument("--output", type=str, required=True, help="Path to output JSON report")
    
    args = parser.parse_args()
    
    logging.info(f"Loading tokenizer from {args.checkpoint}")
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, use_fast=False)
    
    manifest = json.loads(Path(args.pair_manifest).read_text(encoding="utf-8"))
    shards = manifest.get("shard_order", [])
    if not shards:
        shards = [p.name for p in Path(args.pair_dir).glob("*.parquet")]
        
    total_pairs = 0
    truncated_pairs = 0
    lengths = []
    
    # Counter for where truncation happens
    cut_locations = Counter()
    
    for shard in tqdm(shards, desc="Processing Shards"):
        df = pd.read_parquet(Path(args.pair_dir) / shard)
        
        for idx, row in df.iterrows():
            row_dict = row.to_dict()
            full_text = serialize_pair_fields(row_dict)
            
            tokenized = tokenizer(full_text, add_special_tokens=True, truncation=False)["input_ids"]
            length = len(tokenized)
            lengths.append(length)
            
            if length > args.max_length:
                truncated_pairs += 1
                
                # Determine which field got cut
                cumulative_len = 0
                cut_field = "unknown"
                for field in PAIR_TEXT_FIELDS:
                    val_str = str(row_dict.get(field, "") or "")
                    field_tokens = tokenizer(val_str, add_special_tokens=False)["input_ids"]
                    
                    # Approximating byte length plus separator
                    cumulative_len += len(field_tokens) + len(tokenizer(" || ", add_special_tokens=False)["input_ids"])
                    if cumulative_len >= args.max_length:
                        cut_field = field
                        break
                
                cut_locations[cut_field] += 1
                
    lengths_series = pd.Series(lengths)
    
    report = {
        "total_pairs": len(lengths_series),
        "truncated_pairs": truncated_pairs,
        "truncation_rate": float(truncated_pairs / len(lengths_series)) if len(lengths_series) else 0.0,
        "token_length_stats": {
            "p50": float(lengths_series.quantile(0.50)),
            "p90": float(lengths_series.quantile(0.90)),
            "p95": float(lengths_series.quantile(0.95)),
            "p99": float(lengths_series.quantile(0.99)),
            "max": float(lengths_series.max()),
        },
        "cut_locations": dict(cut_locations)
    }
    
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2))
    logging.info(f"Saved truncation report to {args.output}")

if __name__ == "__main__":
    main()
