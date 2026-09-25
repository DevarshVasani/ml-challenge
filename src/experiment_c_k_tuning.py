import pandas as pd
import time
import json
from pathlib import Path
from candidates import load_source, generate_candidates, load_ground_truth, candidate_diagnostics

def slice_candidates(cands: pd.DataFrame, k: int) -> pd.DataFrame:
    # Keep row if it was retrieved within top K for either name or address
    mask_name = (cands["name_rank"] > 0) & (cands["name_rank"] <= k)
    mask_address = (cands["address_rank"] > 0) & (cands["address_rank"] <= k)
    sliced = cands[mask_name | mask_address].copy()
    
    # Update flags
    sliced["retrieved_by_name"] = mask_name.astype(int)
    sliced["retrieved_by_address"] = mask_address.astype(int)
    
    return sliced

def main():
    print("Loading data subset (N=5000)...")
    s1 = load_source("student_resource/dataset/train/train_source1.tsv", 5000)
    s2 = load_source("student_resource/dataset/train/train_source2.tsv")
    s3 = load_source("student_resource/dataset/train/train_source3.tsv")
    gt = load_ground_truth("student_resource/dataset/train/train_ground_truth.tsv")
    
    # Filter ground truth to only include S1 IDs in our subset
    s1_ids = set(s1["entity_id"].astype(str))
    gt = {k: v for k, v in gt.items() if k in s1_ids}

    print("Generating pool of top-100 candidates...")
    start = time.perf_counter()
    pool_100 = generate_candidates(s1, s2, s3, top_k_name=100, top_k_address=100, batch_size=2048)
    runtime = time.perf_counter() - start
    print(f"Pool generated in {runtime:.1f}s")

    Path("artifacts/c_experiments").mkdir(parents=True, exist_ok=True)
    
    results = {}
    for k in [25, 50, 100]:
        print(f"\n--- Deriving subset for top-{k} ---")
        cands_k = slice_candidates(pool_100, k)
        report = candidate_diagnostics(cands_k, s1, s2, s3, gt)
        
        recall = report.get('candidate_recall_overall', 0)
        oracle = report.get('oracle_macro_f05', 0)
        avg_cands = report.get('average_candidates_per_s1', 0)
        
        print(f"Top-{k} Recall: {recall:.4f}")
        print(f"Oracle F0.5:  {oracle:.4f}")
        print(f"Avg Cands/S1: {avg_cands:.1f}")
        
        results[f"top_{k}"] = report
        
    with open("artifacts/c_experiments/k_tuning_report.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
