import pandas as pd
import numpy as np
import re
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import argparse

def normalize_text(text):
    """
    Create normalized copies with consistent case, punctuation, and whitespace; 
    keep numbers and Unicode characters.
    """
    if pd.isna(text):
        return ""
    # Lowercase
    text = str(text).lower()
    # Remove punctuation (keep unicode letters and numbers which \w covers, and whitespace \s)
    # Actually, \w includes underscore _, so let's be more precise or just replace non-alphanumeric.
    text = re.sub(r'[^\w\s]', ' ', text)
    # Remove underscores as they are often treated as punctuation
    text = text.replace('_', ' ')
    # Consistent whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def retrieve_candidates(s1_df, source_df, column, n_candidates=50):
    """
    Prototype character n-gram TF-IDF retrieval.
    Compares short character sequences.
    """
    start_time = time.time()
    
    # Vectorizer for character n-grams (3-grams is a good default for typos)
    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 4), min_df=2)
    
    # Fit on the source data (we want to search inside source_df)
    source_texts = source_df[column].fillna('')
    s1_texts = s1_df[column].fillna('')
    
    print(f"Vectorizing {column}...")
    # To avoid data leakage, we can fit on both or just the corpus we are searching in.
    # Fitting on both gives a shared vocabulary.
    vectorizer.fit(pd.concat([s1_texts, source_texts]))
    
    source_tfidf = vectorizer.transform(source_texts)
    s1_tfidf = vectorizer.transform(s1_texts)
    
    print(f"Building index and retrieving candidates for {column}...")
    # Use NearestNeighbors for cosine similarity retrieval
    # cosine similarity is equivalent to euclidean distance on L2 normalized vectors
    # TfidfVectorizer returns L2 normalized vectors by default.
    nbrs = NearestNeighbors(n_neighbors=min(n_candidates, len(source_df)), metric='cosine', n_jobs=-1)
    nbrs.fit(source_tfidf)
    
    distances, indices = nbrs.kneighbors(s1_tfidf)
    
    candidates = []
    # indices shape is (len(s1_df), n_candidates)
    s1_ids = s1_df['entity_id'].values
    source_ids = source_df['entity_id'].values
    
    for i, s1_id in enumerate(s1_ids):
        for j in range(indices.shape[1]):
            # Optional: threshold on distance
            candidates.append({
                'source1_entity_id': s1_id,
                'candidate_entity_id': source_ids[indices[i, j]],
                'score': 1.0 - distances[i, j] # Convert distance to similarity
            })
            
    retrieval_time = time.time() - start_time
    print(f"Retrieval for {column} took {retrieval_time:.2f} seconds")
    return pd.DataFrame(candidates)

def evaluate_recall(candidates_df, ground_truth_df):
    """
    Measure candidate recall: true matches retrieved / all true matches.
    """
    # ground_truth_df has 'source1_entity_id' and 'matched_entity_ids' (comma separated)
    # Expand ground truth to pairwise
    gt_pairs = []
    for _, row in ground_truth_df.iterrows():
        s1_id = row['source1_entity_id']
        matches = str(row['matched_entity_ids']).split(',')
        for m in matches:
            if m.strip() and m.strip() != 'nan':
                gt_pairs.append({'source1_entity_id': s1_id, 'candidate_entity_id': m.strip()})
                
    gt_df = pd.DataFrame(gt_pairs)
    if len(gt_df) == 0:
        return 0.0, 0
        
    # Merge to find retrieved ones
    retrieved = pd.merge(gt_df, candidates_df, on=['source1_entity_id', 'candidate_entity_id'], how='inner')
    
    recall = len(retrieved) / len(gt_df)
    return recall, len(gt_df)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--s1', type=str, default='student_resource/dataset/train/train_source1.tsv')
    parser.add_argument('--s2', type=str, default='student_resource/dataset/train/train_source2.tsv')
    parser.add_argument('--s3', type=str, default='student_resource/dataset/train/train_source3.tsv')
    parser.add_argument('--gt', type=str, default='student_resource/dataset/train/train_ground_truth.tsv')
    parser.add_argument('--subset', type=int, default=1000, help='Subset size for prototyping')
    args = parser.parse_args()

    print("Loading datasets...")
    try:
        s1 = pd.read_csv(args.s1, sep='\t', dtype=str).head(args.subset)
        s2 = pd.read_csv(args.s2, sep='\t', dtype=str)
        s3 = pd.read_csv(args.s3, sep='\t', dtype=str)
        gt = pd.read_csv(args.gt, sep='\t', dtype=str)
    except FileNotFoundError:
        print("Datasets not found. Please ensure the dataset is downloaded to dataset/train/")
        return

    # Normalize texts
    print("Normalizing texts...")
    for df in [s1, s2, s3]:
        df['norm_name'] = df['business_name'].apply(normalize_text)
        df['norm_address'] = df['business_address'].apply(normalize_text)

    # Process Source 2
    print("\n--- Source 2 Retrieval ---")
    cands_s2_name = retrieve_candidates(s1, s2, 'norm_name', n_candidates=50)
    cands_s2_addr = retrieve_candidates(s1, s2, 'norm_address', n_candidates=50)
    
    # Process Source 3
    print("\n--- Source 3 Retrieval ---")
    cands_s3_name = retrieve_candidates(s1, s3, 'norm_name', n_candidates=50)
    cands_s3_addr = retrieve_candidates(s1, s3, 'norm_address', n_candidates=50)
    
    # Combine all candidates
    all_cands = pd.concat([cands_s2_name, cands_s2_addr, cands_s3_name, cands_s3_addr])
    
    # Remove duplicates
    all_cands = all_cands.groupby(['source1_entity_id', 'candidate_entity_id']).agg({'score': 'max'}).reset_index()
    
    print(f"\nTotal unique candidate pairs generated: {len(all_cands)}")
    avg_cands = len(all_cands) / len(s1)
    print(f"Average candidates per S1 entity: {avg_cands:.2f}")

    # Evaluate Recall
    recall, total_gt = evaluate_recall(all_cands, gt)
    print(f"\nRecall on subset: {recall:.4f} ({int(recall * total_gt)} / {total_gt} true matches retrieved)")
    
    # Save handoff format
    all_cands[['source1_entity_id', 'candidate_entity_id']].to_csv('internal_candidate_table.tsv', sep='\t', index=False)
    print("Saved internal_candidate_table.tsv for Member 3.")

if __name__ == '__main__':
    main()
