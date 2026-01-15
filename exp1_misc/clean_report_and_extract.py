"""
EEG Reports Processing with a given model
=========================================
This script loads EEG reports from CSV, cleans the data, and generates
embeddings using a given model.
"""

CSV_PATH = './asm_data/alfred_1st_regimen.csv'
MODEL_STRING = "NeuML/pubmedbert-base-embeddings"

import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print(f"EEG Reports Processing with {MODEL_STRING}")
print("="*70)

# ============================================================================
# 1. LOAD THE DATA
# ============================================================================
print("\n[1] Loading data...")
df = pd.read_csv(CSV_PATH)

print(f"Initial dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"\nFirst few rows of eeg_report column:")
print(df['eeg_report'].head())

# ============================================================================
# 2. DATA EXPLORATION
# ============================================================================
print("\n[2] Data Exploration...")

# Check for missing values in eeg_report column
missing_count = df['eeg_report'].isna().sum()
print(f"Missing values in eeg_report: {missing_count} ({missing_count/len(df)*100:.1f}%)")

# Check report lengths
df['report_length'] = df['eeg_report'].fillna('').str.len()
print(f"\nReport length statistics:")
print(df['report_length'].describe())

# Visualize distribution
print(f"\nReports by length category:")
print(f"  Empty (0 chars): {(df['report_length'] == 0).sum()}")
print(f"  Very short (1-50 chars): {((df['report_length'] > 0) & (df['report_length'] <= 50)).sum()}")
print(f"  Short (51-200 chars): {((df['report_length'] > 50) & (df['report_length'] <= 200)).sum()}")
print(f"  Medium (201-500 chars): {((df['report_length'] > 200) & (df['report_length'] <= 500)).sum()}")
print(f"  Long (500+ chars): {(df['report_length'] > 500).sum()}")

# ============================================================================
# 3. DATA CLEANING
# ============================================================================
print("\n[3] Cleaning data...")

# Create a copy for processing
df_clean = df.copy()

# Remove rows where eeg_report is NaN or empty
df_clean = df_clean[df_clean['eeg_report'].notna()]
df_clean = df_clean[df_clean['eeg_report'].str.strip() != '']

print(f"After removing NaN/empty: {len(df_clean)} rows remaining")

# Define minimum report length (adjustable threshold)
MIN_REPORT_LENGTH = 20  # characters

# Filter out reports that are too short
df_clean = df_clean[df_clean['report_length'] >= MIN_REPORT_LENGTH]
print(f"After removing reports < {MIN_REPORT_LENGTH} chars: {len(df_clean)} rows remaining")

# Remove any error messages that might be in the reports
error_patterns = ['Err:', 'Exceed time window', '#N/A', 'No EEG data']
for pattern in error_patterns:
    before = len(df_clean)
    df_clean = df_clean[~df_clean['eeg_report'].str.contains(pattern, na=False)]
    removed = before - len(df_clean)
    if removed > 0:
        print(f"  Removed {removed} rows containing '{pattern}'")

print(f"\nFinal cleaned dataset: {len(df_clean)} valid EEG reports")

# Reset index
df_clean = df_clean.reset_index(drop=True)

# ============================================================================
# 4. INITIALIZE CLINICALBERT
# ============================================================================
print("\n[4] Loading ClinicalBERT model...")

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_STRING)
model = AutoModel.from_pretrained(MODEL_STRING)

# Set to evaluation mode
model.eval()

# Move to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
print(f"Using device: {device}")

# ============================================================================
# 5. FUNCTION TO GET EMBEDDINGS
# ============================================================================

def get_embeddings_batch(texts, batch_size=16, max_length=512, pooling='mean'):
    """
    Get embeddings for a list of texts using ClinicalBERT
    
    Args:
        texts: list of strings
        batch_size: number of texts to process at once
        max_length: maximum sequence length (default 512 for ClinicalBERT)
        pooling: 'mean' or 'cls'
    
    Returns:
        numpy array of shape (len(texts), 768)
    """
    all_embeddings = []
    
    # Process in batches
    for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
        batch_texts = texts[i:i + batch_size]
        
        # Tokenize
        inputs = tokenizer(
            batch_texts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=max_length
        )
        
        # Move to device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Get embeddings
        with torch.no_grad():
            outputs = model(**inputs)
        
        if pooling == 'cls':
            # Use [CLS] token (first token)
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        else:
            # Mean pooling (recommended)
            attention_mask = inputs['attention_mask']
            token_embeddings = outputs.last_hidden_state
            
            # Expand attention mask for broadcasting
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(
                token_embeddings.size()
            ).float()
            
            # Sum embeddings, weighted by attention mask
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            
            # Mean pooling
            embeddings = (sum_embeddings / sum_mask).cpu().numpy()
        
        all_embeddings.append(embeddings)
    
    # Concatenate all batches
    return np.vstack(all_embeddings)

# ============================================================================
# 6. GENERATE EMBEDDINGS
# ============================================================================
print("\n[5] Generating embeddings for cleaned reports...")

# Extract report texts as a list
report_texts = df_clean['eeg_report'].tolist()

# Generate embeddings
embeddings = get_embeddings_batch(
    report_texts,
    batch_size=16,
    pooling='mean'  # Use mean pooling (generally better than CLS)
)

print(f"\nGenerated embeddings shape: {embeddings.shape}")
print(f"  - Number of reports: {embeddings.shape[0]}")
print(f"  - Embedding dimension: {embeddings.shape[1]}")

# ============================================================================
# 7. ADD EMBEDDINGS TO DATAFRAME (OPTIONAL)
# ============================================================================
print("\n[6] Saving results...")

# Save embeddings as separate numpy file
np.save('eeg_embeddings.npy', embeddings)
print("Embeddings saved to: eeg_embeddings.npy")

# ============================================================================
# 8. EXAMPLE: COMPUTE SIMILARITY BETWEEN REPORTS
# ============================================================================
print("\n[7] Example: Computing similarity between reports...")

from sklearn.metrics.pairwise import cosine_similarity

# Compute similarity matrix for first 5 reports
n_sample = min(5, len(embeddings))
sample_embeddings = embeddings[:n_sample]
similarity_matrix = cosine_similarity(sample_embeddings)

print(f"\nSimilarity matrix (first {n_sample} reports):")
print(similarity_matrix.round(3))

print("\nFirst few report snippets:")
for i in range(n_sample):
    snippet = report_texts[i][:100] + "..." if len(report_texts[i]) > 100 else report_texts[i]
    print(f"\n[Report {i}] {snippet}")

# ============================================================================
# 9. SUMMARY STATISTICS
# ============================================================================
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"Original dataset: {len(df)} reports")
print(f"Cleaned dataset: {len(df_clean)} reports ({len(df_clean)/len(df)*100:.1f}%)")
print(f"Embeddings generated: {embeddings.shape[0]} x {embeddings.shape[1]}")
print(f"Mean embedding norm: {np.linalg.norm(embeddings, axis=1).mean():.2f}")
print(f"Std embedding norm: {np.linalg.norm(embeddings, axis=1).std():.2f}")
print("\nFiles saved:")
print("  - eeg_embeddings.npy")
print("="*70)