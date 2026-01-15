"""
SMILES Embeddings with ChemBERTa
================================
This script generates SMILES embeddings for anti-seizure medications
using ChemBERTa (HuggingFace). Hard coded SMILES string are retrived 
from DrugBank (https://go.drugbank.com/).
"""

from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import torch

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================
# Available models:
MODEL_STRING = "seyonec/PubChem10M_SMILES_BPE_450k"  # trained on 10M SMILES
# MODEL_STRING = "seyonec/ChemBERTa-zinc-base-v1"    # trained on ZINC
# MODEL_STRING = "unikei/bert-base-smiles"           # BERT architecture

# Load ChemBERTa's tokenizer and model
print(f"LOADING CHEMBERTA MODEL FOR SMILES EMBEDDING")
print(f"  Model: {MODEL_STRING}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_STRING)
model = AutoModel.from_pretrained(MODEL_STRING)

# Set model to evaluation mode (disables dropout, etc.)
model.eval()
print(f"\tComplete\n")


def get_smiles_embeddings(smiles, pooling='mean'):
    """
    Get embeddings from ChemBERTa for SMILES strings.
        
    Args:
        smiles: str or list of SMILES strings
        pooling: 'mean' or 'cls'
    
    Returns:
        torch.Tensor of shape [num_smiles, 768]
    """
    # Ensure smiles is a list
    if isinstance(smiles, str):
        smiles = [smiles]
    
    # Tokenize
    inputs = tokenizer(
        smiles,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=512
    )
    
    # Get model outputs
    with torch.no_grad():
        outputs = model(**inputs)
    
    if pooling == 'cls':
        # Use [CLS] token
        embeddings = outputs.last_hidden_state[:, 0, :]
    else:
        # Mean pooling (recommended for SMILES)
        attention_mask = inputs['attention_mask']
        token_embeddings = outputs.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        embeddings = sum_embeddings / sum_mask
    
    return embeddings


# =============================================================================
# ANTI-SEIZURE MEDICATION SMILES DATABASE
# =============================================================================
# SMILES obtained from DrugBank (https://go.drugbank.com/)
ASM_SMILES = {
    # First-line ASMs
    'Levetiracetam': 'CC[C@H](N1CCCC1=O)C(N)=O',
    'Valproic_acid': 'CCCC(CCC)C(O)=O',
    'Carbamazepine': 'NC(=O)N1C2=CC=CC=C2C=CC2=CC=CC=C12',
    'Lamotrigine': 'NC1=NC(N)=C(N=N1)C1=C(Cl)C(Cl)=CC=C1',
    'Oxcarbazepine': 'NC(=O)N1C2=CC=CC=C2CC(=O)C2=C1C=CC=C2',
    
    # Second-line ASMs  
    'Topiramate': '[H][C@@]12CO[C@@]3(COS(N)(=O)=O)OC(C)(C)O[C@@]3([H])[C@]1([H])OC(C)(C)O2',
    'Phenytoin': 'O=C1NC(=O)C(N1)(C1=CC=CC=C1)C1=CC=CC=C1',
    'Lacosamide': 'COC[C@@H](NC(C)=O)C(=O)NCC1=CC=CC=C1',
    'Brivaracetam': 'CCC[C@H]1CN([C@@H](CC)C(N)=O)C(=O)C1',
    'Perampanel': 'O=C1N(C=C(C=C1C1=CC=CC=C1C#N)C1=NC=CC=C1)C1=CC=CC=C1',
    
    # Adjunctive therapies
    'Zonisamide': 'NS(=O)(=O)CC1=NOC2=CC=CC=C12',
    'Gabapentin': 'NCC1(CC(O)=O)CCCCC1',
    'Pregabalin': 'CC(C)C[C@H](CN)CC(O)=O',
    'Clobazam': 'CN1C2=C(C=C(Cl)C=C2)N(C2=CC=CC=C2)C(=O)CC1=O',
    'Clonazepam': '[O-][N+](=O)C1=CC2=C(NC(=O)CN=C2C2=CC=CC=C2Cl)C=C1',
    
    # Older ASMs
    # 'Phenobarbital': '',
    # 'Ethosuximide': '',
    # 'Primidone': '',
    
    # Specialty ASMs
    # 'Felbamate': '',
    # 'Vigabatrin': '',
    # 'Cannabidiol': '',
    # 'Stiripentol': '',
}


def test_base_functionality():
    """Test basic SMILES embedding functionality."""
    smiles = ASM_SMILES['Levetiracetam']  
    print(f"TESTING BASELINE SMILES EMBEDDING FUNCTION")
    print(f"  Input SMILES: {smiles}")
    embedding = get_smiles_embeddings(smiles)
    assert embedding.shape == torch.Size([1, 768]), \
        f"Assertion invalid, expected shape [1, 768] but got {embedding.shape}"
    print(f"\tEmbedding shape correct: {embedding.shape}")
    print(f"\tComplete\n")


def test_cosine_similarity():
    """Test similarity between related drugs."""
    print(f"TESTING COSINE SIMILARITY FOR RELATED ASMs")
    
    # Compare structurally related drugs
    related_drugs = [
        ('Levetiracetam', ASM_SMILES['Levetiracetam']),
        ('Brivaracetam', ASM_SMILES['Brivaracetam']),    # Structural analogue
        ('Valproic_acid', ASM_SMILES['Valproic_acid']),  # Different class
    ]
    
    drug_names = [d[0] for d in related_drugs]
    smiles_list = [d[1] for d in related_drugs]
    
    embeddings = get_smiles_embeddings(smiles_list).numpy()
    
    # Compute similarity matrix
    similarities = cosine_similarity(embeddings)
    print("\tDrugs tested:")
    for name, sm in related_drugs:
        print(f"\t  - {name}: {sm[:30]}...")
    
    print("\n\tSimilarity matrix:")
    print(f"\t{'':<15}", end='')
    for name in drug_names:
        print(f"{name[:12]:<15}", end='')
    print()
    
    for i, name in enumerate(drug_names):
        print(f"\t{name[:15]:<15}", end='')
        for j in range(len(drug_names)):
            print(f"{similarities[i, j]:.4f}         ", end='')
        print()
    
    # Expected: Levetiracetam and Brivaracetam should be most similar
    lev_briv_sim = similarities[0, 1]
    lev_valp_sim = similarities[0, 2]
    
    print(f"\n\tLevetiracetam-Brivaracetam similarity: {lev_briv_sim:.4f}")
    print(f"\tLevetiracetam-Valproic acid similarity: {lev_valp_sim:.4f}")
    
    if lev_briv_sim > lev_valp_sim:
        print("\tStructural analogues are more similar (as expected)")
    
    print(f"\tComplete\n")


def generate_all_asm_embeddings():
    """Generate embeddings for all ASMs in database."""
    import os
    print(f"GENERATING EMBEDDINGS FOR ALL ASMs")
    print(f"  Number of drugs: {len(ASM_SMILES)}")

    drug_names = list(ASM_SMILES.keys())
    smiles_list = list(ASM_SMILES.values())

    # Generate embeddings
    embeddings = get_smiles_embeddings(smiles_list).numpy()

    print(f"\tEmbeddings shape: {embeddings.shape}")
    print(f"\tMean norm: {np.linalg.norm(embeddings, axis=1).mean():.4f}")

    # Ensure outputs directory exists
    os.makedirs('outputs', exist_ok=True)

    # Save embeddings to outputs folder
    np.save('outputs/chemberta_asm_embeddings.npy', embeddings)
    print(f"\tSaved to: outputs/chemberta_asm_embeddings.npy")

    # Save drug name mapping
    with open('outputs/asm_drug_names.txt', 'w') as f:
        for name in drug_names:
            f.write(f"{name}\n")
    print(f"\tDrug names saved to: outputs/asm_drug_names.txt")

    print(f"\tComplete\n")

    return drug_names, embeddings


def find_most_similar_drugs(target_drug, n=5):
    """Find the n most similar drugs to a target drug."""
    print(f"FINDING MOST SIMILAR DRUGS TO: {target_drug}")
    
    if target_drug not in ASM_SMILES:
        print(f"\tError: {target_drug} not in database")
        return
    
    drug_names = list(ASM_SMILES.keys())
    smiles_list = list(ASM_SMILES.values())
    
    # Get target index
    target_idx = drug_names.index(target_drug)
    
    # Generate all embeddings
    embeddings = get_smiles_embeddings(smiles_list).numpy()
    
    # Compute similarities to target
    target_emb = embeddings[target_idx:target_idx+1]
    similarities = cosine_similarity(target_emb, embeddings)[0]
    
    # Sort by similarity (excluding self)
    ranked = sorted(
        [(drug_names[i], similarities[i]) for i in range(len(drug_names)) if i != target_idx],
        key=lambda x: x[1],
        reverse=True
    )
    
    print(f"\tTop {n} most similar drugs:")
    for drug, sim in ranked[:n]:
        print(f"\t  - {drug}: {sim:.4f}")
    
    print(f"\tComplete\n")


# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    # Run tests
    test_base_functionality()
    test_cosine_similarity()
    
    # Generate all embeddings
    drug_names, embeddings = generate_all_asm_embeddings()
    
    # Find similar drugs example
    find_most_similar_drugs('Levetiracetam', n=5)
    find_most_similar_drugs('Carbamazepine', n=5)