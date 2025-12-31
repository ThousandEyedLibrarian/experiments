from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import torch

# Load ClinicalBERT's tokenizer and model
print(f"LOADING BASELINE MODEL FOR TEST EVAL")
tokenizer = AutoTokenizer.from_pretrained("medicalai/ClinicalBERT")
model = AutoModel.from_pretrained("medicalai/ClinicalBERT")

# Set model to evaluation mode (disables dropout, etc.)
model.eval()
print(f"\tComplete\n")

def get_medical_embeddings(texts, pooling='mean'):
    """
    Get embeddings from ClinicalBERT
    
    Args:
        texts: str or list of str
        pooling: 'mean' or 'cls'
    
    Returns:
        torch.Tensor of shape [num_texts, 768]
    """
    # Ensure texts is a list
    if isinstance(texts, str):
        texts = [texts]
    
    # Tokenize
    inputs = tokenizer(
        texts,
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
        # Mean pooling
        attention_mask = inputs['attention_mask']
        token_embeddings = outputs.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        embeddings = sum_embeddings / sum_mask
    
    return embeddings

def test_base_functionality():
    text = "Patient presents with chest pain and dyspnea."
    print(f"TESTING BASELINE EMBEDDING FUNCTION")
    embedding = get_clinical_embeddings(text)
    assert embedding.shape == torch.Size([1, 768]), f"Assertion invalid, expected embedding of shape [1, 768] but got shape {embedding.shape}."
    print(f"\tEmbedding shape correct: {embedding.shape}")  # [1, 768]
    print(f"\tComplete\n")

def test_cosine_similarity():
    print(f"TESTING COSINE SIMILARITY FUNCTION")
    texts = [
        "Patient has hypertension",
        "Blood pressure is elevated",
        "Patient complains of headache"
    ]

    embeddings = get_clinical_embeddings(texts).numpy()

    # Compute similarity matrix
    similarities = cosine_similarity(embeddings)
    print("\tSimilarity matrix:")
    print(similarities)
    print(f"\tComplete\n")

test_base_functionality()
test_cosine_similarity()