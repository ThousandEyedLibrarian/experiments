"""
SMILES Embeddings with SMILES Transformer
==========================================
This script generates SMILES embeddings for anti-seizure medications
using the SMILES Transformer (autoencoding transformer architecture).

The SMILES Transformer uses a transformer encoder-decoder trained on
ChEMBL24 (~1.7M molecules) for self-supervised molecular representation.
It produces 256-dimensional embeddings.

Repository: https://github.com/DSPsleeporg/smiles-transformer
Paper: "SMILES Transformer: Pre-trained Molecular Fingerprint for Low Data 
        Drug Discovery" (arXiv:1911.04738)

SMILES strings are retrieved from DrugBank (https://go.drugbank.com/).

Installation:
    pip install torch
    # Clone the repository for model files
    git clone https://github.com/DSPsleeporg/smiles-transformer.git
    
Pretrained Model:
    Download trfm_12_23000.pkl from the Google Drive link in the repo
    Place in: ./smiles_transformer/

IMPORTANT NOTE:
    The original repo is missing vocab.pkl. You may need to:
    1. Rebuild vocab from ChEMBL24 dataset, OR
    2. Use the built-in tokenizer in this script
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Optional, Union
import os
import re
import torch
import torch.nn as nn

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================
# Path to pretrained SMILES Transformer model
MODEL_PATH = os.environ.get('SMILES_TRF_MODEL', './smiles_transformer/trfm_12_23000.pkl')
VOCAB_PATH = os.environ.get('SMILES_TRF_VOCAB', './smiles_transformer/vocab.pkl')

# Embedding dimension for SMILES Transformer
EMBEDDING_DIM = 256

# Maximum SMILES length (from paper: trained on SMILES < 100 chars)
MAX_LENGTH = 100

# =============================================================================
# SMILES TOKENIZER
# =============================================================================
# Regex pattern for SMILES tokenisation (from the original paper)
SMILES_TOKENIZER_PATTERN = r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"


class SMILESTokenizer:
    """
    SMILES tokenizer using regex-based tokenisation.
    
    This tokenizer splits SMILES strings into chemically meaningful tokens
    (atoms, bonds, rings, branches, etc.).
    """
    
    def __init__(self, vocab: dict = None):
        """
        Args:
            vocab: Dictionary mapping tokens to indices. If None, builds
                   vocabulary from predefined tokens.
        """
        self.pattern = re.compile(SMILES_TOKENIZER_PATTERN)
        
        if vocab is None:
            self.vocab = self._build_default_vocab()
        else:
            self.vocab = vocab
        
        self.pad_token = '<pad>'
        self.unk_token = '<unk>'
        self.sos_token = '<sos>'
        self.eos_token = '<eos>'
        
        # Ensure special tokens in vocab
        for token in [self.pad_token, self.unk_token, self.sos_token, self.eos_token]:
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)
        
        self.pad_idx = self.vocab[self.pad_token]
        self.unk_idx = self.vocab[self.unk_token]
        self.sos_idx = self.vocab[self.sos_token]
        self.eos_idx = self.vocab[self.eos_token]
        
        self.idx_to_token = {v: k for k, v in self.vocab.items()}
    
    def _build_default_vocab(self) -> dict:
        """Build default vocabulary from common SMILES tokens."""
        # Common SMILES tokens
        tokens = [
            '<pad>', '<unk>', '<sos>', '<eos>',
            # Atoms
            'C', 'c', 'N', 'n', 'O', 'o', 'S', 's', 'P', 'p',
            'F', 'Cl', 'Br', 'I', 'B', 'b',
            # Bonds
            '-', '=', '#', ':', '~',
            # Structure
            '(', ')', '[', ']', '.', '/',  '\\',
            # Charges and other
            '+', '@', '@@', 'H',
            # Ring numbers
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
            # Bracketed atoms (common ones)
            '[H]', '[C]', '[N]', '[O]', '[S]', '[P]', '[F]', '[Cl]', '[Br]', '[I]',
            '[C@H]', '[C@@H]', '[C@]', '[C@@]', '[N+]', '[N-]', '[O-]', '[S+]',
            '[nH]', '[NH]', '[NH2]', '[NH3+]', '[O-]', '[OH]',
        ]
        
        vocab = {token: idx for idx, token in enumerate(tokens)}
        return vocab
    
    def tokenize(self, smiles: str) -> List[str]:
        """Tokenize a SMILES string."""
        tokens = self.pattern.findall(smiles)
        return tokens
    
    def encode(self, smiles: str, max_length: int = MAX_LENGTH) -> List[int]:
        """Convert SMILES to token indices."""
        tokens = self.tokenize(smiles)
        
        # Add SOS and EOS
        indices = [self.sos_idx]
        for token in tokens[:max_length - 2]:
            indices.append(self.vocab.get(token, self.unk_idx))
        indices.append(self.eos_idx)
        
        # Pad to max_length
        while len(indices) < max_length:
            indices.append(self.pad_idx)
        
        return indices[:max_length]
    
    def decode(self, indices: List[int]) -> str:
        """Convert token indices back to SMILES."""
        tokens = []
        for idx in indices:
            if idx == self.sos_idx or idx == self.pad_idx:
                continue
            if idx == self.eos_idx:
                break
            tokens.append(self.idx_to_token.get(idx, '?'))
        return ''.join(tokens)
    
    def batch_encode(self, smiles_list: List[str], max_length: int = MAX_LENGTH) -> torch.Tensor:
        """Encode a batch of SMILES strings."""
        batch = [self.encode(s, max_length) for s in smiles_list]
        return torch.tensor(batch, dtype=torch.long)


# =============================================================================
# SMILES TRANSFORMER MODEL (Compatible with pretrained weights)
# =============================================================================
class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding with dropout (matches pretrained checkpoint).
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create sinusoidal positional encoding (non-learnable)
        # Shape: (max_len, d_model) then unsqueezed to (1, max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input embeddings and apply dropout."""
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class SMILESTransformerEncoder(nn.Module):
    """
    SMILES Transformer encoder compatible with pretrained weights.

    Architecture matches the pretrained checkpoint (TrfmSeq2seq):
    - embed: token embedding layer
    - pe: sinusoidal positional encoding with dropout
    - trfm.encoder: 4-layer transformer encoder (nhead=4)
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        nhead: int = 4,  # Pretrained uses 4 heads
        num_layers: int = 4,  # Pretrained uses 4 layers
        dim_feedforward: int = 256,  # Pretrained uses 256
        dropout: float = 0.1,
        max_length: int = 5000,
    ):
        super().__init__()

        self.d_model = d_model

        # Token embedding (named 'embed' to match checkpoint)
        self.embed = nn.Embedding(vocab_size, d_model)

        # Sinusoidal positional encoding with dropout (named 'pe' to match checkpoint)
        self.pe = PositionalEncoding(d_model, dropout, max_length)

        # Full transformer (encoder + decoder) to match checkpoint state dict
        # We only use the encoder for embeddings, but need decoder for state dict loading
        self.trfm = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len) token indices
            padding_mask: (batch, seq_len) True for padding positions

        Returns:
            (batch, seq_len, d_model) encoder outputs
        """
        x = self.embed(x)  # (B, T, H)
        x = self.pe(x)  # (B, T, H) - includes dropout

        # Use the encoder part of the transformer
        x = self.trfm.encoder(x, src_key_padding_mask=padding_mask)

        return x

    def get_embedding(self, x: torch.Tensor, padding_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Get fixed-size embedding by mean pooling over non-padding tokens.

        Args:
            x: (batch, seq_len) token indices
            padding_mask: (batch, seq_len) True for padding positions

        Returns:
            (batch, d_model) embeddings
        """
        outputs = self.forward(x, padding_mask)

        if padding_mask is not None:
            # Mask out padding for mean pooling
            mask = (~padding_mask).unsqueeze(-1).float()
            sum_embeddings = (outputs * mask).sum(dim=1)
            lengths = mask.sum(dim=1).clamp(min=1)
            embeddings = sum_embeddings / lengths
        else:
            embeddings = outputs.mean(dim=1)

        return embeddings


# =============================================================================
# PRETRAINED VOCABULARY (45 tokens from ChEMBL24 with min_freq=500)
# =============================================================================
# This vocabulary matches the pretrained model's embedding layer (45 tokens)
# Format matches the original smiles-transformer Vocab class (stoi/itos)
PRETRAINED_VOCAB_ITOS = [
    '<pad>', '<unk>', '<eos>', '<sos>', '<mask>',  # Special tokens (0-4)
    # Most common SMILES tokens from ChEMBL24 (indices 5-44)
    'C', 'c', 'N', 'n', 'O', 'o', 'S', 's', 'F', 'Cl',
    'Br', '(', ')', '[', ']', '=', '#', '@', '@@', 'H',
    '+', '-', '1', '2', '3', '4', '5', '6', '7', '8',
    '9', '0', '/', '\\', '.', 'I', 'P', 'B', 'Si', 'Se',
]
PRETRAINED_VOCAB_STOI = {tok: i for i, tok in enumerate(PRETRAINED_VOCAB_ITOS)}
PRETRAINED_VOCAB_SIZE = len(PRETRAINED_VOCAB_ITOS)  # 45


class PretrainedSMILESTokenizer:
    """
    Tokenizer compatible with the pretrained SMILES Transformer vocabulary.

    Uses the same tokenization logic as the original smiles-transformer repo.
    """

    def __init__(self):
        self.stoi = PRETRAINED_VOCAB_STOI
        self.itos = PRETRAINED_VOCAB_ITOS
        self.vocab = self.stoi  # Alias for compatibility

        self.pad_index = 0
        self.unk_index = 1
        self.eos_index = 2
        self.sos_index = 3
        self.mask_index = 4

        # Aliases for compatibility with existing code
        self.pad_idx = self.pad_index
        self.unk_idx = self.unk_index
        self.eos_idx = self.eos_index
        self.sos_idx = self.sos_index

    def split(self, smiles: str) -> List[str]:
        """
        Split SMILES into tokens (from original smiles-transformer utils.py).
        Handles multi-character tokens like Cl, Br, Si, Se, @@, etc.
        """
        arr = []
        i = 0
        sm = smiles
        while i < len(sm) - 1:
            if not sm[i] in ['%', 'C', 'B', 'S', 'N', 's', '@', '\\']:
                arr.append(sm[i])
                i += 1
            elif sm[i] == '%':
                arr.append(sm[i:i+3])
                i += 3
            elif sm[i] == 'C' and sm[i+1] == 'l':
                arr.append('Cl')
                i += 2
            elif sm[i] == 'B' and sm[i+1] == 'r':
                arr.append('Br')
                i += 2
            elif sm[i] == 'S' and sm[i+1] == 'i':
                arr.append('Si')
                i += 2
            elif sm[i] == 'S' and sm[i+1] == 'e':
                arr.append('Se')
                i += 2
            elif sm[i] == 's' and sm[i+1] == 'e':
                arr.append('se')
                i += 2
            elif sm[i] == '@' and sm[i+1] == '@':
                arr.append('@@')
                i += 2
            elif sm[i] == '\\' and i + 1 < len(sm):
                arr.append('\\')
                i += 1
            else:
                arr.append(sm[i])
                i += 1
        if i == len(sm) - 1:
            arr.append(sm[i])
        return arr

    def decode(self, indices: List[int]) -> str:
        """Convert token indices back to SMILES string."""
        tokens = []
        for idx in indices:
            if idx == self.sos_index or idx == self.pad_index:
                continue
            if idx == self.eos_index:
                break
            if idx < len(self.itos):
                tokens.append(self.itos[idx])
            else:
                tokens.append('?')
        return ''.join(tokens)

    def tokenize(self, smiles: str) -> List[str]:
        """Tokenize a SMILES string."""
        return self.split(smiles)

    def encode(self, smiles: str, max_length: int = MAX_LENGTH) -> List[int]:
        """Convert SMILES to token indices with SOS/EOS."""
        tokens = self.tokenize(smiles)

        # Add SOS and EOS
        indices = [self.sos_index]
        for token in tokens[:max_length - 2]:
            indices.append(self.stoi.get(token, self.unk_index))
        indices.append(self.eos_index)

        # Pad to max_length
        while len(indices) < max_length:
            indices.append(self.pad_index)

        return indices[:max_length]

    def batch_encode(self, smiles_list: List[str], max_length: int = MAX_LENGTH) -> torch.Tensor:
        """Encode a batch of SMILES strings."""
        batch = [self.encode(s, max_length) for s in smiles_list]
        return torch.tensor(batch, dtype=torch.long)


# =============================================================================
# GLOBAL MODEL INSTANCE
# =============================================================================
tokenizer = None
model = None
TRANSFORMER_AVAILABLE = False


def load_model(model_path: str = MODEL_PATH, vocab_path: str = VOCAB_PATH):
    """
    Load the pretrained SMILES Transformer model.

    Args:
        model_path: Path to the pretrained model (.pkl file)
        vocab_path: Path to the vocabulary file (.pkl file)
    """
    global tokenizer, model, TRANSFORMER_AVAILABLE

    print(f"LOADING SMILES TRANSFORMER MODEL")
    print(f"  Model path: {model_path}")
    print(f"  Vocab path: {vocab_path}")

    # Load vocabulary
    if os.path.exists(vocab_path):
        import pickle
        with open(vocab_path, 'rb') as f:
            vocab_obj = pickle.load(f)
        # Handle both dict and Vocab object formats
        if hasattr(vocab_obj, 'stoi'):
            tokenizer = PretrainedSMILESTokenizer()
            tokenizer.stoi = vocab_obj.stoi
            tokenizer.itos = vocab_obj.itos
            tokenizer.vocab = vocab_obj.stoi
        else:
            tokenizer = SMILESTokenizer(vocab_obj)
        print(f"\tVocabulary loaded: {len(tokenizer.vocab)} tokens")
    else:
        print(f"\tVocab file not found, using pretrained vocabulary ({PRETRAINED_VOCAB_SIZE} tokens)")
        tokenizer = PretrainedSMILESTokenizer()
        print(f"\tPretrained vocabulary: {len(tokenizer.vocab)} tokens")

    # Load model
    if os.path.exists(model_path):
        try:
            state_dict = torch.load(model_path, map_location='cpu', weights_only=False)

            # Check if it's a state dict or a full model
            if isinstance(state_dict, dict) and 'embed.weight' in state_dict:
                # It's a state dict - need to create model and load weights
                vocab_size = state_dict['embed.weight'].shape[0]
                print(f"\tState dict loaded, vocab_size={vocab_size}")

                # Create model with matching architecture
                model = SMILESTransformerEncoder(
                    vocab_size=vocab_size,
                    d_model=256,
                    nhead=4,  # Pretrained uses 4 heads
                    num_layers=4,
                    dim_feedforward=256,
                )

                # Load state dict (strict=False to handle any minor differences)
                model.load_state_dict(state_dict, strict=False)
                model.eval()
                TRANSFORMER_AVAILABLE = True
                print(f"\tModel weights loaded successfully")
            else:
                # It's a full model object
                model = state_dict
                model.eval()
                TRANSFORMER_AVAILABLE = True
                print(f"\tModel object loaded successfully")

        except Exception as e:
            print(f"\tError loading model: {e}")
            import traceback
            traceback.print_exc()
            print(f"\tUsing randomly initialised model")
            model = SMILESTransformerEncoder(vocab_size=len(tokenizer.vocab))
            model.eval()
            TRANSFORMER_AVAILABLE = False
    else:
        print(f"\tModel file not found, using randomly initialised model")
        model = SMILESTransformerEncoder(vocab_size=len(tokenizer.vocab))
        model.eval()
        TRANSFORMER_AVAILABLE = False

    print(f"\tComplete\n")


# Initialise with pretrained model if available
load_model()

print(f"SMILES TRANSFORMER INITIALISED")
print(f"  Vocab size: {len(tokenizer.vocab)}")
print(f"  Embedding dimension: {EMBEDDING_DIM}")
print(f"  Pretrained: {TRANSFORMER_AVAILABLE}")
print(f"\tComplete\n")


def get_smiles_embeddings(
    smiles: Union[str, List[str]], 
    pooling: str = 'mean'
) -> np.ndarray:
    """
    Get embeddings from SMILES Transformer for SMILES strings.
    
    Args:
        smiles: str or list of SMILES strings
        pooling: 'mean' (mean pooling) or 'cls' (first token)
    
    Returns:
        np.ndarray of shape [num_smiles, 256]
    """
    global tokenizer, model
    
    # Ensure smiles is a list
    if isinstance(smiles, str):
        smiles = [smiles]
    
    # Filter out SMILES that are too long
    valid_smiles = []
    for s in smiles:
        if len(s) <= MAX_LENGTH:
            valid_smiles.append(s)
        else:
            print(f"\tWarning: SMILES too long ({len(s)} chars), truncating: {s[:30]}...")
            valid_smiles.append(s[:MAX_LENGTH])
    
    # Tokenize and encode
    token_ids = tokenizer.batch_encode(valid_smiles)
    
    # Create padding mask
    padding_mask = (token_ids == tokenizer.pad_idx)
    
    # Get embeddings
    with torch.no_grad():
        if pooling == 'cls':
            # Use first token (SOS) embedding
            outputs = model(token_ids, padding_mask)
            embeddings = outputs[:, 0, :]
        else:
            # Mean pooling
            embeddings = model.get_embedding(token_ids, padding_mask)
    
    return embeddings.numpy()


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
}


def test_tokenizer():
    """Test the SMILES tokenizer."""
    print(f"TESTING SMILES TOKENIZER")
    
    test_smiles = ASM_SMILES['Levetiracetam']
    print(f"  Input SMILES: {test_smiles}")
    
    tokens = tokenizer.tokenize(test_smiles)
    print(f"\tTokens ({len(tokens)}): {tokens}")
    
    indices = tokenizer.encode(test_smiles)
    print(f"\tIndices ({len(indices)}): {indices[:20]}...")
    
    decoded = tokenizer.decode(indices)
    print(f"\tDecoded: {decoded}")
    
    match = "YES" if test_smiles == decoded else "NO"
    print(f"\tRoundtrip: {match}")
    
    print(f"\tComplete\n")


def test_base_functionality():
    """Test basic SMILES embedding functionality."""
    smiles = ASM_SMILES['Levetiracetam']  
    print(f"TESTING BASELINE SMILES EMBEDDING FUNCTION (Transformer)")
    print(f"  Input SMILES: {smiles}")
    embedding = get_smiles_embeddings(smiles)
    
    expected_shape = (1, EMBEDDING_DIM)
    assert embedding.shape == expected_shape, \
        f"Assertion invalid, expected shape {expected_shape} but got {embedding.shape}"
    print(f"\tEmbedding shape correct: {embedding.shape}")
    print(f"\tEmbedding norm: {np.linalg.norm(embedding[0]):.4f}")
    print(f"\tComplete\n")


def test_cosine_similarity():
    """Test similarity between related drugs."""
    print(f"TESTING COSINE SIMILARITY FOR RELATED ASMs (Transformer)")
    
    # Compare structurally related drugs
    related_drugs = [
        ('Levetiracetam', ASM_SMILES['Levetiracetam']),
        ('Brivaracetam', ASM_SMILES['Brivaracetam']),    # Structural analogue
        ('Valproic_acid', ASM_SMILES['Valproic_acid']),  # Different class
    ]
    
    drug_names = [d[0] for d in related_drugs]
    smiles_list = [d[1] for d in related_drugs]
    
    embeddings = get_smiles_embeddings(smiles_list)
    
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
    print(f"GENERATING EMBEDDINGS FOR ALL ASMs (Transformer)")
    print(f"  Number of drugs: {len(ASM_SMILES)}")
    print(f"  Embedding dimension: {EMBEDDING_DIM}")
    
    drug_names = list(ASM_SMILES.keys())
    smiles_list = list(ASM_SMILES.values())
    
    # Generate embeddings
    embeddings = get_smiles_embeddings(smiles_list)
    
    print(f"\tEmbeddings shape: {embeddings.shape}")
    print(f"\tMean norm: {np.linalg.norm(embeddings, axis=1).mean():.4f}")
    
    # Save embeddings
    np.save('asm_smiles_embeddings_transformer.npy', embeddings)
    print(f"\tSaved to: asm_smiles_embeddings_transformer.npy")
    
    # Save drug name mapping
    with open('asm_drug_names_transformer.txt', 'w') as f:
        for name in drug_names:
            f.write(f"{name}\n")
    print(f"\tDrug names saved to: asm_drug_names_transformer.txt")
    
    print(f"\tComplete\n")
    
    return drug_names, embeddings


def find_most_similar_drugs(target_drug, n=5):
    """Find the n most similar drugs to a target drug."""
    print(f"FINDING MOST SIMILAR DRUGS TO: {target_drug} (Transformer)")
    
    if target_drug not in ASM_SMILES:
        print(f"\tError: {target_drug} not in database")
        return
    
    drug_names = list(ASM_SMILES.keys())
    smiles_list = list(ASM_SMILES.values())
    
    # Get target index
    target_idx = drug_names.index(target_drug)
    
    # Generate all embeddings
    embeddings = get_smiles_embeddings(smiles_list)
    
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


def compare_with_chemberta(chemberta_embeddings_path: str = None):
    """
    Compare Transformer embeddings with ChemBERTa embeddings.
    """
    print(f"COMPARING TRANSFORMER WITH CHEMBERTA EMBEDDINGS")
    
    if chemberta_embeddings_path is None:
        chemberta_embeddings_path = 'asm_smiles_embeddings.npy'
    
    if not os.path.exists(chemberta_embeddings_path):
        print(f"\tChemBERTa embeddings not found at: {chemberta_embeddings_path}")
        print(f"\tRun e1_LLM_SMILES_ChemBERTa.py first.")
        return
    
    # Load ChemBERTa embeddings
    chemberta_emb = np.load(chemberta_embeddings_path)
    
    # Generate Transformer embeddings
    smiles_list = list(ASM_SMILES.values())
    transformer_emb = get_smiles_embeddings(smiles_list)
    
    # Compute similarity matrices
    chemberta_sim = cosine_similarity(chemberta_emb)
    transformer_sim = cosine_similarity(transformer_emb)
    
    # Flatten upper triangular (excluding diagonal)
    n = len(smiles_list)
    triu_idx = np.triu_indices(n, k=1)
    
    chemberta_flat = chemberta_sim[triu_idx]
    transformer_flat = transformer_sim[triu_idx]
    
    # Compute correlation
    correlation = np.corrcoef(chemberta_flat, transformer_flat)[0, 1]
    
    print(f"\tChemBERTa embedding dim: {chemberta_emb.shape[1]}")
    print(f"\tTransformer embedding dim: {transformer_emb.shape[1]}")
    print(f"\tPairwise similarity correlation: {correlation:.4f}")
    print(f"\tComplete\n")


# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    print("="*60)
    print("SMILES TRANSFORMER EMBEDDING SCRIPT")
    print("="*60)
    print(f"Pretrained Model Available: {TRANSFORMER_AVAILABLE}")
    print(f"Embedding Dimension: {EMBEDDING_DIM}")
    print(f"Vocabulary Size: {len(tokenizer.vocab)}")
    print("="*60 + "\n")
    
    # Test tokenizer
    test_tokenizer()
    
    # Run tests
    test_base_functionality()
    test_cosine_similarity()
    
    # Generate all embeddings
    drug_names, embeddings = generate_all_asm_embeddings()
    
    # Find similar drugs example
    find_most_similar_drugs('Levetiracetam', n=5)
    find_most_similar_drugs('Carbamazepine', n=5)
    
    # Compare with ChemBERTa if available
    compare_with_chemberta()