"""Experiment 16: reduced-capacity quad-modal fusion (transportability probe).

Same four modalities and data pipeline as exp7_all_modalities, but with a
deliberately smaller model, to test the reviewer hypothesis that the headline
~2M-parameter model over-fits the internal cohort and that a lower-capacity
model transports better to the external HEP1 cohort.

Capacity is reduced via exp11's QuadMLPv2, which (unlike exp7's QuadFusionMLP)
exposes the per-modality projection width (hidden_dim), the EEG-encoder
embedding width (eeg_embed_dim), and the aggregator type. Two size points are
run: "small" (hidden_dim 32, eeg_embed_dim 64, MeanMax pooling) and "tiny"
(hidden_dim 16). Everything else (folds, seed, training loop) is identical to
exp7/exp15 for an apples-to-apples comparison.
"""
