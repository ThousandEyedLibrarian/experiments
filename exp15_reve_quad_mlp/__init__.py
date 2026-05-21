"""Experiment 15: Quad-modal fusion with REVE-base as the EEG encoder.

Mirrors exp7_all_modalities (Clinical + Text + EEG + SMILES) but uses
pre-computed REVE-base per-window features (512-dim) instead of the
EEG2Vec encoder. REVE is frozen; the projection layer, aggregator and
fusion classifier are trained per fold via Stage A discipline.
"""
