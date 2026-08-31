"""Experiment 17: focal-only quad-modal fusion (case-mix transportability probe).

Identical to exp7_all_modalities (Clinical + Text + EEG + SMILES, standard
~2M-parameter QuadFusionMLP) but the cohort is filtered to focal-epilepsy
patients before the cross-validation split. This matches the case mix of the
external HEP1 cohort (which is 100% focal), so a focal-only model is the fair
internal analogue for the external transportability test the reviewers asked
for. The focal subset of the quad cohort is small (~80), so results carry a
wide-CI caveat; the robust n=154 focal signal is at the clinical/base tiers,
handled in the thesis external-validation script.
"""
