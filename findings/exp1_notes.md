Results Table
  ┌──────────────────────────┬─────────────┬─────────────┬─────────────┐
  │        Experiment        │  Accuracy   │     AUC     │     F1      │
  ├──────────────────────────┼─────────────┼─────────────┼─────────────┤
  │ Exp 1a (MLP)             │             │             │             │
  ├──────────────────────────┼─────────────┼─────────────┼─────────────┤
  │ clinicalbert + chemberta │ 0.586±0.090 │ 0.594±0.137 │ 0.617±0.122 │
  ├──────────────────────────┼─────────────┼─────────────┼─────────────┤
  │ clinicalbert + smilestrf │ 0.527±0.121 │ 0.695±0.100 │ 0.387±0.327 │
  ├──────────────────────────┼─────────────┼─────────────┼─────────────┤
  │ pubmedbert + chemberta   │ 0.595±0.073 │ 0.662±0.052 │ 0.647±0.053 │
  ├──────────────────────────┼─────────────┼─────────────┼─────────────┤
  │ pubmedbert + smilestrf   │ 0.504±0.033 │ 0.585±0.084 │ 0.425±0.237 │
  ├──────────────────────────┼─────────────┼─────────────┼─────────────┤
  │ Exp 1b (FuseMoE)         │             │             │             │
  ├──────────────────────────┼─────────────┼─────────────┼─────────────┤
  │ clinicalbert + chemberta │ 0.611±0.078 │ 0.648±0.082 │ 0.615±0.132 │
  ├──────────────────────────┼─────────────┼─────────────┼─────────────┤
  │ clinicalbert + smilestrf │ 0.488±0.115 │ 0.621±0.090 │ 0.415±0.264 │
  ├──────────────────────────┼─────────────┼─────────────┼─────────────┤
  │ pubmedbert + chemberta   │ 0.577±0.093 │ 0.607±0.074 │ 0.547±0.278 │
  ├──────────────────────────┼─────────────┼─────────────┼─────────────┤
  │ pubmedbert + smilestrf   │ 0.495±0.077 │ 0.578±0.058 │ 0.288±0.310 │
  └──────────────────────────┴─────────────┴─────────────┴─────────────┘
  Key Findings

  1. Best AUC: exp1a_clinicalbert_smilestrf (MLP) with AUC 0.695
  2. Best Accuracy: exp1b_clinicalbert_chemberta (FuseMoE) with 61.1%
  3. Most stable F1: exp1a_pubmedbert_chemberta (MLP) with F1 0.647±0.053

  Observations

  - The simple MLP baseline (Exp 1a) achieved the best AUC, suggesting the small dataset (121 samples) may not benefit from the more complex FuseMoE architecture
  - High variance across folds indicates the model is sensitive to the train/test split
  - ClinicalBERT text embeddings generally performed better than PubMedBERT
