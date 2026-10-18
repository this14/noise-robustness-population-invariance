# Noise robustness across populations: code

Code and configuration for:
"Is Noise Robustness Population-Invariant? A Simulation Study of Wearable
Electrocardiogram Deep Learning"  -- Chuelwon Lee, HUINNO Co., Ltd.

Full documentation (file-by-file description, mapping to manuscript tables and
figures, reproduction order, prespecified parameters, verification checks, and
dependency notes) is in the Supplementary Material 1 (Supporting Information) of the manuscript.

CONTENTS
  src/                 this study's scripts (training, noise, evaluation,
                       pre-flight verification, statistics, sex-stratified and
                       post hoc analyses)
  configs/grid51.yaml  experiment grid; all paths are placeholders (CHANGE ME)
  figures/             scripts that regenerate Figures 1-2 and the Central
                       Illustration from the analysis CSVs
  companion/           unmodified code from the two companion studies that
                       this study imports (see Supporting Information, §2);
                       MD5 checksums listed in MD5SUMS.txt
  MD5SUMS.txt          checksums of every file in this package

NOT INCLUDED: raw ECG data (public, PhysioNet), trained checkpoints, per-record
predictions, and the PTB-XL / Chapman-Shaoxing-Ningbo loaders and model
definitions of the companion sampling-rate study (see Supporting Information).
