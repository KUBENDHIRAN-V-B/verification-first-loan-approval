# V-Loan Data Management & Storage Policy

## 1. Datasets Supported

### A. Real-World Track: UCI Statlog German Credit Dataset
- **Origin**: UCI Machine Learning Repository (Prof. Dr. Hans Hofmann, University of Hamburg).
- **Instances**: 1,000 credit applications.
- **Attributes**: 20 predictor features (7 numerical, 13 categorical) + 1 binary credit risk target.
- **Target Encoding**:
  - `1` = Good Credit Risk (Approved) — 700 instances (70.0%)
  - `0` = Bad Credit Risk (Rejected) — 300 instances (30.0%)
- **Raw Location**: `data/raw/german.data`
- **Processed Location**: `data/processed/german_credit.csv`

### B. Benchmark Track: Controlled Synthetic Credit Dataset
- **Instances**: 1,000 synthetic applicants.
- **Attributes**: 10 features (income, credit score, DTI, loan amount, employment years, existing debts, age, housing status, loan purpose, gender).
- **Target**: Deterministic ground-truth latent utility with known feature weights for exact mathematical benchmarking.
- **Processed Location**: `data/processed/synthetic_credit.csv`

## 2. Integrity and Ethics Guidelines
- No personally identifiable information (PII) is stored or processed.
- The original raw files in `data/raw/` are preserved without in-place modification.
- All train/test splits enforce strict stratification and seed control.
