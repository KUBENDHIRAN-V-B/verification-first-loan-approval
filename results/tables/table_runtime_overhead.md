# Table: Table Runtime Overhead

| component_id                     | component_name                 | unit              |   mean_ms |     std_ms |   min_ms |    max_ms | overhead_vs_unverified_pct   |
|:---------------------------------|:-------------------------------|:------------------|----------:|-----------:|---------:|----------:|:-----------------------------|
| 1_Model_Inference_Single         | Model Inference Single         | milliseconds (ms) |  15.5363  |   3.30891  |  12.028  |   39.3155 | N/A                          |
| 2_SHAP_Attribution_Top5          | SHAP Attribution Top5          | milliseconds (ms) |   4.36223 |   0.888266 |   3.5569 |    8.3429 | N/A                          |
| 3_LIME_Attribution_Top5          | LIME Attribution Top5          | milliseconds (ms) | 912.866   |  43.5996   | 851.754  | 1075.31   | N/A                          |
| 4_VLoan_Explanation_Verification | VLoan Explanation Verification | milliseconds (ms) | 234.24    |  18.89     | 212.199  |  344.083  | N/A                          |
| 5_Constraint_Aware_Recourse      | Constraint Aware Recourse      | milliseconds (ms) | 145.189   | 486.555    |  12.321  | 2240.28   | N/A                          |
| 6_Full_VLoan_Per_Applicant       | Full VLoan Per Applicant       | milliseconds (ms) | 383.791   | 487.117    | 231.965  | 2470.53   | 1928.7%                      |
