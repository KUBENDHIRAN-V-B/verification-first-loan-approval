# Table: Table Parameter Sensitivity

| sweep_parameter          |   parameter_value |   top_k |   perturbation_delta |   rank_threshold_rho |   materiality_pass_rate |   direction_pass_rate |   rank_pass_rate |     DEVR |   abstention_rate |
|:-------------------------|------------------:|--------:|---------------------:|---------------------:|------------------------:|----------------------:|-----------------:|---------:|------------------:|
| Materiality_Tau          |              0.01 |       5 |                 0.1  |                  0.3 |                   0.82  |              0.504    |             0.52 | 0.268    |              0.48 |
| Materiality_Tau          |              0.02 |       5 |                 0.1  |                  0.3 |                   0.764 |              0.504    |             0.52 | 0.252    |              0.48 |
| Materiality_Tau          |              0.03 |       5 |                 0.1  |                  0.3 |                   0.72  |              0.504    |             0.52 | 0.236    |              0.52 |
| Materiality_Tau          |              0.05 |       5 |                 0.1  |                  0.3 |                   0.64  |              0.504    |             0.52 | 0.216    |              0.52 |
| Materiality_Tau          |              0.1  |       5 |                 0.1  |                  0.3 |                   0.484 |              0.504    |             0.52 | 0.16     |              0.54 |
| Perturbation_Scale_Delta |              0.05 |       5 |                 0.05 |                  0.3 |                   0.72  |              0.504    |             0.52 | 0.236    |              0.52 |
| Perturbation_Scale_Delta |              0.1  |       5 |                 0.1  |                  0.3 |                   0.72  |              0.504    |             0.52 | 0.236    |              0.52 |
| Perturbation_Scale_Delta |              0.2  |       5 |                 0.2  |                  0.3 |                   0.744 |              0.528    |             0.54 | 0.24     |              0.5  |
| Rank_Threshold_Rho       |              0    |       5 |                 0.1  |                  0   |                   0.72  |              0.504    |             0.64 | 0.296    |              0.4  |
| Rank_Threshold_Rho       |              0.3  |       5 |                 0.1  |                  0.3 |                   0.72  |              0.504    |             0.52 | 0.236    |              0.52 |
| Rank_Threshold_Rho       |              0.5  |       5 |                 0.1  |                  0.5 |                   0.72  |              0.504    |             0.26 | 0.124    |              0.76 |
| Rank_Threshold_Rho       |              0.7  |       5 |                 0.1  |                  0.7 |                   0.72  |              0.504    |             0.22 | 0.104    |              0.8  |
| Rank_Threshold_Rho       |              0.9  |       5 |                 0.1  |                  0.9 |                   0.72  |              0.504    |             0.06 | 0.036    |              0.94 |
| Top_K_Features           |              3    |       3 |                 0.1  |                  0.3 |                   0.7   |              0.513333 |             0.76 | 0.346667 |              0.34 |
| Top_K_Features           |              5    |       5 |                 0.1  |                  0.3 |                   0.72  |              0.504    |             0.52 | 0.236    |              0.52 |
| Top_K_Features           |             10    |      10 |                 0.1  |                  0.3 |                   0.68  |              0.454    |             0.26 | 0.08     |              0.76 |
