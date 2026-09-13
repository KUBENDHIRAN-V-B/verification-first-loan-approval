# Table: Table12 Control Experiments

| Experiment Type                                       |   Total Claims | Materiality Pass   | Directional Pass   | DEVR   | Verification Decision   |
|:------------------------------------------------------|---------------:|:-------------------|:-------------------|:-------|:------------------------|
| Positive Control (True Model Weights, tau=0.01)       |            150 | 52.0%              | 100.0%             | 52.0%  | ACCEPTED (Verified)     |
| Negative Control (Sign-Flipped Attribution, tau=0.01) |            150 | 52.0%              | 0.0%               | 0.0%   | REJECTED (Filtered)     |
| Positive Control (True Model Weights, tau=0.03)       |            150 | 0.0%               | 100.0%             | 0.0%   | PARTIAL / FILTERED      |
| Negative Control (Sign-Flipped Attribution, tau=0.03) |            150 | 0.0%               | 0.0%               | 0.0%   | REJECTED (Filtered)     |
