# Contributing to Verification-First Loan Approval

We welcome contributions from researchers and practitioners in Explainable AI (XAI), Algorithmic Recourse, and Responsible Machine Learning!

---

## Contribution Workflow

1. **Fork the Repository**: Fork on GitHub and clone your local copy.
2. **Create a Feature Branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Set Up Python Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```
4. **Implementation Guidelines**:
   - Adhere strictly to PEP 8 style standards.
   - Maintain comprehensive docstrings and explicit type annotations.
   - **Data Isolation Guarantee**: Any new transformers or feature scalers must be fitted strictly on training data splits to prevent data leakage.
   - **Zero-Denominator Compliance**: All metrics must respect formal mathematical boundary conditions (e.g. `RVR = "N/A"` when $N_{\text{feas}} = 0$).
5. **Run the Verification Suite**:
   ```bash
   pytest tests/ -v
   python experiments/validate_results.py
   ```
6. **Submit a Pull Request**: Submit your pull request with a detailed description of methodology, test results, and experimental verification.

---

## Code of Conduct
We are committed to providing a friendly, constructive, and inclusive environment for all contributors.
