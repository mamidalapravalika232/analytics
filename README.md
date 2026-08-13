# Titanic Analytics Pipeline

## Overview

This module implements the complete Titanic analytics pipeline:

1. Dataset profiling
2. Missing-value handling
3. Univariate analysis
4. Bivariate analysis
5. Multivariate data story
6. Standardization sanity check
7. Stratified train/test split
8. Leakage-safe preprocessing
9. Classification
10. Classification evaluation
11. Class imbalance comparison
12. Random Forest hyperparameter tuning
13. Fare regression
14. Model comparison
15. Complete pipeline persistence

---

# Dataset Loading Strategy

The Titanic dataset is loaded using:

```python
sns.load_dataset("titanic")
