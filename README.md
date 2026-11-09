# Fraud Detection Classification

Detecting fraudulent card transactions on a heavily imbalanced dataset of about 300,000 transactions. The notebook builds time- and behaviour-based features, compares four classifiers with stratified cross-validation, and evaluates the chosen gradient boosting model on a held-out test set.

## Dataset

`transactions.csv`: 299,695 transactions, 17 columns, with the target `is_fraud`.

| Group | Columns |
|---|---|
| Identifiers | `transaction_id`, `user_id` |
| Account | `account_age_days`, `total_transactions_user`, `avg_amount_user` |
| Transaction | `amount`, `country`, `bin_country`, `channel`, `merchant_category`, `promo_used`, `transaction_time`, `shipping_distance_km` |
| Verification | `avs_match`, `cvv_result`, `three_ds_flag` |
| Target | `is_fraud` |


## Approach

`notebooks/fraud_detection.ipynb`:

1. **Explore**: structure, summary statistics and class balance of `is_fraud`.
2. **Clean**: cast the target to boolean and parse `transaction_time` as a datetime.
3. **Engineer features**:
   - hour, day of week and month of the transaction
   - `amount_vs_avg`: the transaction amount relative to the user's average amount
   - `days_since_first`: days since the user's first transaction in the data
4. **Preprocess**: one-hot encode the categorical features and standardise the numeric ones in a scikit-learn `Pipeline` / `ColumnTransformer`. Stratified 80/20 train/test split.
5. **Compare models** with 5-fold stratified cross-validation scored on F1: SGD classifier, random forest, XGBoost and gradient boosting. Class imbalance is handled with class weights in the models that support them (SGD, random forest, XGBoost).
6. **Evaluate** the final `GradientBoostingClassifier` (100 trees, depth 6, learning rate 0.05, subsample 0.8) on the test set.

### Result

As saved in the notebook, the final model reaches **precision 0.919** and **recall 0.723** on the held-out test set (the positive class is fraud).

## Repository layout

```
.
├── data/
│   └── raw/transactions.csv
├── notebooks/
│   └── fraud_detection.ipynb
└── requirements.txt
```

## Getting started

Requires Python 3.9+.

```bash
git clone https://github.com/Oualidlaib/fraud-detection-classification.git
cd fraud-detection-classification

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt


jupyter notebook notebooks/fraud_detection.ipynb
```
