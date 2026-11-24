# (French below\...)

# Credit Card Fraud Detection

Solution for the Kaggle **Credit Card Fraud Detection** competition (binary classification of fraudulent and legitimate credit card transactions).

## Results

- **Validation ROC AUC (stratified 80/20 split):** approximately `0.9464` for the baseline XGBoost model.
- **Public Leaderboard ROC AUC:** `0.94791` (XGBoost baseline).
- **Feature Engineering variant — Public Leaderboard ROC AUC:** `0.94758` (unweighted security risk count).
- **Weighted Feature Engineering variant — Public Leaderboard ROC AUC:** `0.94783`.

The baseline XGBoost model achieved the best public leaderboard score among the tested approaches.

## Approach

### 1. Data Preparation & Encoding

The dataset contains approximately 700,000 training transactions and 300,000 test transactions, with 25 input features.

- **Target:** `is_fraud` (binary classification).
- **Numerical Features:** transaction amounts, account balances, card age, transaction frequency, merchant risk scores and other behavioral indicators.
- **Categorical Features:** merchant category, card type, authentication method, transaction channel and device type.
- **Categorical Encoding:** One-Hot Encoding implemented through Scikit-learn's `ColumnTransformer` and `OneHotEncoder(handle_unknown="ignore")`.
- **Validation Strategy:** stratified 80/20 train-validation split (`random_state=42`) to preserve the fraud class distribution.

### 2. Model & Hyperparameter Tuning (`xgboost_baseline.py`)

The main model is a single `XGBClassifier` based on gradient-boosted decision trees.

The model was optimized through successive experiments on tree complexity, learning rate and regularization.

**Selected hyperparameters:**

- `max_depth=2`
- `learning_rate=0.02`
- `subsample=0.8`
- `colsample_bytree=0.8`
- `min_child_weight=3`
- `reg_lambda=3`
- `gamma=0.5`

Early stopping (`early_stopping_rounds=700`) was used to determine the optimal number of boosting iterations based on validation ROC AUC.

For the final Kaggle submission, the selected model was retrained on the full training dataset before generating fraud probabilities for the test set.

### 3. Feature Engineering Experiments (`xgboost_feature_engineering.py`)

Several additional features were designed to capture relationships between transaction characteristics:

- **Transaction Amount Ratios:** transaction amount relative to account balance and card age.
- **Security Risk Count:** aggregation of six binary security indicators, including foreign transactions, VPN usage, IP-country mismatch and billing-shipping mismatch.
- **Weighted Security Risk Count:** an alternative version of the security risk count was tested, assigning different weights to the six binary indicators according to their observed association with fraud. This approach achieved a public leaderboard ROC AUC of `0.94783`, slightly outperforming the unweighted version (`0.94758`), but remaining below the baseline model (`0.94791`).
- **Merchant-Foreign Interaction:** interaction between merchant risk score and foreign transaction status.

The engineered features were evaluated alongside the original variables. Despite a slight improvement in local validation ROC AUC, the feature-engineered variants did not outperform the baseline on the public leaderboard.

### 4. Alternative Models & Experiments

Additional experiments were conducted to assess the potential benefits of alternative algorithms and model combinations:

- **CatBoost:** evaluated as an alternative gradient-boosting algorithm with native categorical feature handling.
- **XGBoost:** systematically tuned across different tree depths, learning rates and regularization settings.
- **Ensembling:** prediction blending between CatBoost and XGBoost was tested but did not improve upon the standalone XGBoost model.



# French

Solution pour la compétition Kaggle **Credit Card Fraud Detection** (classification binaire de transactions bancaires frauduleuses et légitimes).

## Résultats

- **ROC AUC en validation (split stratifié 80/20) :** environ `0.9464` pour le modèle XGBoost initial.
- **ROC AUC Public Leaderboard :** `0.94791` (XGBoost initial).
- **Variante avec Feature Engineering — ROC AUC Public Leaderboard :** `0.94758` (comptage non pondéré des signaux de sécurité).
- **Variante avec Feature Engineering pondéré — ROC AUC Public Leaderboard :** `0.94783`.

Le modèle XGBoost initial obtient le meilleur score public parmi les approches testées.

## Démarche

### 1. Préparation des données & Encodage

Le jeu de données contient environ 700 000 transactions d'entraînement et 300 000 transactions de test, avec 25 variables explicatives.

- **Variable cible :** `is_fraud` (classification binaire).
- **Variables numériques :** montants des transactions, soldes des comptes, ancienneté des cartes, fréquence des transactions, scores de risque des commerçants et indicateurs comportementaux.
- **Variables catégorielles :** catégorie du commerçant, type de carte, méthode d'authentification, canal de transaction et type d'appareil.
- **Encodage catégoriel :** One-Hot Encoding avec `ColumnTransformer` et `OneHotEncoder(handle_unknown="ignore")` de Scikit-learn.
- **Stratégie de validation :** séparation stratifiée 80/20 (`random_state=42`) afin de préserver la proportion de transactions frauduleuses.

### 2. Modélisation & Optimisation des hyperparamètres (`xgboost_baseline.py`)

Le modèle principal est un `XGBClassifier` reposant sur le gradient boosting d'arbres de décision.

L'optimisation a porté sur la complexité des arbres, le taux d'apprentissage et la régularisation.

**Hyperparamètres retenus :**

- `max_depth=2`
- `learning_rate=0.02`
- `subsample=0.8`
- `colsample_bytree=0.8`
- `min_child_weight=3`
- `reg_lambda=3`
- `gamma=0.5`

Un mécanisme d'early stopping (`early_stopping_rounds=700`) a permis de déterminer le nombre optimal d'itérations à partir du ROC AUC en validation.

Pour la soumission finale, le modèle sélectionné a été réentraîné sur l'intégralité des données d'entraînement avant de produire les probabilités de fraude sur le jeu de test.

### 3. Expérimentations de Feature Engineering (`xgboost_feature_engineering.py`)

Plusieurs variables supplémentaires ont été construites afin de capturer des relations entre les caractéristiques des transactions :

- **Ratios de montants :** montant de la transaction rapporté au solde du compte et à l'ancienneté de la carte.
- **Comptage des signaux de sécurité :** agrégation de six indicateurs binaires, notamment les transactions étrangères, l'utilisation d'un VPN, les incohérences de pays IP et les divergences entre adresses de facturation et de livraison.
- **Comptage pondéré des signaux de sécurité :** une variante a également été testée en attribuant des pondérations différentes aux six indicateurs binaires, en fonction de leur association observée avec la fraude. Cette approche a obtenu un ROC AUC de `0.94783` sur le leaderboard public, légèrement supérieur à la version non pondérée (`0.94758`), mais restant inférieur au modèle initial (`0.94791`).
- **Interaction risque commerçant / transaction étrangère :** produit du score de risque du commerçant et de l'indicateur de transaction étrangère.

Ces variables ont été évaluées en complément des variables originales. Malgré une légère amélioration du ROC AUC en validation locale, les variantes avec Feature Engineering n'ont pas dépassé le modèle initial sur le leaderboard public.

### 4. Modèles alternatifs & Expérimentations

D'autres expérimentations ont été menées afin d'évaluer l'intérêt d'algorithmes alternatifs et de combinaisons de modèles :

- **CatBoost :** évalué comme alternative de gradient boosting, avec gestion native des variables catégorielles.
- **XGBoost :** optimisation successive de la profondeur des arbres, du taux d'apprentissage et de la régularisation.
- **Ensembling :** combinaison des prédictions de CatBoost et XGBoost, sans amélioration par rapport au modèle XGBoost seul.

## Repository Structure

```text
├── xgboost_baseline.py                 # Baseline XGBoost: training, validation & submission
├── xgboost_feature_engineering.py      # XGBoost with engineered features
├── requirements.txt                    # Python dependencies
├── .gitignore                          # Excluded files
└── README.md                            # Project documentation
```
