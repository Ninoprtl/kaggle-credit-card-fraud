
import pandas as pd
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer


# ==========================================
# 1. CHARGEMENT DES DONNEES
# ==========================================

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# Feature Engineering 

def create_features(df):
    df = df.copy()

    # 1. Montant de la transaction relativement au solde
    df["amount_balance_ratio"] = (
        df["amount_usd"] / (df["account_balance_usd"] + 1)
    )

    # 2. Montant relativement à l'ancienneté de la carte
    df["amount_per_card_month"] = (
        df["amount_usd"] / (df["card_age_months"] + 1)
    )

    # 3. Nombre de signaux de sécurité suspects
    security_columns = [
        "is_foreign_transaction",
        "is_new_merchant",
        "used_vpn",
        "ip_country_mismatch",
        "billing_shipping_mismatch",
        "is_ai_generated_scam_attempt"
    ]

    df["security_risk_count"] = (
        df[security_columns].astype(int).sum(axis=1)
    )

    # 4. Interaction entre risque commerçant et transaction étrangère
    df["merchant_foreign_risk"] = (
        df["merchant_risk_score"]
        * df["is_foreign_transaction"].astype(int)
    )

    return df


X = train.drop(columns=["is_fraud", "id"])
y = train["is_fraud"].astype(int)
X = create_features(X)

X_test = test.drop(columns=["id"])
X_test = create_features(X_test)


# ==========================================
# 2. PREPARATION DES VARIABLES
# ==========================================
cat_cols = X.select_dtypes(include="object").columns.tolist()
num_cols = X.select_dtypes(exclude="object").columns.tolist()

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", "passthrough", num_cols)
    ]
)

X_train_encoded = preprocessor.fit_transform(X_train)
X_valid_encoded = preprocessor.transform(X_valid)


# ==========================================
# 3. PREMIER ENTRAINEMENT : TROUVER LE
#    NOMBRE OPTIMAL D'ARBRES
# ==========================================

model = xgb.XGBClassifier(
    n_estimators=8000,
    max_depth=2,
    learning_rate=0.02,
    objective="binary:logistic",
    eval_metric="auc",
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    reg_lambda=3,
    gamma=0.5,
    random_state=42,
    early_stopping_rounds=700,
    n_jobs=-1,
    tree_method="hist"
)

model.fit(
    X_train_encoded,
    y_train,
    eval_set=[(X_valid_encoded, y_valid)],
    verbose=100
)

predictions = model.predict_proba(X_valid_encoded)[:, 1]

score = roc_auc_score(y_valid, predictions)

print("\nROC AUC validation :", score)
print("Meilleure iteration :", model.best_iteration)

best_n_estimators = model.best_iteration + 1

print("Nombre d'arbres retenu :", best_n_estimators)


# ==========================================
# 4. REENTRAINEMENT SUR TOUTES LES DONNEES
# ==========================================

final_preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", "passthrough", num_cols)
    ]
)

X_encoded = final_preprocessor.fit_transform(X)
X_test_encoded = final_preprocessor.transform(X_test)

final_model = xgb.XGBClassifier(
    n_estimators=best_n_estimators,
    max_depth=2,
    learning_rate=0.02,
    objective="binary:logistic",
    eval_metric="auc",
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    reg_lambda=3,
    random_state=42,
    n_jobs=-1,
    gamma=0.5,
    tree_method="hist"
)

final_model.fit(X_encoded, y, verbose=False)


# ==========================================
# 5. PREDICTIONS ET FICHIER KAGGLE
# ==========================================

test_predictions = final_model.predict_proba(X_test_encoded)[:, 1]

submission = pd.read_csv("sample_submission.csv")

submission["is_fraud"] = test_predictions

submission.to_csv("submission_fe.csv", index=False)

print("\nFichier créé !")
print(submission.head())
