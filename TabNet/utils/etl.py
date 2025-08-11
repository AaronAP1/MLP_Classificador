import os, joblib
import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder

def preprocess_and_load(config, paths, feature_names_out_path=None):
    test_size = config["data"]["test_size"]
    val_size  = config["data"]["val_size"]
    rs        = config["data"]["random_state"]
    df = pd.read_csv(config["data"]["raw_csv"], low_memory=False)
    cat = config["etl"]["cat_cols"]
    num = config["etl"]["num_cols"]
    tgt = config["etl"]["target_col"]
    leakage_vars = set(config["etl"].get("leakage_vars", []))
    cat = [c for c in cat if c not in leakage_vars]
    num = [c for c in num if c not in leakage_vars]
    X = df[cat + num].copy()
    y = df[tgt]
    for col in num:
        if col in X.columns:
            X[col] = X[col].replace([np.inf,-np.inf], np.nan)
            med = X[col].median()
            X[col] = X[col].fillna(med if not np.isnan(med) else 0)
    for col in cat:
        if col in X.columns:
            X[col] = X[col].fillna('Desconocido')
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    X_tmp, X_test, y_tmp, y_test = train_test_split(X, y_enc, test_size=test_size, random_state=rs, stratify=y_enc)
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(X_tmp, y_tmp, test_size=val_ratio, random_state=rs, stratify=y_tmp)
    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    scaler = StandardScaler()
    X_tr_cat = ohe.fit_transform(X_train[cat]) if cat else np.zeros((len(X_train),0))
    X_tr_num = scaler.fit_transform(X_train[num]) if num else np.zeros((len(X_train),0))
    X_va_cat = ohe.transform(X_val[cat]) if cat else np.zeros((len(X_val),0))
    X_va_num = scaler.transform(X_val[num]) if num else np.zeros((len(X_val),0))
    X_te_cat = ohe.transform(X_test[cat]) if cat else np.zeros((len(X_test),0))
    X_te_num = scaler.transform(X_test[num]) if num else np.zeros((len(X_test),0))
    joblib.dump(ohe, os.path.join(paths["processed_data_dir"], "ohe.pkl"))
    joblib.dump(scaler, os.path.join(paths["processed_data_dir"], "scaler.pkl"))
    joblib.dump(le, os.path.join(paths["processed_data_dir"], "label_encoder.pkl"))
    cat_idxs = list(range(len(cat)))
    cat_dims = [len(ohe.categories_[i]) for i in range(len(cat))]
    cat_emb_dim = [min(50, (dim+1)//2) for dim in cat_dims]
    return X_train.values, y_train, X_val.values, y_val, X_test.values, y_test, cat_idxs, cat_dims, cat_emb_dim, len(le.classes_)
