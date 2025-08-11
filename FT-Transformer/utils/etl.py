import os, glob, joblib
import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
import torch

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
    def to_dl(X_num, X_cat, y_arr, bs, shuffle):
        Xt_num = torch.from_numpy(X_num.astype(np.float32))
        Xt_cat = torch.from_numpy(X_cat.astype(np.float32))
        yt = torch.from_numpy(y_arr.astype(np.int64))
        ds = TensorDataset(Xt_num, Xt_cat, yt)
        return DataLoader(ds, batch_size=bs, shuffle=shuffle)
    bs = config["model"]["batch_size"]
    dl_tr = to_dl(X_tr_num, X_tr_cat, y_train, bs, True)
    dl_va = to_dl(X_va_num, X_va_cat, y_val, bs, False)
    dl_te = to_dl(X_te_num, X_te_cat, y_test, bs, False)
    return dl_tr, dl_va, dl_te, X_tr_num.shape[1], X_tr_cat.shape[1], len(le.classes_)
