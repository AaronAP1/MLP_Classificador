import os, joblib
import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple

def _build_target(df: pd.DataFrame, tgt: str):
    leakage_vars = set()
    if tgt == "habito_3clases":
        mapping = {1.0: "no_lector", 2.0: "lector_medio", 3.0: "lector_medio", 4.0: "lector_frecuente", 5.0: "lector_frecuente"}
        if "P517" in df.columns:
            df[tgt] = df["P517"].map(mapping)
            if not df[tgt].notna().any():
                df.drop(columns=[tgt], errors='ignore', inplace=True)
        if tgt not in df.columns:
            df['lectura_score'] = 0.0
            lectura_vars = ['P315_1','P315_2','P315_3','P315_4','P315_5','P315_6']
            for v in lectura_vars:
                if v in df.columns:
                    m = df[v].notna()
                    df.loc[m, 'lectura_score'] += df.loc[m, v] * 0.1
            if 'P311' in df.columns:
                m = df['P311'].notna()
                df.loc[m & (df['P311'] == 0), 'lectura_score'] = 0.0
            if 'P409' in df.columns:
                m = df['P409'].notna()
                df.loc[m, 'lectura_score'] += (df.loc[m, 'P409'] - 1) / 5 * 0.3
            df['lectura_score'] = np.clip(df['lectura_score'], 0, 1)
            def s2c(s):
                if pd.isna(s) or s < 0.2: return "no_lector"
                if s < 0.6: return "lector_medio"
                return "lector_frecuente"
            df[tgt] = df['lectura_score'].apply(s2c)
            leakage_vars.update({"P409","P311","lectura_score","P315_1","P315_2","P315_3","P315_4","P315_5","P315_6"})
        else:
            leakage_vars.update({"P517"})
    return df, leakage_vars

def preprocess_and_load(config, paths, feature_names_out_path=None):
    test_size = config["data"]["test_size"]
    val_size  = config["data"]["val_size"]
    rs        = config["data"]["random_state"]
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    df = pd.read_csv(config["data"]["raw_csv"], low_memory=False)
    tgt = config["etl"]["target_col"]
    df, auto_leak = _build_target(df, tgt)
    df = df.dropna(subset=[tgt])
    cat = list(config["etl"]["cat_cols"])
    num = list(config["etl"]["num_cols"])
    leakage_vars = set(config["etl"].get("leakage_vars", [])) | auto_leak
    cat = [c for c in cat if c in df.columns and c not in leakage_vars]
    num = [c for c in num if c in df.columns and c not in leakage_vars]
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
    # Numeric scaler
    scaler = StandardScaler()
    X_tr_num = scaler.fit_transform(X_train[num]) if num else np.zeros((len(X_train),0))
    X_va_num = scaler.transform(X_val[num]) if num else np.zeros((len(X_val),0))
    X_te_num = scaler.transform(X_test[num]) if num else np.zeros((len(X_test),0))

    # Map categorical columns to integer indices with OOV per column
    cat_mappings = {}
    if cat:
        X_tr_cat = np.zeros((len(X_train), len(cat)), dtype=np.int64)
        X_va_cat = np.zeros((len(X_val), len(cat)), dtype=np.int64)
        X_te_cat = np.zeros((len(X_test), len(cat)), dtype=np.int64)
        cat_dims = []
        for j, col in enumerate(cat):
            cats = pd.Index(X_train[col].astype(str).fillna('Desconocido').unique())
            mapping = {v: i for i, v in enumerate(cats)}
            oov_idx = len(mapping)
            cat_mappings[col] = {"mapping": mapping, "oov": oov_idx}
            cat_dims.append(oov_idx + 1)
            def map_series(s):
                return s.astype(str).fillna('Desconocido').map(mapping).fillna(oov_idx).astype(np.int64).values
            X_tr_cat[:, j] = map_series(X_train[col])
            X_va_cat[:, j] = map_series(X_val[col])
            X_te_cat[:, j] = map_series(X_test[col])
    else:
        X_tr_cat = np.zeros((len(X_train), 0), dtype=np.int64)
        X_va_cat = np.zeros((len(X_val), 0), dtype=np.int64)
        X_te_cat = np.zeros((len(X_test), 0), dtype=np.int64)
        cat_dims = []

    # Combine features: place cat indices first so cat_idxs are [0..len(cat)-1]
    X_train_full = np.hstack([X_tr_cat, X_tr_num]).astype(np.float32)
    X_val_full   = np.hstack([X_va_cat, X_va_num]).astype(np.float32)
    X_test_full  = np.hstack([X_te_cat, X_te_num]).astype(np.float32)

    # cat_idxs positions and embedding dims
    cat_idxs = list(range(len(cat)))
    cat_emb_dim = [min(50, (d + 1)//2) for d in cat_dims]

    proc_dir = paths["processed_data_dir"]
    if not os.path.isabs(proc_dir):
        proc_dir = os.path.join(base_dir, proc_dir)
    os.makedirs(proc_dir, exist_ok=True)
    joblib.dump(scaler, os.path.join(proc_dir, "scaler.pkl"))
    joblib.dump(le, os.path.join(proc_dir, "label_encoder.pkl"))
    joblib.dump({"cat_mappings": cat_mappings, "cat_dims": cat_dims, "cat_cols": cat}, os.path.join(proc_dir, "tabnet_cat_mappings.pkl"))
    return X_train_full, y_train, X_val_full, y_val, X_test_full, y_test, cat_idxs, cat_dims, cat_emb_dim, len(le.classes_)
