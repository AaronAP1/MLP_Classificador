import os, joblib
import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
import torch


def _build_target(df: pd.DataFrame, tgt: str):
    leakage_vars = set()
    if tgt == "habito_3clases":
        mapping = {1.0: "no_lector", 2.0: "lector_medio", 3.0: "lector_medio", 4.0: "lector_frecuente", 5.0: "lector_frecuente"}
        if "P517" in df.columns:
            df[tgt] = df["P517"].map(mapping)
            if not df[tgt].notna().any():
                df.drop(columns=[tgt], errors='ignore', inplace=True)
        if tgt not in df.columns:
            # Fallback via lectura_score (como en MLP_MA)
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
    elif tgt == "habito_lectura_completo":
        df['lectura_score'] = 0.0
        if 'P517' in df.columns:
            m = df['P517'].notna()
            df.loc[m, 'lectura_score'] = (df.loc[m, 'P517'] - 1) / 4
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
        leakage_vars.update({"P517","P409","P311","lectura_score", *lectura_vars})
    else:
        # Assume ya existe la columna
        if tgt not in df.columns:
            raise KeyError(f"Target {tgt} no encontrado en DF")
    return df, leakage_vars


def preprocess_and_load(config, paths, feature_names_out_path=None):
    test_size = config["data"]["test_size"]
    val_size  = config["data"]["val_size"]
    rs        = config["data"]["random_state"]

    df = pd.read_csv(config["data"]["raw_csv"], low_memory=False)
    tgt = config["etl"]["target_col"]
    df, auto_leak = _build_target(df, tgt)
    df = df.dropna(subset=[tgt])

    # Features y anti-fuga
    cat_cols = list(config["etl"]["cat_cols"])
    num_cols = list(config["etl"]["num_cols"])
    leakage_vars = set(config["etl"].get("leakage_vars", [])) | auto_leak
    cat_cols = [c for c in cat_cols if c in df.columns and c not in leakage_vars]
    num_cols = [c for c in num_cols if c in df.columns and c not in leakage_vars]
    if not (cat_cols or num_cols):
        raise ValueError("No hay columnas válidas tras anti-fuga")

    X = df[cat_cols + num_cols].copy()
    y = df[tgt]

    # Limpieza numéricas
    for col in num_cols:
        if col in X.columns:
            X[col] = X[col].replace([np.inf, -np.inf], np.nan)
            med = X[col].median()
            X[col] = X[col].fillna(med if not np.isnan(med) else 0)
    # Categóricas a string para mapeo
    for col in cat_cols:
        if col in X.columns:
            X[col] = X[col].astype(str).fillna('Desconocido')

    # Target encoder
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # Split 70/15/15
    X_tmp, X_test, y_tmp, y_test = train_test_split(X, y_enc, test_size=test_size, random_state=rs, stratify=y_enc)
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(X_tmp, y_tmp, test_size=val_ratio, random_state=rs, stratify=y_tmp)

    # Fit scaler en numéricas del train
    scaler = StandardScaler()
    if num_cols:
        X_tr_num = scaler.fit_transform(X_train[num_cols])
        X_va_num = scaler.transform(X_val[num_cols])
        X_te_num = scaler.transform(X_test[num_cols])
    else:
        X_tr_num = np.zeros((len(X_train), 0), dtype=np.float32)
        X_va_num = np.zeros((len(X_val), 0), dtype=np.float32)
        X_te_num = np.zeros((len(X_test), 0), dtype=np.float32)

    # Mapear categóricas a índices por columna (train-only) con OOV
    cat_mappings = {}
    cat_dims = []
    if cat_cols:
        X_tr_cat = np.zeros((len(X_train), len(cat_cols)), dtype=np.int64)
        X_va_cat = np.zeros((len(X_val), len(cat_cols)), dtype=np.int64)
        X_te_cat = np.zeros((len(X_test), len(cat_cols)), dtype=np.int64)
        for j, col in enumerate(cat_cols):
            cats = pd.Index(X_train[col].astype(str).unique())
            mapping = {v: i for i, v in enumerate(cats)}
            oov_idx = len(mapping)
            cat_mappings[col] = {"mapping": mapping, "oov": oov_idx}
            cat_dims.append(oov_idx + 1)
            # Transform
            def map_series(s):
                return s.astype(str).map(mapping).fillna(oov_idx).astype(np.int64).values
            X_tr_cat[:, j] = map_series(X_train[col])
            X_va_cat[:, j] = map_series(X_val[col])
            X_te_cat[:, j] = map_series(X_test[col])
    else:
        X_tr_cat = np.zeros((len(X_train), 0), dtype=np.int64)
        X_va_cat = np.zeros((len(X_val), 0), dtype=np.int64)
        X_te_cat = np.zeros((len(X_test), 0), dtype=np.int64)

    # Guardar artefactos
    os.makedirs(paths["processed_data_dir"], exist_ok=True)
    joblib.dump(scaler, os.path.join(paths["processed_data_dir"], "ft_scaler.pkl"))
    joblib.dump(le, os.path.join(paths["processed_data_dir"], "label_encoder.pkl"))
    joblib.dump({"cat_mappings": cat_mappings, "cat_dims": cat_dims, "cat_cols": cat_cols, "num_cols": num_cols},
                os.path.join(paths["processed_data_dir"], "ft_cat_mappings.pkl"))

    # DataLoaders
    def to_dl(Xn, Xc, y_arr, bs, shuffle):
        Xn_t = torch.from_numpy(Xn.astype(np.float32))
        Xc_t = torch.from_numpy(Xc.astype(np.int64))
        yt = torch.from_numpy(y_arr.astype(np.int64))
        ds = TensorDataset(Xn_t, Xc_t, yt)
        return DataLoader(ds, batch_size=bs, shuffle=shuffle)

    bs = config["model"]["batch_size"]
    dl_tr = to_dl(X_tr_num, X_tr_cat, y_train, bs, True)
    dl_va = to_dl(X_va_num, X_va_cat, y_val, bs, False)
    dl_te = to_dl(X_te_num, X_te_cat, y_test, bs, False)

    return dl_tr, dl_va, dl_te, X_tr_num.shape[1], cat_dims, len(le.classes_)
