import os, glob, joblib
import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
import torch


def load_metadata_mappings(metadata_dir: str):
    mapping_dict = {}
    for xlsx in glob.glob(os.path.join(metadata_dir, "*.xlsx")):
        try:
            sheets = pd.read_excel(xlsx, sheet_name=None)
            for var, df_map in sheets.items():
                if {"Código","Etiqueta"}.issubset(df_map.columns):
                    # eliminar duplicados por Código conservando la primera
                    df_map = df_map.drop_duplicates(subset=["Código"], keep='first')
                    mapping_dict[var] = df_map.set_index("Código")["Etiqueta"].to_dict()
        except Exception as e:
            print(f"Advertencia: no se pudo leer {xlsx}: {e}")
    return mapping_dict


def preprocess_and_load(config: dict, paths: dict, feature_names_out_path: str = None):
    # Seeds y params
    test_size = config["data"]["test_size"]
    val_size  = config["data"]["val_size"]
    rs        = config["data"]["random_state"]

    # Diccionarios
    # Derivar carpeta metadata a partir de paths (e.g., data/raw -> data/metadata)
    base_data_dir = os.path.dirname(paths.get("raw_data_dir", "data/raw")) or "data"
    metadata_dir = os.path.join(base_data_dir, "metadata")
    mapping_dict = load_metadata_mappings(metadata_dir)

    # Datos
    df = pd.read_csv(config["data"]["raw_csv"], low_memory=False)

    # Expandir categóricas a texto si hay mapping
    cat = config["etl"]["cat_cols"]
    for var in cat:
        txt_col = f"{var}_txt"
        if var in mapping_dict:
            df[txt_col] = df[var].map(mapping_dict[var])
        else:
            df[txt_col] = df[var].astype(str)

    # Target configurable
    tgt = config["etl"]["target_col"]
    leakage_vars = set()
    if tgt == "habito_3clases":
        mapping = {1.0:"no_lector", 2.0:"lector_medio", 3.0:"lector_medio", 4.0:"lector_frecuente", 5.0:"lector_frecuente"}
        if "P517" in df.columns:
            df["habito_3clases"] = df["P517"].map(mapping)
            # Si todo es NaN (o no hay valores), caemos a derivación por score
            if df["habito_3clases"].notna().any():
                df = df.dropna(subset=["habito_3clases"])       
                leakage_vars.update({"P517"})
            else:
                df.drop(columns=["habito_3clases"], inplace=True)
                # Fallback a derivación por score
                df['lectura_score'] = 0.0
                lectura_vars = ['P315_1','P315_2','P315_3','P315_4','P315_5','P315_6']
                for v in lectura_vars:
                    if v in df.columns:
                        mask = df[v].notna()
                        df.loc[mask,'lectura_score'] += df.loc[mask,v]*0.1
                if 'P311' in df.columns:
                    mask = df['P311'].notna()
                    df.loc[mask & (df['P311']==0), 'lectura_score'] = 0.0
                if 'P409' in df.columns:
                    mask = df['P409'].notna()
                    df.loc[mask, 'lectura_score'] += (df.loc[mask,'P409']-1)/5 * 0.3
                df['lectura_score'] = np.clip(df['lectura_score'], 0, 1)
                def score_to_category(s):
                    if pd.isna(s) or s < 0.2: return "no_lector"
                    if s < 0.6: return "lector_medio"
                    return "lector_frecuente"
                df["habito_3clases"] = df['lectura_score'].apply(score_to_category)
                leakage_vars.update({"P409","P311","lectura_score", *lectura_vars})
        else:
            # Fallback directo cuando no existe P517 (p. ej., Niños)
            df['lectura_score'] = 0.0
            lectura_vars = ['P315_1','P315_2','P315_3','P315_4','P315_5','P315_6']
            for v in lectura_vars:
                if v in df.columns:
                    mask = df[v].notna()
                    df.loc[mask,'lectura_score'] += df.loc[mask,v]*0.1
            if 'P311' in df.columns:
                mask = df['P311'].notna()
                df.loc[mask & (df['P311']==0), 'lectura_score'] = 0.0
            if 'P409' in df.columns:
                mask = df['P409'].notna()
                df.loc[mask, 'lectura_score'] += (df.loc[mask,'P409']-1)/5 * 0.3
            df['lectura_score'] = np.clip(df['lectura_score'], 0, 1)
            def score_to_category(s):
                if pd.isna(s) or s < 0.2: return "no_lector"
                if s < 0.6: return "lector_medio"
                return "lector_frecuente"
            df["habito_3clases"] = df['lectura_score'].apply(score_to_category)
            leakage_vars.update({"P409","P311","lectura_score", *lectura_vars})
    elif tgt == "habito_lectura_completo":
        df['lectura_score'] = 0.0
        if 'P517' in df.columns:
            mask = df['P517'].notna()
            df.loc[mask,'lectura_score'] = (df.loc[mask,'P517'] - 1)/4
        lectura_vars = ['P315_1','P315_2','P315_3','P315_4','P315_5','P315_6']
        for v in lectura_vars:
            if v in df.columns:
                mask = df[v].notna()
                df.loc[mask,'lectura_score'] += df.loc[mask,v]*0.1
        if 'P311' in df.columns:
            mask = df['P311'].notna()
            df.loc[mask & (df['P311']==0), 'lectura_score'] = 0.0
        if 'P409' in df.columns:
            mask = df['P409'].notna()
            df.loc[mask, 'lectura_score'] += (df.loc[mask,'P409']-1)/5 * 0.3
        df['lectura_score'] = np.clip(df['lectura_score'], 0, 1)
        def score_to_category(s):
            if pd.isna(s) or s < 0.2: return "no_lector"
            if s < 0.6: return "lector_medio"
            return "lector_frecuente"
        df["habito_lectura_completo"] = df['lectura_score'].apply(score_to_category)
        leakage_vars.update({"P517","P409","P311","lectura_score", *lectura_vars})
    else:
        df = df.dropna(subset=[tgt])

    # Conjunto de features
    cat_txt = [f"{v}_txt" for v in cat]
    num     = config["etl"]["num_cols"]

    # Verificar presencia, evitando fuga de etiquetas (drop explícito)
    available_cols = []
    dropped_for_leakage = []
    for col in cat_txt + num:
        base = col.replace('_txt','')
        if base in leakage_vars or col in leakage_vars:
            dropped_for_leakage.append(col)
            continue
        if col in df.columns:
            available_cols.append(col)
        else:
            print(f"Advertencia: columna {col} no encontrada")
    if not available_cols:
        raise ValueError("No se encontraron columnas válidas")
    if dropped_for_leakage:
        print(f"🔒 Evitando fuga de etiquetas. Se excluyeron: {sorted(set(dropped_for_leakage))}")

    X = df[available_cols].copy()
    y = df[tgt]

    # Limpieza
    for col in num:
        if col in X.columns:
            X[col] = X[col].replace([np.inf,-np.inf], np.nan)
            med = X[col].median()
            if np.isnan(med): med = 0
            X[col] = X[col].fillna(med)
    for col in cat_txt:
        if col in X.columns:
            X[col] = X[col].fillna('Desconocido')

    # Encode target
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # Split 70/15/15
    X_tmp, X_test, y_tmp, y_test = train_test_split(X, y_enc, test_size=test_size, random_state=rs, stratify=y_enc)
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(X_tmp, y_tmp, test_size=val_ratio, random_state=rs, stratify=y_tmp)

    # Preprocesador (usar solo columnas existentes tras exclusiones)
    cat_txt_used = [c for c in cat_txt if c in X.columns]
    num_used = [c for c in num if c in X.columns]
    preproc = ColumnTransformer([
        ("cat", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), cat_txt_used),
        ("num", StandardScaler(), num_used)
    ])

    X_tr = preproc.fit_transform(X_train)
    X_va = preproc.transform(X_val)
    X_te = preproc.transform(X_test)

    # Guardar artefactos
    os.makedirs(paths["processed_data_dir"], exist_ok=True)
    joblib.dump(preproc, os.path.join(paths["processed_data_dir"], "preprocessor.pkl"))
    joblib.dump(le,      os.path.join(paths["processed_data_dir"], "label_encoder.pkl"))

    # Guardar nombres de features procesadas para interpretabilidad
    if feature_names_out_path is not None:
        try:
            feat_names = []
            # Obtener nombres desde el OneHotEncoder
            ohe: OneHotEncoder = preproc.named_transformers_["cat"]
            ohe_features = []
            if hasattr(ohe, 'get_feature_names_out'):
                ohe_features = list(ohe.get_feature_names_out(cat_txt_used))
            else:
                ohe_features = []
            feat_names = ohe_features + num_used
            with open(feature_names_out_path, 'w', encoding='utf-8') as f:
                for n in feat_names:
                    f.write(str(n)+"\n")
        except Exception as e:
            print(f"No se pudieron obtener nombres de features: {e}")

    # Guardar splits si se pide
    if config["data"].get("save_splits", False):
        os.makedirs(paths["processed_data_dir"], exist_ok=True)
        for name, idx in [("train_idx", X_train.index), ("val_idx", X_val.index), ("test_idx", X_test.index)]:
            np.savetxt(os.path.join(paths["processed_data_dir"], f"{name}.txt"), idx, fmt="%s")

    # DataLoaders
    def to_dl(X_arr, y_arr, bs, shuffle, weights=None):
        Xt = torch.from_numpy(X_arr.astype(np.float32))
        yt = torch.from_numpy(y_arr.astype(np.int64))
        ds = TensorDataset(Xt, yt)
        if weights is not None:
            sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
            return DataLoader(ds, batch_size=bs, sampler=sampler)
        return DataLoader(ds, batch_size=bs, shuffle=shuffle)

    bs = config["model"]["batch_size"]

    # Sampler opcional
    sampler = None
    if config["model"].get("use_weighted_sampler", False):
        classes, counts = np.unique(y_train, return_counts=True)
        class_weights = {c: 1.0 / cnt for c, cnt in zip(classes, counts)}
        weights = np.array([class_weights[c] for c in y_train], dtype=np.float32)
        sampler = weights

    dl_tr = to_dl(X_tr, y_train, bs, shuffle=(sampler is None), weights=sampler)
    dl_va = to_dl(X_va, y_val,   bs, shuffle=False)
    dl_te = to_dl(X_te, y_test,  bs, shuffle=False)

    return dl_tr, dl_va, dl_te, X_tr.shape[1], len(le.classes_)
