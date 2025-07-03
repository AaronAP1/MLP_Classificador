import os, glob, joblib
import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from torch.utils.data import DataLoader, TensorDataset
import torch

def preprocess_and_load(config: dict, paths: dict):
    # 0) Parámetros de split del config
    test_size = config["data"]["test_size"]
    val_size  = config["data"]["val_size"]
    rs        = config["data"]["random_state"]

    # 1) Cargar todos los diccionarios .xlsx
    mapping_dict = {}
    for xlsx in glob.glob("data/metadata/*.xlsx"):
        sheets = pd.read_excel(xlsx, sheet_name=None)
        for var, df_map in sheets.items():
            if {"Código","Etiqueta"}.issubset(df_map.columns):
                mapping_dict[var] = df_map.set_index("Código")["Etiqueta"].to_dict()

    # 2) Leer CSV bruto
    df = pd.read_csv(config["data"]["raw_csv"])

    # 3) Crear columnas de texto para cada cat_col
    cat = config["etl"]["cat_cols"]
    for var in cat:
        txt_col = f"{var}_txt"
        if var in mapping_dict:
            df[txt_col] = df[var].map(mapping_dict[var])
        else:
            df[txt_col] = df[var].astype(str)

    # 4) Reagrupar el target en 3 clases
    mapping = {
        0: "no_lector",
        2: "lector_medio", 3: "lector_medio",
        4: "lector_frecuente", 5: "lector_frecuente"
    }
    df["habito_3clases"] = df["P517"].map(mapping)
    df = df.dropna(subset=["habito_3clases"])

    # 5) Definir X (con las columnas _txt) y y
    cat_txt = [f"{v}_txt" for v in cat]
    num     = config["etl"]["num_cols"]
    tgt     = config["etl"]["target_col"]      # debe ser "habito_3clases"

    X = df[cat_txt + num]
    y = df[tgt]

    # 6) Label encode del target
    le    = LabelEncoder()
    y_enc = le.fit_transform(y)
    print("Clases target:", le.classes_)
    n_classes = len(le.classes_)

    # 7) Split estratificado
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y_enc,
        test_size=test_size,
        random_state=rs,
        stratify=y_enc
    )
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp,
        test_size=val_ratio,
        random_state=rs,
        stratify=y_tmp
    )

    # 8) Preprocesamiento: one-hot sobre cat_txt, escalado numérico
    preproc = ColumnTransformer([
        ("cat", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), cat_txt),
        ("num", StandardScaler(), num)
    ])
    X_tr = preproc.fit_transform(X_train)
    X_va = preproc.transform(X_val)
    X_te = preproc.transform(X_test)

    # 9) Guardar preprocesadores
    os.makedirs(paths["processed_data_dir"], exist_ok=True)
    joblib.dump(preproc, os.path.join(paths["processed_data_dir"], "preprocessor.pkl"))
    joblib.dump(le,     os.path.join(paths["processed_data_dir"], "label_encoder.pkl"))

    # 10) DataLoaders PyTorch
    def to_dl(X_arr, y_arr, bs, shuffle):
        Xt = torch.from_numpy(X_arr.astype(np.float32))
        yt = torch.from_numpy(y_arr.astype(np.int64))
        ds = TensorDataset(Xt, yt)
        return DataLoader(ds, batch_size=bs, shuffle=shuffle)

    bs = config["model"]["batch_size"]
    dl_tr = to_dl(X_tr, y_train, bs, True)
    dl_va = to_dl(X_va, y_val,   bs, False)
    dl_te = to_dl(X_te, y_test,  bs, False)

    # 11) Retornar loaders, dimensión de entrada y n° de clases
    return dl_tr, dl_va, dl_te, X_tr.shape[1], n_classes
