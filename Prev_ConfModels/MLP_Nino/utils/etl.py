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

    # 4) Procesar la variable objetivo según el tipo
    tgt = config["etl"]["target_col"]
    
    if tgt == "habito_3clases":
        # Reagrupar el target en 3 clases para hábitos de lectura usando P517
        mapping = {
            1.0: "no_lector",           # Nunca lee
            2.0: "lector_medio",        # Lee poco
            3.0: "lector_medio",        # Lee ocasionalmente
            4.0: "lector_frecuente",    # Lee frecuentemente
            5.0: "lector_frecuente"     # Lee muy frecuentemente
        }
        df["habito_3clases"] = df["P517"].map(mapping)
        df = df.dropna(subset=["habito_3clases"])
        print(f"Distribución de hábitos de lectura (P517):")
        print(df["habito_3clases"].value_counts())
    elif tgt == "habito_lectura_completo":
        # Usar el target completo creado para todo el dataset (14K muestras)
        print("🔧 Creando target de hábitos de lectura para dataset completo...")
        
        # Crear score compuesto de lectura
        df['lectura_score'] = 0.0
        
        # 1. Base: P517 (frecuencia directa) - normalizado 0-1
        if 'P517' in df.columns:
            mask_p517 = df['P517'].notna()
            df.loc[mask_p517, 'lectura_score'] = (df.loc[mask_p517, 'P517'] - 1) / 4
        
        # 2. Para el resto: variables P315 (tipos de material)
        mask_no_p517 = ~mask_p517 if 'P517' in df.columns else df.index
        lectura_vars = ['P315_1', 'P315_2', 'P315_3', 'P315_4', 'P315_5', 'P315_6']
        for var in lectura_vars:
            if var in df.columns:
                mask_valid = mask_no_p517 & df[var].notna()
                df.loc[mask_valid, 'lectura_score'] += df.loc[mask_valid, var] * 0.1
        
        # 3. P311: saber leer (base fundamental)
        if 'P311' in df.columns:
            mask_valid = mask_no_p517 & df['P311'].notna()
            df.loc[mask_valid & (df['P311'] == 0), 'lectura_score'] = 0.0
        
        # 4. P409: frecuencia alternativa
        if 'P409' in df.columns:
            mask_valid = mask_no_p517 & df['P409'].notna()
            df.loc[mask_valid, 'lectura_score'] += (df.loc[mask_valid, 'P409'] - 1) / 5 * 0.3
        
        # 5. Normalizar y categorizar
        df['lectura_score'] = np.clip(df['lectura_score'], 0, 1)
        
        def score_to_category(score):
            if pd.isna(score) or score < 0.2:
                return "no_lector"
            elif score < 0.6:
                return "lector_medio"
            else:
                return "lector_frecuente"
        
        df["habito_lectura_completo"] = df['lectura_score'].apply(score_to_category)
        print(f"Distribución de hábitos de lectura (dataset completo):")
        print(df["habito_lectura_completo"].value_counts())
    elif tgt == "edad_total":
        # Para clasificación por edades, limpiar datos faltantes
        df = df.dropna(subset=[tgt])
        # Convertir edades a enteros si están como float
        df[tgt] = df[tgt].astype(int)
        print(f"Distribución de edades:")
        print(df[tgt].value_counts().sort_index())
    else:
        # Para otros targets, simplemente limpiar datos faltantes
        df = df.dropna(subset=[tgt])

    # 5) Definir X (con las columnas _txt) y y
    cat_txt = [f"{v}_txt" for v in cat]
    num     = config["etl"]["num_cols"]
    tgt     = config["etl"]["target_col"]

    # Verificar que las columnas existen en el dataframe
    available_cols = []
    for col in cat_txt + num:
        if col in df.columns:
            available_cols.append(col)
        else:
            print(f"Advertencia: Columna {col} no encontrada en el dataset")
    
    if not available_cols:
        raise ValueError("No se encontraron columnas válidas para el modelo")
    
    print(f"Usando {len(available_cols)} características: {available_cols}")
    
    X = df[available_cols].copy()  # Crear copia para evitar warnings
    y = df[tgt]

    # 6) Limpiar datos ANTES del split
    print(f"📊 Limpiando datos antes del preprocesamiento...")
    
    # Limpiar datos numéricos - reemplazar NaN/Inf con mediana
    for col in num:
        if col in X.columns:
            # Reemplazar infinitos con NaN primero
            X[col] = X[col].replace([np.inf, -np.inf], np.nan)
            # Luego reemplazar NaN con la mediana
            median_val = X[col].median()
            if np.isnan(median_val):
                median_val = 0
            X[col] = X[col].fillna(median_val)
            print(f"Columna {col}: NaN reemplazados con {median_val}")
    
    # Limpiar datos categóricos - reemplazar NaN con 'Desconocido'
    for col in cat_txt:
        if col in X.columns:
            X[col] = X[col].fillna('Desconocido')
            print(f"Columna {col}: NaN reemplazados con 'Desconocido'")
    
    # Verificar limpieza
    print(f"✅ Datos limpios:")
    print(f"  X - NaN: {X.isnull().sum().sum()}")
    print(f"  y - NaN: {y.isnull().sum()}")

    # 7) Label encode del target
    le    = LabelEncoder()
    y_enc = le.fit_transform(y)
    print("Clases target:", le.classes_)
    n_classes = len(le.classes_)

    # 8) Split estratificado: 70% entrenamiento, 15% validación, 15% test
    # Primero separamos test (15%)
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y_enc,
        test_size=test_size,  # 0.15
        random_state=rs,
        stratify=y_enc
    )
    
    # Luego separamos entrenamiento y validación del 85% restante
    # val_size / (1 - test_size) = 0.15 / 0.85 ≈ 0.176
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp,
        test_size=val_ratio,
        random_state=rs,
        stratify=y_tmp
    )
    
    # Verificar las proporciones
    total_samples = len(X)
    train_pct = len(X_train) / total_samples * 100
    val_pct = len(X_val) / total_samples * 100  
    test_pct = len(X_test) / total_samples * 100
    
    print(f"División de datos:")
    print(f"  Entrenamiento: {len(X_train)} muestras ({train_pct:.1f}%)")
    print(f"  Validación: {len(X_val)} muestras ({val_pct:.1f}%)")
    print(f"  Test: {len(X_test)} muestras ({test_pct:.1f}%)")

    # 10) Preprocesamiento: one-hot sobre cat_txt, escalado numérico
    print(f"📊 Preparando preprocesamiento...")
    
    preproc = ColumnTransformer([
        ("cat", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), cat_txt),
        ("num", StandardScaler(), num)
    ])
    
    print(f"Ajustando preprocesador...")
    X_tr = preproc.fit_transform(X_train)
    X_va = preproc.transform(X_val)
    X_te = preproc.transform(X_test)
    
    # Verificar si hay NaN/Inf después del preprocesamiento
    print(f"✅ Verificando datos procesados:")
    print(f"  Train - NaN: {np.isnan(X_tr).sum()}, Inf: {np.isinf(X_tr).sum()}")
    print(f"  Val - NaN: {np.isnan(X_va).sum()}, Inf: {np.isinf(X_va).sum()}")
    print(f"  Test - NaN: {np.isnan(X_te).sum()}, Inf: {np.isinf(X_te).sum()}")
    
    # Si aún hay NaN/Inf (no debería), reemplazar con 0
    if np.isnan(X_tr).sum() > 0 or np.isinf(X_tr).sum() > 0:
        print("⚠️  Reemplazando NaN/Inf restantes con 0...")
        X_tr = np.nan_to_num(X_tr, nan=0.0, posinf=0.0, neginf=0.0)
        X_va = np.nan_to_num(X_va, nan=0.0, posinf=0.0, neginf=0.0)
        X_te = np.nan_to_num(X_te, nan=0.0, posinf=0.0, neginf=0.0)

    # 11) Guardar preprocesadores
    os.makedirs(paths["processed_data_dir"], exist_ok=True)
    joblib.dump(preproc, os.path.join(paths["processed_data_dir"], "preprocessor.pkl"))
    joblib.dump(le,     os.path.join(paths["processed_data_dir"], "label_encoder.pkl"))

    # 12) DataLoaders PyTorch
    def to_dl(X_arr, y_arr, bs, shuffle):
        Xt = torch.from_numpy(X_arr.astype(np.float32))
        yt = torch.from_numpy(y_arr.astype(np.int64))
        ds = TensorDataset(Xt, yt)
        return DataLoader(ds, batch_size=bs, shuffle=shuffle)

    bs = config["model"]["batch_size"]
    dl_tr = to_dl(X_tr, y_train, bs, True)
    dl_va = to_dl(X_va, y_val,   bs, False)
    dl_te = to_dl(X_te, y_test,  bs, False)

    # 13) Retornar loaders, dimensión de entrada y n° de clases
    return dl_tr, dl_va, dl_te, X_tr.shape[1], n_classes
