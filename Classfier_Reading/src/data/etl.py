"""
Pipeline ETL completo para clasificación de comportamiento lector.

Este módulo centraliza todo el procesamiento de datos:
1. Carga de datos brutos
2. Selección de features relevantes
3. Creación de variable objetivo (target)
4. Split estratificado 70/15/15
5. Preprocesamiento (encoding + scaling)
6. Guardado de datasets procesados
"""

import os
import json
import glob
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
import joblib


def load_metadata_mappings(metadata_dir: str) -> Dict[str, Dict]:
    """
    Carga los diccionarios de datos desde archivos Excel.
    
    Args:
        metadata_dir: Directorio con archivos .xlsx de diccionarios
        
    Returns:
        Dict con mapeos {variable: {codigo: etiqueta}}
    """
    mapping_dict = {}
    metadata_path = Path(metadata_dir)
    
    if not metadata_path.exists():
        print(f"⚠️  Directorio metadata no encontrado: {metadata_dir}")
        return mapping_dict
    
    print(f"\n📚 Cargando diccionarios desde: {metadata_dir}")
    
    for xlsx_file in metadata_path.glob("*.xlsx"):
        try:
            sheets = pd.read_excel(xlsx_file, sheet_name=None)
            for var, df_map in sheets.items():
                # Verificar que tenga las columnas necesarias
                if {"Código", "Etiqueta"}.issubset(df_map.columns):
                    # Eliminar duplicados por Código conservando la primera ocurrencia
                    df_map = df_map.drop_duplicates(subset=["Código"], keep='first')
                    mapping_dict[var] = df_map.set_index("Código")["Etiqueta"].to_dict()
                    print(f"   ✅ {var}: {len(mapping_dict[var])} mapeos")
        except Exception as e:
            print(f"   ⚠️  Error leyendo {xlsx_file.name}: {e}")
    
    print(f"✅ Total variables con diccionario: {len(mapping_dict)}\n")
    return mapping_dict


def select_features(grupo: str) -> Dict[str, List[str]]:
    """
    Selecciona las features relevantes según el grupo etario.
    
    Basado en el análisis de los diccionarios de datos:
    - Capitulo 200: Características demográficas
    - Capitulo 300-400: Prácticas lectoras adultos (18-64 años)
    - Capitulo 500: Prácticas lectoras niños/jóvenes (0-17 años)
    
    Args:
        grupo: 'adultos', 'jovenes', o 'ninos'
        
    Returns:
        Dict con 'cat_cols', 'num_cols', 'target_related'
    """
    # Variables demográficas comunes (Cap. 200)
    demograficas_cat = ["P201", "P203", "P206", "P207", "P208"]  # Relación, sexo, edad, etc.
    demograficas_num = ["P210_A", "P210_M"]  # Fecha de nacimiento (año, mes)
    
    # Variables socioeconómicas
    socioeconomicas_num = ["FACTORFINAL_HOGAR", "P310_N", "P310_A"]
    
    if grupo == "adultos":
        # Capítulo 300-400: Prácticas lectoras adultos
        lectura_cat = [
            "P311",    # ¿Lee por placer/información?
            "P315_1", "P315_2", "P315_3", "P315_4", "P315_5", "P315_6",  # Tipos de lectura
            "P409",    # Frecuencia de lectura
            "P505", "P506", "P507", "P508", "P509"  # Del hogar
        ]
        lectura_num = []
        
    elif grupo in ["jovenes", "ninos"]:
        # Capítulo 500: Prácticas lectoras niños/jóvenes
        lectura_cat = [
            "P505", "P506", "P507", "P508", "P509", "P510",  # Acceso a materiales
            "P513", "P513A",  # Lectura digital
            "P514_1", "P514_2", "P514_3",  # Tipos de lectura
            "P515", "P516", "P517"  # Frecuencia y hábitos
        ]
        lectura_num = []
    else:
        raise ValueError(f"Grupo inválido: {grupo}. Use 'adultos', 'jovenes' o 'ninos'")
    
    # Combinar todas las features
    cat_cols = demograficas_cat + lectura_cat
    num_cols = demograficas_num + socioeconomicas_num + lectura_num
    
    # Variables relacionadas con el target (para evitar data leakage)
    target_related = ["P517", "P311", "P409", "lectura_score", "habito_3clases"]
    
    return {
        "cat_cols": cat_cols,
        "num_cols": num_cols,
        "target_related": target_related
    }


def create_target_variable(df: pd.DataFrame, grupo: str) -> pd.DataFrame:
    """
    Crea la variable objetivo 'comportamiento_lector' con 3 clases.
    
    Clases:
    - 0: No lector
    - 1: Lector medio
    - 2: Lector frecuente
    
    Args:
        df: DataFrame con datos crudos
        grupo: 'adultos', 'jovenes', o 'ninos'
        
    Returns:
        DataFrame con columna 'comportamiento_lector'
    """
    print(f"\n🎯 Creando variable objetivo para grupo: {grupo.upper()}")
    
    df = df.copy()
    
    # Estrategia 1: Usar P517 si está disponible (frecuencia de lectura)
    if "P517" in df.columns:
        print(f"   Usando P517 (frecuencia de lectura)")
        mapping = {
            1.0: 0,  # Nunca → No lector
            2.0: 1,  # De vez en cuando → Lector medio
            3.0: 1,  # 1-2 veces por semana → Lector medio
            4.0: 2,  # 3-4 veces por semana → Lector frecuente
            5.0: 2   # Todos los días → Lector frecuente
        }
        df["comportamiento_lector"] = df["P517"].map(mapping)
        
        # Si hay valores válidos, usamos esta variable
        valid_count = df["comportamiento_lector"].notna().sum()
        if valid_count > len(df) * 0.1:  # Al menos 10% de datos válidos
            print(f"   ✅ Target creado desde P517: {valid_count:,} registros válidos")
            return df
        else:
            print(f"   ⚠️  P517 tiene muy pocos valores válidos ({valid_count}), usando método alternativo")
            df.drop(columns=["comportamiento_lector"], inplace=True)
    
    # Estrategia 2: Derivar score de lectura desde múltiples variables
    print(f"   Derivando score desde variables múltiples...")
    df['lectura_score'] = 0.0
    
    # Variables de tipo de lectura (P315_* para adultos, P514_* para niños/jóvenes)
    if grupo == "adultos":
        lectura_vars = ['P315_1', 'P315_2', 'P315_3', 'P315_4', 'P315_5', 'P315_6']
        frecuencia_var = 'P409'
    else:
        lectura_vars = ['P514_1', 'P514_2', 'P514_3']
        frecuencia_var = 'P515'
    
    # Sumar puntos por tipos de lectura
    for var in lectura_vars:
        if var in df.columns:
            mask = df[var].notna()
            df.loc[mask, 'lectura_score'] += df.loc[mask, var] * 0.1
    
    # Ajustar por P311 (si lee) para adultos
    if 'P311' in df.columns:
        mask = df['P311'].notna()
        df.loc[mask & (df['P311'] == 0), 'lectura_score'] = 0.0
    
    # Ajustar por frecuencia de lectura
    if frecuencia_var in df.columns:
        mask = df[frecuencia_var].notna()
        # Normalizar a [0, 1] y ponderar
        df.loc[mask, 'lectura_score'] += (df.loc[mask, frecuencia_var] - 1) / 5 * 0.3
    
    # Clip al rango [0, 1]
    df['lectura_score'] = np.clip(df['lectura_score'], 0, 1)
    
    # Convertir score a categorías
    def score_to_category(score):
        if pd.isna(score) or score < 0.2:
            return 0  # No lector
        elif score < 0.6:
            return 1  # Lector medio
        else:
            return 2  # Lector frecuente
    
    df["comportamiento_lector"] = df['lectura_score'].apply(score_to_category)
    
    # Estadísticas
    print(f"   📊 Distribución del target:")
    for clase, nombre in [(0, "No lector"), (1, "Lector medio"), (2, "Lector frecuente")]:
        count = (df["comportamiento_lector"] == clase).sum()
        pct = count / len(df) * 100
        print(f"      {clase}: {nombre:20s} - {count:6,} ({pct:5.2f}%)")
    
    print(f"   ✅ Target derivado desde score de lectura\n")
    return df


def split_data(
    X: pd.DataFrame, 
    y: pd.Series,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Split estratificado en train/val/test (70/15/15).
    
    Args:
        X: Features
        y: Target
        test_size: Proporción para test (default: 0.15)
        val_size: Proporción para validación (default: 0.15)
        random_state: Semilla para reproducibilidad
        
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    print(f"\n✂️  Separando datos (70/15/15)...")
    print(f"   Total: {len(X):,} registros")
    
    # Calcular tamaños
    train_size = 1.0 - test_size - val_size
    
    # Primera división: train+val vs test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, 
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    
    # Segunda división: train vs val
    val_size_adjusted = val_size / (train_size + val_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_size_adjusted,
        random_state=random_state,
        stratify=y_temp
    )
    
    print(f"   Train: {len(X_train):6,} ({len(X_train)/len(X)*100:5.2f}%)")
    print(f"   Val:   {len(X_val):6,} ({len(X_val)/len(X)*100:5.2f}%)")
    print(f"   Test:  {len(X_test):6,} ({len(X_test)/len(X)*100:5.2f}%)")
    
    # Verificar estratificación
    print(f"\n   📊 Distribución por split:")
    for split_name, y_split in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
        dist = y_split.value_counts(normalize=True).sort_index()
        dist_str = " | ".join([f"Clase {i}: {v*100:5.2f}%" for i, v in dist.items()])
        print(f"      {split_name:5s}: {dist_str}")
    
    print(f"   ✅ Split completado\n")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def preprocess_features(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    cat_cols: List[str],
    num_cols: List[str]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, ColumnTransformer, List[str]]:
    """
    Aplica preprocessing: OneHotEncoding para categóricas, StandardScaler para numéricas.
    
    Args:
        X_train, X_val, X_test: DataFrames de features
        cat_cols: Columnas categóricas
        num_cols: Columnas numéricas
        
    Returns:
        X_train_proc, X_val_proc, X_test_proc, preprocessor, feature_names
    """
    print(f"🔧 Aplicando preprocessing...")
    
    # Filtrar columnas que realmente existen
    cat_cols_exist = [c for c in cat_cols if c in X_train.columns]
    num_cols_exist = [c for c in num_cols if c in X_train.columns]
    
    print(f"   Categóricas: {len(cat_cols_exist)} columnas")
    print(f"   Numéricas:   {len(num_cols_exist)} columnas")
    
    # Crear preprocessor
    transformers = []
    
    if cat_cols_exist:
        transformers.append(
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols_exist)
        )
    
    if num_cols_exist:
        transformers.append(
            ('num', StandardScaler(), num_cols_exist)
        )
    
    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder='drop'
    )
    
    # Fit en train, transform en todos
    X_train_proc = preprocessor.fit_transform(X_train)
    X_val_proc = preprocessor.transform(X_val)
    X_test_proc = preprocessor.transform(X_test)
    
    # Obtener nombres de features
    feature_names = []
    
    if cat_cols_exist:
        cat_encoder = preprocessor.named_transformers_['cat']
        for i, col in enumerate(cat_cols_exist):
            categories = cat_encoder.categories_[i]
            feature_names.extend([f"{col}_{cat}" for cat in categories])
    
    if num_cols_exist:
        feature_names.extend(num_cols_exist)
    
    print(f"   ✅ Features transformadas: {len(feature_names)}")
    print(f"      Train: {X_train_proc.shape}")
    print(f"      Val:   {X_val_proc.shape}")
    print(f"      Test:  {X_test_proc.shape}\n")
    
    return X_train_proc, X_val_proc, X_test_proc, preprocessor, feature_names


def save_processed_data(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    y_train: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series,
    preprocessor: ColumnTransformer,
    feature_names: List[str],
    grupo: str,
    output_dir: str
):
    """
    Guarda datasets procesados y metadata.
    
    Args:
        X_train, X_val, X_test: Arrays de features procesadas
        y_train, y_val, y_test: Series de targets
        preprocessor: Objeto ColumnTransformer fitted
        feature_names: Lista de nombres de features
        grupo: 'adultos', 'jovenes', o 'ninos'
        output_dir: Directorio de salida (e.g., 'data/processed/adultos')
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"💾 Guardando datos procesados en: {output_dir}")
    
    # Guardar CSVs
    for split_name, X_split, y_split in [
        ("train", X_train, y_train),
        ("val", X_val, y_val),
        ("test", X_test, y_test)
    ]:
        # Convertir a DataFrame
        df = pd.DataFrame(X_split, columns=feature_names)
        df['target'] = y_split.values
        
        # Guardar
        csv_path = output_path / f"{grupo}_{split_name}.csv"
        df.to_csv(csv_path, index=False)
        print(f"   ✅ {csv_path.name}: {len(df):,} registros")
    
    # Guardar preprocessor
    preprocessor_path = output_path / f"{grupo}_preprocessor.pkl"
    joblib.dump(preprocessor, preprocessor_path)
    print(f"   ✅ Preprocessor guardado: {preprocessor_path.name}")
    
    # Crear config.json
    config = {
        "grupo": grupo,
        "total_registros": int(len(X_train) + len(X_val) + len(X_test)),
        "splits": {
            "train": int(len(X_train)),
            "val": int(len(X_val)),
            "test": int(len(X_test))
        },
        "target": {
            "name": "comportamiento_lector",
            "classes": {
                0: "No lector",
                1: "Lector medio",
                2: "Lector frecuente"
            },
            "distribution": {
                "train": y_train.value_counts().to_dict(),
                "val": y_val.value_counts().to_dict(),
                "test": y_test.value_counts().to_dict()
            }
        },
        "features": {
            "total": len(feature_names),
            "names": feature_names
        },
        "preprocessing": {
            "categorical_encoding": "OneHotEncoder",
            "numerical_scaling": "StandardScaler"
        }
    }
    
    config_path = output_path / f"{grupo}_config.json"
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"   ✅ Config guardado: {config_path.name}\n")
    
    print(f"✅ Datos procesados guardados exitosamente\n")


def run_etl_pipeline(
    grupo: str,
    raw_csv_path: str,
    metadata_dir: str,
    output_dir: str,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42
):
    """
    Ejecuta el pipeline ETL completo para un grupo.
    
    Args:
        grupo: 'adultos', 'jovenes', o 'ninos'
        raw_csv_path: Ruta al CSV sin procesar
        metadata_dir: Directorio con diccionarios de datos
        output_dir: Directorio de salida
        test_size: Proporción para test (default: 0.15)
        val_size: Proporción para validación (default: 0.15)
        random_state: Semilla para reproducibilidad
    """
    print(f"\n{'='*80}")
    print(f"🚀 PIPELINE ETL - GRUPO: {grupo.upper()}")
    print(f"{'='*80}\n")
    
    # 1. Cargar datos crudos
    print(f"📂 Cargando datos desde: {raw_csv_path}")
    df = pd.read_csv(raw_csv_path, low_memory=False)
    print(f"   ✅ Cargados: {len(df):,} registros, {len(df.columns)} columnas\n")
    
    # 2. Cargar diccionarios
    mapping_dict = load_metadata_mappings(metadata_dir)
    
    # 3. Seleccionar features
    feature_config = select_features(grupo)
    print(f"📋 Features seleccionadas:")
    print(f"   Categóricas: {len(feature_config['cat_cols'])}")
    print(f"   Numéricas: {len(feature_config['num_cols'])}")
    
    # 4. Crear variable objetivo
    df = create_target_variable(df, grupo)
    
    # 5. Filtrar datos con target válido
    df_clean = df.dropna(subset=["comportamiento_lector"])
    print(f"🧹 Limpieza de datos:")
    print(f"   Antes:  {len(df):,} registros")
    print(f"   Después: {len(df_clean):,} registros")
    print(f"   Removidos: {len(df) - len(df_clean):,} ({(len(df) - len(df_clean))/len(df)*100:.2f}%)\n")
    
    # 6. Separar X e y
    all_features = feature_config['cat_cols'] + feature_config['num_cols']
    available_features = [f for f in all_features if f in df_clean.columns]
    
    X = df_clean[available_features].copy()
    y = df_clean["comportamiento_lector"].astype(int)
    
    print(f"📊 Dataset final:")
    print(f"   Registros: {len(X):,}")
    print(f"   Features: {len(available_features)}")
    
    # 7. Split estratificado
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        X, y, test_size=test_size, val_size=val_size, random_state=random_state
    )
    
    # 8. Preprocessing
    X_train_proc, X_val_proc, X_test_proc, preprocessor, feature_names = preprocess_features(
        X_train, X_val, X_test,
        cat_cols=feature_config['cat_cols'],
        num_cols=feature_config['num_cols']
    )
    
    # 9. Guardar datos procesados
    save_processed_data(
        X_train_proc, X_val_proc, X_test_proc,
        y_train, y_val, y_test,
        preprocessor, feature_names,
        grupo, output_dir
    )
    
    print(f"{'='*80}")
    print(f"✅ PIPELINE COMPLETADO PARA: {grupo.upper()}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    # Configuración base
    base_dir = Path(__file__).parent.parent.parent / "data"
    
    # Ejecutar para cada grupo
    grupos = ["adultos", "jovenes", "ninos"]
    
    for grupo in grupos:
        run_etl_pipeline(
            grupo=grupo,
            raw_csv_path=str(base_dir / "unprocessed" / f"{grupo}.csv"),
            metadata_dir=str(base_dir / "metadata"),
            output_dir=str(base_dir / "processed" / grupo),
            test_size=0.15,
            val_size=0.15,
            random_state=42
        )
