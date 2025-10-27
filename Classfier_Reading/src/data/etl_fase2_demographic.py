"""
Pipeline ETL FASE 2: Solo Variables Demográficas y Socioculturales

Este módulo procesa datos EXCLUYENDO variables de comportamiento lector:
- Incluye: Edad, sexo, educación, región, nivel socioeconómico
- Excluye: P514 (tipos lectura), P515 (frecuencia), P517, P311, P409

Objetivo: Demostrar capacidad predictiva de Deep Learning con datos censales
Aplicación: Clasificación masiva sin necesidad de encuestas de lectura

Universidad Peruana Unión - 2025
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import joblib

# Configurar UTF-8 para Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Agregar path del proyecto
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def select_features_demographic_only(grupo: str) -> Dict[str, List[str]]:
    """
    Selecciona SOLO features demográficas y socioculturales.
    
    FASE 2: Excluye todas las variables relacionadas con comportamiento lector
    para evaluar capacidad predictiva usando únicamente datos censales.
    
    Args:
        grupo: 'adultos', 'jovenes', o 'ninos'
        
    Returns:
        Dict con 'cat_cols', 'num_cols', 'excluded_lectura'
    """
    print(f"\n📋 Seleccionando features DEMOGRÁFICAS para: {grupo.upper()}")
    
    # ===================================================================
    # VARIABLES DEMOGRÁFICAS (Cap. 200)
    # ===================================================================
    demograficas_cat = [
        "P203",    # Sexo (1=Hombre, 2=Mujer)
        "P206",    # Nivel educativo alcanzado
        "P207",    # Ocupación principal
        "P208",    # Estado civil / conyugal
    ]
    
    demograficas_num = [
        "P210_A",  # Año de nacimiento
        "P210_M",  # Mes de nacimiento
    ]
    
    # ===================================================================
    # VARIABLES SOCIOCULTURALES
    # ===================================================================
    socioculturales_cat = [
        "CCDD",           # Departamento
        "DOMINIO",        # Dominio geográfico
        "ESTRATO",        # Estrato
        "ESTRATOSOCIO",   # Nivel socioeconómico
    ]
    
    socioculturales_num = [
        "FACTORFINAL_HOGAR",  # Factor de expansión
    ]
    
    # ===================================================================
    # VARIABLES ESPECÍFICAS POR GRUPO
    # ===================================================================
    if grupo == "adultos":
        # Adultos: Más variables laborales/educativas
        especificas_cat = [
            "P201",    # Relación de parentesco
        ]
        especificas_num = []
        
    elif grupo in ["jovenes", "ninos"]:
        # Niños/Jóvenes: Variables del hogar
        especificas_cat = [
            "P201",    # Relación de parentesco
        ]
        especificas_num = []
    else:
        raise ValueError(f"Grupo inválido: {grupo}")
    
    # ===================================================================
    # VARIABLES DE LECTURA EXCLUIDAS (FASE 2)
    # ===================================================================
    excluded_lectura = [
        # Adultos
        "P311", "P315_1", "P315_2", "P315_3", "P315_4", "P315_5", "P315_6",
        "P409", "P410", "P411", "P412",
        
        # Niños/Jóvenes
        "P505", "P506", "P507", "P508", "P509", "P510",
        "P513", "P513A",
        "P514_1", "P514_2", "P514_3",
        "P515", "P516", "P517",
        
        # Derivadas
        "lectura_score", "habito_3clases"
    ]
    
    # Combinar todas las features
    cat_cols = demograficas_cat + socioculturales_cat + especificas_cat
    num_cols = demograficas_num + socioculturales_num + especificas_num
    
    print(f"   ✅ Categóricas: {len(cat_cols)}")
    print(f"   ✅ Numéricas:   {len(num_cols)}")
    print(f"   🚫 Excluidas (lectura): {len(excluded_lectura)}")
    
    return {
        "cat_cols": cat_cols,
        "num_cols": num_cols,
        "excluded_lectura": excluded_lectura
    }


def create_target_from_lectura_vars(df: pd.DataFrame, grupo: str) -> pd.DataFrame:
    """
    Crea variable objetivo desde variables de lectura (que NO están en features).
    
    IMPORTANTE: El target se deriva de variables de lectura, pero estas
    NO se incluyen como features de entrada. Esto permite evaluar la
    capacidad predictiva de variables demográficas/socioculturales únicamente.
    
    Args:
        df: DataFrame con datos crudos
        grupo: 'adultos', 'jovenes', o 'ninos'
        
    Returns:
        DataFrame con columna 'comportamiento_lector'
    """
    print(f"\n🎯 Creando TARGET desde variables de lectura (NO incluidas en features)")
    
    df = df.copy()
    
    # Estrategia: Usar P517 si disponible (niños/jóvenes)
    if "P517" in df.columns and grupo in ["jovenes", "ninos"]:
        print(f"   Usando P517 (frecuencia actividades de lectura)")
        mapping = {
            1.0: 0,  # Nunca → No lector
            2.0: 1,  # De vez en cuando → Lector medio
            3.0: 1,  # 1-2 veces por semana → Lector medio
            4.0: 2,  # 3-4 veces por semana → Lector frecuente
            5.0: 2   # Todos los días → Lector frecuente
        }
        df["comportamiento_lector"] = df["P517"].map(mapping)
        
        valid_count = df["comportamiento_lector"].notna().sum()
        if valid_count > len(df) * 0.1:
            print(f"   ✅ Target creado desde P517: {valid_count:,} registros válidos")
            return df
        else:
            df.drop(columns=["comportamiento_lector"], inplace=True)
    
    # Estrategia alternativa: Derivar desde múltiples variables
    print(f"   Derivando score desde variables de lectura múltiples...")
    df['lectura_score'] = 0.0
    
    # Variables según grupo
    if grupo == "adultos":
        lectura_vars = ['P315_1', 'P315_2', 'P315_3', 'P315_4', 'P315_5', 'P315_6']
        frecuencia_var = 'P409'
    else:
        lectura_vars = ['P514_1', 'P514_2', 'P514_3']
        frecuencia_var = 'P515'
    
    # Sumar tipos de lectura
    for var in lectura_vars:
        if var in df.columns:
            mask = df[var].notna()
            df.loc[mask, 'lectura_score'] += df.loc[mask, var].fillna(0) * 0.15
    
    # Ajustar por frecuencia
    if frecuencia_var in df.columns:
        mask = df[frecuencia_var].notna()
        df.loc[mask, 'lectura_score'] += (df.loc[mask, frecuencia_var].fillna(1) - 1) / 5 * 0.4
    
    # Ajustar por P311 (adultos)
    if 'P311' in df.columns:
        mask = df['P311'].notna()
        df.loc[mask & (df['P311'] == 0), 'lectura_score'] = 0.0
    
    # Clip al rango [0, 1]
    df['lectura_score'] = np.clip(df['lectura_score'], 0, 1)
    
    # Convertir a categorías
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
    
    print(f"   ✅ Target derivado correctamente\n")
    return df


def split_data(
    X: pd.DataFrame, 
    y: pd.Series,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Split estratificado 70/15/15."""
    print(f"\n✂️  Separando datos (70/15/15)...")
    print(f"   Total: {len(X):,} registros")
    
    train_size = 1.0 - test_size - val_size
    
    # Primera división
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Segunda división
    val_size_adjusted = val_size / (train_size + val_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state, stratify=y_temp
    )
    
    print(f"   Train: {len(X_train):6,} ({len(X_train)/len(X)*100:5.2f}%)")
    print(f"   Val:   {len(X_val):6,} ({len(X_val)/len(X)*100:5.2f}%)")
    print(f"   Test:  {len(X_test):6,} ({len(X_test)/len(X)*100:5.2f}%)")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def preprocess_features(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    cat_cols: List[str],
    num_cols: List[str]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, ColumnTransformer, List[str]]:
    """Preprocessing: OneHotEncoding + StandardScaler."""
    print(f"\n🔧 Aplicando preprocessing...")
    
    # Filtrar columnas existentes
    cat_cols_exist = [c for c in cat_cols if c in X_train.columns]
    num_cols_exist = [c for c in num_cols if c in X_train.columns]
    
    print(f"   Categóricas: {len(cat_cols_exist)}")
    print(f"   Numéricas:   {len(num_cols_exist)}")
    
    # Crear pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols_exist),
            ('num', StandardScaler(), num_cols_exist)
        ],
        remainder='drop'
    )
    
    # Fit en train, transform en todos
    print(f"   Entrenando preprocessor en train set...")
    X_train_proc = preprocessor.fit_transform(X_train)
    X_val_proc = preprocessor.transform(X_val)
    X_test_proc = preprocessor.transform(X_test)
    
    # Feature names
    feature_names = []
    if cat_cols_exist:
        cat_features = preprocessor.named_transformers_['cat'].get_feature_names_out(cat_cols_exist)
        feature_names.extend(cat_features)
    if num_cols_exist:
        feature_names.extend(num_cols_exist)
    
    print(f"   ✅ Features procesadas: {X_train_proc.shape[1]}")
    
    return X_train_proc, X_val_proc, X_test_proc, preprocessor, feature_names


def run_etl_fase2(grupo: str, data_dir: str, output_dir: str):
    """
    Ejecuta pipeline ETL completo para FASE 2 (solo demográficas).
    
    Args:
        grupo: 'adultos', 'jovenes', o 'ninos'
        data_dir: Directorio con datos brutos
        output_dir: Directorio de salida
    """
    print("="*70)
    print(f"🚀 ETL FASE 2 - SOLO DEMOGRÁFICAS: {grupo.upper()}")
    print("="*70)
    
    # Crear directorio de salida
    output_path = Path(output_dir) / grupo
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 1. Cargar datos brutos
    print(f"\n📂 Cargando datos brutos...")
    data_file = Path(data_dir) / f"{grupo}.csv"
    
    if not data_file.exists():
        raise FileNotFoundError(f"No se encontró: {data_file}")
    
    df = pd.read_csv(data_file, low_memory=False)
    print(f"   ✅ Cargados: {len(df):,} registros, {len(df.columns)} columnas")
    
    # 2. Seleccionar features demográficas
    feature_config = select_features_demographic_only(grupo)
    cat_cols = feature_config['cat_cols']
    num_cols = feature_config['num_cols']
    excluded = feature_config['excluded_lectura']
    
    # 3. Crear target (desde variables de lectura, pero NO en features)
    df = create_target_from_lectura_vars(df, grupo)
    
    # 4. Filtrar registros válidos
    print(f"\n🔍 Filtrando registros válidos...")
    initial_count = len(df)
    df = df[df['comportamiento_lector'].notna()].copy()
    final_count = len(df)
    print(f"   Inicial: {initial_count:,}")
    print(f"   Final:   {final_count:,}")
    print(f"   Descartados: {initial_count - final_count:,} ({(initial_count-final_count)/initial_count*100:.2f}%)")
    
    # 5. Separar features y target
    all_feature_cols = cat_cols + num_cols
    available_cols = [c for c in all_feature_cols if c in df.columns]
    
    X = df[available_cols].copy()
    y = df['comportamiento_lector'].copy()
    
    print(f"\n📊 Dataset final:")
    print(f"   Features: {X.shape[1]} ({len([c for c in cat_cols if c in X.columns])} cat, {len([c for c in num_cols if c in X.columns])} num)")
    print(f"   Target: {len(y):,} registros")
    
    # 6. Split 70/15/15
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    
    # 7. Preprocessing
    X_train_proc, X_val_proc, X_test_proc, preprocessor, feature_names = preprocess_features(
        X_train, X_val, X_test, cat_cols, num_cols
    )
    
    # 8. Guardar datasets procesados
    print(f"\n💾 Guardando datasets procesados...")
    
    # Train
    train_df = pd.DataFrame(X_train_proc, columns=feature_names)
    train_df['target'] = y_train.values
    train_df.to_csv(output_path / f'{grupo}_train.csv', index=False)
    print(f"   ✅ {grupo}_train.csv: {len(train_df):,} registros")
    
    # Val
    val_df = pd.DataFrame(X_val_proc, columns=feature_names)
    val_df['target'] = y_val.values
    val_df.to_csv(output_path / f'{grupo}_val.csv', index=False)
    print(f"   ✅ {grupo}_val.csv: {len(val_df):,} registros")
    
    # Test
    test_df = pd.DataFrame(X_test_proc, columns=feature_names)
    test_df['target'] = y_test.values
    test_df.to_csv(output_path / f'{grupo}_test.csv', index=False)
    print(f"   ✅ {grupo}_test.csv: {len(test_df):,} registros")
    
    # 9. Guardar preprocessor y config
    joblib.dump(preprocessor, output_path / f'{grupo}_preprocessor.pkl')
    print(f"   ✅ {grupo}_preprocessor.pkl")
    
    config = {
        'grupo': grupo,
        'fase': 2,
        'descripcion': 'Solo variables demográficas y socioculturales',
        'n_samples': len(df),
        'n_features': X_train_proc.shape[1],
        'cat_features': [c for c in cat_cols if c in X.columns],
        'num_features': [c for c in num_cols if c in X.columns],
        'excluded_lectura': excluded,
        'target_distribution': {
            'train': y_train.value_counts().to_dict(),
            'val': y_val.value_counts().to_dict(),
            'test': y_test.value_counts().to_dict()
        },
        'feature_names': feature_names
    }
    
    with open(output_path / f'{grupo}_config.json', 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4, ensure_ascii=False)
    print(f"   ✅ {grupo}_config.json")
    
    print(f"\n✅ ETL FASE 2 completado: {output_path}")
    print("="*70)


def main():
    """Ejecutar ETL para los 3 grupos."""
    # Configuración
    project_root = Path(__file__).parent.parent.parent
    data_dir = project_root / "data" / "unprocessed"
    output_dir = project_root / "data" / "processed_fase2"
    
    print("\n" + "="*70)
    print("🚀 PIPELINE ETL FASE 2 - SOLO DEMOGRÁFICAS Y SOCIOCULTURALES")
    print("="*70)
    print(f"\nObjetivo: Evaluar capacidad predictiva de Deep Learning")
    print(f"          usando ÚNICAMENTE datos censales (sin encuestas de lectura)")
    print(f"\nDatos brutos: {data_dir}")
    print(f"Salida: {output_dir}")
    
    # Ejecutar para cada grupo
    for grupo in ['ninos', 'jovenes', 'adultos']:
        try:
            run_etl_fase2(grupo, str(data_dir), str(output_dir))
        except Exception as e:
            print(f"\n❌ Error procesando {grupo}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*70)
    print("✅ PIPELINE FASE 2 COMPLETADO")
    print("="*70)
    print(f"\n📁 Datasets generados en: {output_dir}")
    print("\nPróximo paso: Entrenar modelos con:")
    print("   python training/train_fase2_ninos.py")
    print("   python training/train_fase2_jovenes.py")
    print("   python training/train_fase2_adultos.py")


if __name__ == "__main__":
    main()
