"""
Script para inspeccionar la estructura de los datos sin procesar.
Genera información sobre columnas, tipos de datos y tamaños.
"""

import pandas as pd
import json
from pathlib import Path

def inspect_dataset(file_path: Path, name: str):
    """Inspecciona un archivo CSV y retorna información básica."""
    print(f"\n{'='*80}")
    print(f"📊 INSPECCIONANDO: {name}")
    print(f"{'='*80}")
    
    # Cargar datos
    df = pd.read_csv(file_path)
    
    # Información básica
    print(f"\n✅ Total de registros: {len(df):,}")
    print(f"✅ Total de columnas: {len(df.columns)}")
    print(f"\n📋 Primeras 5 columnas: {list(df.columns[:5])}")
    print(f"📋 Últimas 5 columnas: {list(df.columns[-5:])}")
    
    # Verificar columna objetivo
    target_candidates = ['target', 'TARGET', 'label', 'LABEL', 'clase', 'CLASE']
    target_col = None
    for col in target_candidates:
        if col in df.columns:
            target_col = col
            break
    
    if target_col:
        print(f"\n🎯 Columna objetivo encontrada: '{target_col}'")
        print(f"   Distribución de clases:")
        print(df[target_col].value_counts())
    else:
        print(f"\n⚠️  No se encontró columna objetivo obvia")
        print(f"   Columnas disponibles: {list(df.columns)}")
    
    # Tipos de datos
    print(f"\n📊 Tipos de datos:")
    types_count = df.dtypes.value_counts()
    for dtype, count in types_count.items():
        print(f"   - {dtype}: {count} columnas")
    
    # Valores nulos
    null_cols = df.isnull().sum()
    null_cols = null_cols[null_cols > 0]
    if len(null_cols) > 0:
        print(f"\n⚠️  Columnas con valores nulos: {len(null_cols)}")
        print(f"   Top 5 con más nulos:")
        for col, count in null_cols.nlargest(5).items():
            print(f"   - {col}: {count} ({count/len(df)*100:.2f}%)")
    else:
        print(f"\n✅ No hay valores nulos")
    
    # Guardar metadata
    metadata = {
        'dataset': name,
        'total_rows': int(len(df)),
        'total_columns': int(len(df.columns)),
        'columns': list(df.columns),
        'dtypes': {str(k): int(v) for k, v in df.dtypes.value_counts().items()},
        'target_column': target_col,
        'null_counts': {k: int(v) for k, v in null_cols.items()} if len(null_cols) > 0 else {}
    }
    
    return metadata

def main():
    # Rutas
    base_path = Path(__file__).parent.parent
    unprocessed_path = base_path / 'unprocessed'
    
    # Inspeccionar cada dataset
    datasets = {
        'adultos': unprocessed_path / 'adultos.csv',
        'jovenes': unprocessed_path / 'jovenes.csv',
        'ninos': unprocessed_path / 'ninos.csv'
    }
    
    all_metadata = {}
    
    for name, file_path in datasets.items():
        if file_path.exists():
            metadata = inspect_dataset(file_path, name.upper())
            all_metadata[name] = metadata
        else:
            print(f"\n❌ No se encontró: {file_path}")
    
    # Guardar metadata
    output_file = base_path / 'metadata' / 'datasets_info.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_metadata, f, indent=2, ensure_ascii=False)
    
    print(f"\n\n{'='*80}")
    print(f"✅ Metadata guardada en: {output_file}")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()
