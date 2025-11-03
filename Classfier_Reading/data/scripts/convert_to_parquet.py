"""
Convierte CSVs procesados a Parquet para entrenamiento rápido.
"""
import pandas as pd
from pathlib import Path


def convert_to_parquet(group: str):
    """Convierte train/val/test.csv a .parquet"""
    data_path = Path(f"data/processed/{group}")
    
    for split in ['train', 'val', 'test']:
        csv_file = data_path / f"{group}_{split}.csv"
        parquet_file = data_path / f"{split}.parquet"
        
        if csv_file.exists():
            print(f"Convirtiendo {csv_file.name} -> {parquet_file.name}")
            df = pd.read_csv(csv_file)
            df.to_parquet(parquet_file, index=False)
            print(f"  ✅ {len(df)} registros guardados")
        else:
            print(f"  ⚠️  No existe: {csv_file}")


if __name__ == "__main__":
    for group in ['adultos', 'jovenes', 'ninos']:
        print(f"\n{'='*50}")
        print(f"Grupo: {group}")
        print(f"{'='*50}")
        convert_to_parquet(group)
    
    print("\n✅ Conversión completada")
