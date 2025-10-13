"""
Script principal para ejecutar el pipeline ETL.
"""

import sys
from pathlib import Path

# Agregar el directorio raíz al path
root_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root_dir))

from src.data.etl import run_etl_pipeline

def main():
    # Configuración base
    base_dir = root_dir / "data"
    
    print("\n" + "="*80)
    print("🚀 ETL PIPELINE - CLASIFICADOR DE COMPORTAMIENTO LECTOR")
    print("="*80 + "\n")
    
    print("📋 Grupos a procesar:")
    grupos = ["adultos", "jovenes", "ninos"]
    for i, grupo in enumerate(grupos, 1):
        print(f"   {i}. {grupo.capitalize()}")
    
    print("\n" + "="*80 + "\n")
    
    # Ejecutar para cada grupo
    for i, grupo in enumerate(grupos, 1):
        print(f"\n{'#'*80}")
        print(f"# PROCESANDO GRUPO {i}/{len(grupos)}: {grupo.upper()}")
        print(f"{'#'*80}\n")
        
        try:
            run_etl_pipeline(
                grupo=grupo,
                raw_csv_path=str(base_dir / "unprocessed" / f"{grupo}.csv"),
                metadata_dir=str(base_dir / "metadata"),
                output_dir=str(base_dir / "processed" / grupo),
                test_size=0.15,
                val_size=0.15,
                random_state=42
            )
        except Exception as e:
            print(f"\n❌ ERROR procesando {grupo}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n" + "="*80)
    print("✅ ETL PIPELINE COMPLETADO PARA TODOS LOS GRUPOS")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
