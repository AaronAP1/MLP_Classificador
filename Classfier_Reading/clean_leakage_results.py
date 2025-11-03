"""Limpiar resultados con data leakage y preparar para re-entrenamiento limpio"""
from pathlib import Path
import shutil

def clean_baseline_results():
    """Limpia todos los resultados baseline para re-entrenar sin data leakage"""
    
    results_dir = Path('results/baseline')
    
    print("\n" + "="*70)
    print("LIMPIEZA DE RESULTADOS CON DATA LEAKAGE")
    print("="*70 + "\n")
    
    # Listar directorios a limpiar
    model_dirs = [d for d in results_dir.glob('*/') if d.name != '.gitkeep']
    
    print(f"Se encontraron {len(model_dirs)} directorios de modelos:\n")
    for d in sorted(model_dirs):
        files = list(d.glob('*'))
        print(f"  📁 {d.name}: {len(files)} archivos")
    
    print(f"\n{'='*70}")
    print("ACCIÓN: Eliminar TODOS los resultados para re-entrenamiento limpio")
    print("="*70 + "\n")
    
    confirm = input("¿Confirmar limpieza? (escribe 'SI' para confirmar): ")
    
    if confirm.upper() == 'SI':
        for model_dir in model_dirs:
            print(f"  🗑️  Eliminando {model_dir.name}...")
            shutil.rmtree(model_dir)
            model_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n✅ Limpieza completada. {len(model_dirs)} directorios vaciados.")
        print("\n📋 Siguiente paso: ejecutar train_all_baseline.py")
    else:
        print("\n❌ Limpieza cancelada.")

if __name__ == "__main__":
    clean_baseline_results()
