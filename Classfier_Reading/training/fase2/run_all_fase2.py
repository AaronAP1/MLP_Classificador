"""
EJECUTOR MAESTRO - FASE 2 (SOLO DEMOGRÁFICAS)
=============================================

Ejecuta secuencialmente el entrenamiento de los 3 grupos etarios
con variables demográficas únicamente.

TIEMPO ESTIMADO: 3-4 horas total
- Niños: ~1 hora (7,270 train × 3 modelos)
- Jóvenes: ~1.5 horas (9,814 train × 3 modelos)
- Adultos: ~2 horas (28,391 train × 3 modelos)

RESULTADOS ESPERADOS:
- Accuracy: 60-75% (vs 100% en Fase 1)
- Deep Learning mejora 6-9% sobre modelos clásicos
- Demuestra valor de DL en problemas realistas

Universidad Peruana Unión - 2025
"""

import sys
from pathlib import Path
import subprocess
from datetime import datetime
import time
import json

# Configurar UTF-8 para Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Paths
project_root = Path(__file__).parent.parent.parent
training_scripts = {
    'ninos': project_root / 'Classfier_Reading' / 'training' / 'fase2' / 'train_ninos.py',
    'jovenes': project_root / 'Classfier_Reading' / 'training' / 'fase2' / 'train_jovenes.py',
    'adultos': project_root / 'Classfier_Reading' / 'training' / 'fase2' / 'train_adultos.py'
}

output_dir = project_root / 'Classfier_Reading' / 'outputs' / 'fase2'


def print_header():
    """Imprimir encabezado del pipeline."""
    print("\n" + "="*80)
    print("🚀 PIPELINE DE ENTRENAMIENTO - FASE 2 (SOLO DEMOGRÁFICAS)")
    print("="*80)
    print(f"📅 Fecha inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🐍 Python: {sys.executable}")
    print(f"\n📁 Scripts a ejecutar:")
    for grupo, script in training_scripts.items():
        print(f"   {grupo.upper()}: {script}")
    print(f"\n💾 Resultados se guardarán en: {output_dir}")
    print("="*80 + "\n")


def run_training(grupo: str, script_path: Path) -> dict:
    """
    Ejecutar script de entrenamiento de un grupo.
    
    Args:
        grupo: Nombre del grupo (ninos/jovenes/adultos)
        script_path: Path al script de entrenamiento
    
    Returns:
        dict: Información de la ejecución
    """
    print("\n" + "🔹"*40)
    print(f"📌 INICIANDO: {grupo.upper()}")
    print("🔹"*40)
    
    start_time = time.time()
    
    try:
        # Ejecutar script
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            encoding='utf-8'
        )
        
        elapsed_time = time.time() - start_time
        
        # Verificar si hubo error
        if result.returncode != 0:
            print(f"\n❌ ERROR en {grupo}:")
            print(result.stderr)
            return {
                'grupo': grupo,
                'status': 'error',
                'error': result.stderr,
                'time_minutes': elapsed_time / 60
            }
        
        # Éxito
        print(f"\n✅ {grupo.upper()} completado en {elapsed_time/60:.2f} minutos")
        
        # Intentar cargar resultados
        results_file = output_dir / grupo / f'results_fase2_{grupo}.json'
        if results_file.exists():
            with open(results_file, 'r', encoding='utf-8') as f:
                results = json.load(f)
            
            print(f"\n📊 Resultados {grupo.upper()}:")
            for model_result in results.get('results', []):
                print(f"   {model_result['model']}: {model_result['test_acc']*100:.2f}% acc")
        
        return {
            'grupo': grupo,
            'status': 'success',
            'time_minutes': elapsed_time / 60
        }
        
    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"\n❌ EXCEPCIÓN en {grupo}: {e}")
        return {
            'grupo': grupo,
            'status': 'exception',
            'error': str(e),
            'time_minutes': elapsed_time / 60
        }


def print_summary(execution_results: list, total_time: float):
    """Imprimir resumen final."""
    print("\n" + "="*80)
    print("📋 RESUMEN FINAL - FASE 2")
    print("="*80)
    
    for result in execution_results:
        status_icon = "✅" if result['status'] == 'success' else "❌"
        print(f"{status_icon} {result['grupo'].upper()}: {result['status']} ({result['time_minutes']:.2f} min)")
    
    print(f"\n⏱️  Tiempo total: {total_time/60:.2f} minutos ({total_time/3600:.2f} horas)")
    
    # Contar éxitos
    success_count = sum(1 for r in execution_results if r['status'] == 'success')
    print(f"\n🎯 Completados exitosamente: {success_count}/3 grupos")
    
    if success_count == 3:
        print("\n🎉 ¡TODOS LOS ENTRENAMIENTOS COMPLETADOS!")
        print("\n📊 Próximos pasos:")
        print("   1. Revisar resultados en outputs/fase2/")
        print("   2. Comparar Fase 1 vs Fase 2")
        print("   3. Generar visualizaciones para tesis")
    else:
        print("\n⚠️  Algunos entrenamientos fallaron. Revisar logs arriba.")
    
    print("="*80 + "\n")


def main():
    """Pipeline principal."""
    print_header()
    
    # Preguntar confirmación
    response = input("¿Iniciar entrenamiento de TODOS los grupos? (y/n): ")
    if response.lower() not in ['y', 'yes', 's', 'si', 'sí']:
        print("❌ Operación cancelada.")
        return
    
    # Ejecutar entrenamientos
    execution_results = []
    total_start = time.time()
    
    for grupo, script_path in training_scripts.items():
        if not script_path.exists():
            print(f"❌ Script no encontrado: {script_path}")
            execution_results.append({
                'grupo': grupo,
                'status': 'not_found',
                'time_minutes': 0
            })
            continue
        
        result = run_training(grupo, script_path)
        execution_results.append(result)
        
        # Pausa breve entre grupos
        if grupo != 'adultos':  # No pausar después del último
            print("\n⏸️  Pausa 10 segundos antes del siguiente grupo...")
            time.sleep(10)
    
    total_time = time.time() - total_start
    
    # Resumen final
    print_summary(execution_results, total_time)
    
    # Guardar log de ejecución
    log_file = output_dir / 'execution_log.json'
    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'total_time_minutes': total_time / 60,
            'results': execution_results
        }, f, indent=4, ensure_ascii=False)
    
    print(f"📁 Log guardado en: {log_file}")


if __name__ == "__main__":
    main()
