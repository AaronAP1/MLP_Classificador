"""
Script Automatizado de Entrenamiento 3×3
Entrena 9 modelos para clasificación de lectores:

3 Grupos de Edad:
- Niños (0-11 años)
- Jóvenes (12-17 años)
- Adultos (18+ años)

3 Modelos por grupo:
- TabNet
- MLP con Multihead Attention
- FT-Transformer

Universidad Peruana Unión - 2025
"""

import sys
from pathlib import Path
import subprocess
import json
import time
from datetime import datetime

# Configurar UTF-8 para Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Paths del proyecto
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# ===================================================================
# CONFIGURACIÓN
# ===================================================================

# Grupos de edad a entrenar
GRUPOS = ['ninos', 'jovenes', 'adultos']

# Scripts de entrenamiento
TRAINING_SCRIPTS = {
    'ninos': project_root / 'training' / 'fase2' / 'train_ninos_simple.py',
    'jovenes': project_root / 'training' / 'fase2' / 'train_jovenes_simple.py',
    'adultos': project_root / 'training' / 'fase2' / 'train_adultos_simple.py'
}

# Directorio de salida
OUTPUT_DIR = project_root / 'outputs' / 'fase2'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Python executable
PYTHON_EXE = sys.executable


# ===================================================================
# FUNCIONES
# ===================================================================

def print_banner(text, char="="):
    """Imprimir banner decorativo."""
    print("\n" + char * 80)
    print(f"  {text}")
    print(char * 80 + "\n")


def train_group(grupo):
    """Entrenar los 3 modelos para un grupo específico."""
    print_banner(f"🚀 ENTRENANDO GRUPO: {grupo.upper()}", "=")
    
    script_path = TRAINING_SCRIPTS[grupo]
    
    if not script_path.exists():
        print(f"❌ Script no encontrado: {script_path}")
        return {
            'grupo': grupo,
            'status': 'error',
            'error': f'Script no encontrado: {script_path}'
        }
    
    print(f"📄 Ejecutando: {script_path.name}")
    print(f"⏰ Inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    start_time = time.time()
    
    try:
        # Ejecutar script de entrenamiento
        result = subprocess.run(
            [PYTHON_EXE, str(script_path)],
            cwd=str(project_root),
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace'
        )
        
        elapsed = time.time() - start_time
        
        # Mostrar salida
        print("\n" + "-" * 80)
        print("SALIDA DEL ENTRENAMIENTO:")
        print("-" * 80)
        print(result.stdout)
        
        if result.stderr:
            print("\n" + "-" * 80)
            print("ERRORES/WARNINGS:")
            print("-" * 80)
            print(result.stderr)
        
        if result.returncode == 0:
            print(f"\n✅ Grupo '{grupo}' completado exitosamente")
            print(f"⏱️  Tiempo total: {elapsed/60:.2f} minutos")
            
            return {
                'grupo': grupo,
                'status': 'success',
                'elapsed_min': elapsed / 60,
                'timestamp': datetime.now().isoformat()
            }
        else:
            print(f"\n❌ Error en grupo '{grupo}' (código {result.returncode})")
            
            return {
                'grupo': grupo,
                'status': 'error',
                'exit_code': result.returncode,
                'elapsed_min': elapsed / 60,
                'error': result.stderr[-500:] if result.stderr else 'Error desconocido'
            }
    
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n❌ Excepción durante entrenamiento: {e}")
        
        return {
            'grupo': grupo,
            'status': 'exception',
            'elapsed_min': elapsed / 60,
            'error': str(e)
        }


def verify_outputs():
    """Verificar que los modelos se hayan guardado correctamente."""
    print_banner("🔍 VERIFICANDO SALIDAS", "=")
    
    verification = {}
    
    for grupo in GRUPOS:
        grupo_path = OUTPUT_DIR / grupo / 'models'
        
        if not grupo_path.exists():
            verification[grupo] = {
                'exists': False,
                'models': []
            }
            print(f"⚠️  {grupo}: Directorio no encontrado - {grupo_path}")
            continue
        
        # Verificar cada modelo
        models_found = []
        
        # TabNet
        tabnet_path = grupo_path / 'tabnet'
        if tabnet_path.exists() and list(tabnet_path.glob('*')):
            models_found.append('TabNet')
            print(f"✅ {grupo}/TabNet: {len(list(tabnet_path.glob('*')))} archivos")
        
        # MLP+Attention
        mlp_path = grupo_path / 'mlp_attention'
        if mlp_path.exists() and list(mlp_path.glob('*.pth')):
            models_found.append('MLP+Attention')
            print(f"✅ {grupo}/MLP+Attention: {len(list(mlp_path.glob('*.pth')))} archivos")
        
        # FT-Transformer (placeholder)
        ft_path = grupo_path / 'ft_transformer'
        if ft_path.exists() and list(ft_path.glob('*')):
            models_found.append('FT-Transformer')
            print(f"✅ {grupo}/FT-Transformer: {len(list(ft_path.glob('*')))} archivos")
        
        verification[grupo] = {
            'exists': True,
            'models': models_found,
            'total': len(models_found)
        }
        
        if len(models_found) == 0:
            print(f"⚠️  {grupo}: Ningún modelo encontrado en {grupo_path}")
    
    return verification


def main():
    """Pipeline principal de entrenamiento."""
    print_banner("🎯 ENTRENAMIENTO AUTOMATIZADO 3×3", "█")
    print(f"📅 Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🐍 Python: {sys.version.split()[0]}")
    print(f"📁 Directorio: {project_root}")
    print(f"💾 Salida: {OUTPUT_DIR}")
    
    print("\n📋 PLAN DE ENTRENAMIENTO:")
    for i, grupo in enumerate(GRUPOS, 1):
        print(f"   {i}. {grupo.capitalize()}: 3 modelos (TabNet, MLP+Attention, FT-Transformer)")
    
    print("\n⚡ Total: 9 modelos a entrenar")
    print("\n▶️  Iniciando entrenamiento automáticamente...\n")
    
    # Registrar inicio
    start_time_total = time.time()
    results = []
    
    # Entrenar cada grupo
    for i, grupo in enumerate(GRUPOS, 1):
        print_banner(f"GRUPO {i}/{len(GRUPOS)}: {grupo.upper()}", "█")
        
        result = train_group(grupo)
        results.append(result)
        
        # Pausa entre grupos (excepto el último)
        if i < len(GRUPOS):
            print("\n⏸️  Pausando 5 segundos antes del siguiente grupo...")
            time.sleep(5)
    
    elapsed_total = time.time() - start_time_total
    
    # Verificar salidas
    verification = verify_outputs()
    
    # Resumen final
    print_banner("📊 RESUMEN FINAL", "█")
    
    successful = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] in ['error', 'exception']]
    
    print(f"✅ Exitosos: {len(successful)}/{len(GRUPOS)}")
    print(f"❌ Fallidos: {len(failed)}/{len(GRUPOS)}")
    print(f"⏱️  Tiempo total: {elapsed_total/60:.2f} minutos ({elapsed_total/3600:.2f} horas)")
    
    if successful:
        print("\n✅ GRUPOS EXITOSOS:")
        for r in successful:
            print(f"   - {r['grupo'].capitalize()}: {r['elapsed_min']:.2f} min")
    
    if failed:
        print("\n❌ GRUPOS FALLIDOS:")
        for r in failed:
            print(f"   - {r['grupo'].capitalize()}: {r.get('error', 'Error desconocido')[:100]}")
    
    print("\n📦 MODELOS GENERADOS:")
    total_models = 0
    for grupo, info in verification.items():
        if info['exists']:
            print(f"   {grupo.capitalize()}: {info['total']}/3 modelos")
            total_models += info['total']
        else:
            print(f"   {grupo.capitalize()}: 0/3 modelos (directorio no existe)")
    
    print(f"\n📊 Total de modelos: {total_models}/9")
    
    # Guardar resumen
    summary = {
        'timestamp': datetime.now().isoformat(),
        'total_time_min': elapsed_total / 60,
        'total_time_hours': elapsed_total / 3600,
        'groups_trained': len(GRUPOS),
        'successful': len(successful),
        'failed': len(failed),
        'results': results,
        'verification': verification,
        'models_generated': total_models
    }
    
    summary_path = OUTPUT_DIR / 'training_summary_3x3.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)
    
    print(f"\n💾 Resumen guardado en: {summary_path}")
    
    if total_models == 9:
        print_banner("🎉 ¡ENTRENAMIENTO COMPLETADO CON ÉXITO!", "█")
        print("✅ Los 9 modelos están listos para clasificación")
        print(f"📁 Modelos en: {OUTPUT_DIR}")
        print("\n💡 Siguiente paso: Ejecutar 'run_classification_3x3.py' para clasificar datos")
    elif total_models > 0:
        print_banner("⚠️  ENTRENAMIENTO PARCIALMENTE COMPLETADO", "█")
        print(f"⚠️  Solo {total_models}/9 modelos fueron generados")
        print("🔍 Revisa los errores arriba para más detalles")
    else:
        print_banner("❌ ENTRENAMIENTO FALLIDO", "█")
        print("❌ No se generó ningún modelo")
        print("🔍 Revisa los errores arriba para diagnóstico")
    
    return 0 if total_models == 9 else 1


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⚠️  Entrenamiento interrumpido por el usuario")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error crítico: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
