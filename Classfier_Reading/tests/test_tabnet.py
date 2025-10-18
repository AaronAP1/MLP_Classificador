"""
Script de prueba para TabNet Classifier
"""

import sys
from pathlib import Path

# Agregar el directorio raíz al path
root_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root_dir))

try:
    from src.models.tabnet_classifier import TabNetClassifier, create_tabnet_for_ninos
    
    print("\n" + "="*80)
    print("✅ TABNET CLASSIFIER - IMPORTACIÓN EXITOSA")
    print("="*80 + "\n")
    
    # Crear modelo
    print("📦 Creando modelo TabNet para NIÑOS...")
    model = create_tabnet_for_ninos(input_dim=131)
    
    # Mostrar resumen
    model.summary()
    
    print("✅ TabNet Classifier funcionando correctamente")
    print("🎯 Listo para entrenamiento con datos procesados")
    
except ImportError as e:
    print(f"\n❌ Error de importación: {e}")
    print("💡 Asegúrate de tener pytorch-tabnet instalado:")
    print("   pip install pytorch-tabnet")
 
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
