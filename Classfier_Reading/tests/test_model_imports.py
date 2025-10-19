"""
Script de prueba para verificar imports de modelos organizados por grupo etario.

Ejecutar desde la raíz del proyecto:
    python -m tests.test_model_imports
"""

import sys
from pathlib import Path

# Agregar directorio raíz al path
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

def test_ninos_imports():
    """Probar imports de modelos de Niños"""
    print("🧪 Probando imports de Niños...")
    
    try:
        from src.models.Ninos import TabNetClassifier, create_tabnet_for_ninos
        print("   ✅ TabNetClassifier importado correctamente")
        print("   ✅ create_tabnet_for_ninos importado correctamente")
        
        # Verificar que se puede instanciar
        model = create_tabnet_for_ninos(num_features=131)
        print(f"   ✅ Modelo creado: {type(model)}")
        
        return True
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


def test_jovenes_imports():
    """Probar imports de modelos de Jóvenes"""
    print("\n🧪 Probando imports de Jóvenes...")
    
    try:
        from src.models.Jovenes import (
            MLPWithAttention,
            MLPWithAttentionTrainer,
            create_mlp_attention_for_jovenes
        )
        print("   ✅ MLPWithAttention importado correctamente")
        print("   ✅ MLPWithAttentionTrainer importado correctamente")
        print("   ✅ create_mlp_attention_for_jovenes importado correctamente")
        
        # Verificar que se puede instanciar
        model, trainer = create_mlp_attention_for_jovenes(num_features=139)
        print(f"   ✅ Modelo creado: {type(model).__name__}")
        print(f"   ✅ Trainer creado: {type(trainer).__name__}")
        
        return True
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_adultos_imports():
    """Probar imports de modelos de Adultos"""
    print("\n🧪 Probando imports de Adultos...")
    
    try:
        from src.models.Adultos import (
            FTTransformerClassifier,
            FTTransformerTrainer,
            create_ft_transformer_for_adultos
        )
        print("   ✅ FTTransformerClassifier importado correctamente")
        print("   ✅ FTTransformerTrainer importado correctamente")
        print("   ✅ create_ft_transformer_for_adultos importado correctamente")
        
        # Verificar que se puede instanciar
        model, trainer = create_ft_transformer_for_adultos(num_features=71)
        print(f"   ✅ Modelo creado: {type(model).__name__}")
        print(f"   ✅ Trainer creado: {type(trainer).__name__}")
        
        return True
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_main_module_imports():
    """Probar imports desde módulo principal"""
    print("\n🧪 Probando imports desde módulo principal...")
    
    try:
        from src.models import (
            # Niños
            TabNetClassifier,
            create_tabnet_for_ninos,
            
            # Jóvenes
            MLPWithAttention,
            MLPWithAttentionTrainer,
            create_mlp_attention_for_jovenes,
            
            # Adultos
            FTTransformerClassifier,
            FTTransformerTrainer,
            create_ft_transformer_for_adultos
        )
        
        print("   ✅ Todos los imports desde src.models funcionan correctamente")
        
        return True
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Ejecutar todas las pruebas"""
    print("=" * 70)
    print("🧪 PRUEBAS DE IMPORTS - MODELOS POR GRUPO ETARIO")
    print("=" * 70)
    
    results = {
        "Niños": test_ninos_imports(),
        "Jóvenes": test_jovenes_imports(),
        "Adultos": test_adultos_imports(),
        "Módulo Principal": test_main_module_imports()
    }
    
    print("\n" + "=" * 70)
    print("📊 RESUMEN DE RESULTADOS")
    print("=" * 70)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:20s} : {status}")
    
    all_passed = all(results.values())
    
    print("\n" + "=" * 70)
    if all_passed:
        print("🎉 ¡TODAS LAS PRUEBAS PASARON! Estructura organizada correctamente.")
    else:
        print("⚠️  Algunas pruebas fallaron. Revisar imports.")
    print("=" * 70)
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
