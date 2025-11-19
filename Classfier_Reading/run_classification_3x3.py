"""
Script rápido para ejecutar la clasificación 3x3 completa
"""

from classify_data_3x3 import Classifier3x3

def main():
    print("\n🚀 Ejecutando Clasificación 3x3\n")
    print("📋 Configuración:")
    print("   - 3 Grupos: Niños, Jóvenes, Adultos")
    print("   - 3 Modelos: FT-Transformer, TabNet, MLP_MA")
    print("   - Total: 9 clasificadores\n")
    
    # Crear clasificador
    classifier = Classifier3x3()
    
    # Ejecutar todas las combinaciones
    results = classifier.classify_all_combinations(
        data_dir='data/processed_fase2',
        output_dir='outputs/classified_data_3x3'
    )
    
    print("\n✅ Proceso completado!")
    print(f"📁 Los archivos clasificados se guardaron en: outputs/classified_data_3x3/")
    print(f"\nEstructura generada:")
    print(f"  outputs/classified_data_3x3/")
    print(f"    ├── ninos/")
    print(f"    │   ├── FT-Transformer/")
    print(f"    │   │   ├── clasificado_completo_*.csv")
    print(f"    │   │   ├── no_lector_*.csv")
    print(f"    │   │   ├── lector_medio_*.csv")
    print(f"    │   │   └── lector_frecuente_*.csv")
    print(f"    │   ├── TabNet/...")
    print(f"    │   └── MLP_MA/...")
    print(f"    ├── jovenes/...")
    print(f"    └── adultos/...")
    print(f"\n💡 Cada combinación genera:")
    print(f"   1. CSV completo con todas las predicciones")
    print(f"   2. CSVs separados por clase (No Lector, Lector Medio, Lector Frecuente)")

if __name__ == '__main__':
    main()
