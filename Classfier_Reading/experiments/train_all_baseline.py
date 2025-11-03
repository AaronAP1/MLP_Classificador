"""
Script para entrenar los 9 modelos baseline de forma secuencial.
"""
import subprocess
from pathlib import Path


def main():
    # Configs de los 9 modelos
    configs = [
        # Adultos
        "configs/models/adultos_mlp_ma.yaml",
        "configs/models/adultos_tabnet.yaml",
        "configs/models/adultos_ft_transformer.yaml",
        
        # Jóvenes
        "configs/models/jovenes_mlp_ma.yaml",
        "configs/models/jovenes_tabnet.yaml",
        "configs/models/jovenes_ft_transformer.yaml",
        
        # Niños
        "configs/models/ninos_mlp_ma.yaml",
        "configs/models/ninos_tabnet.yaml",
        "configs/models/ninos_ft_transformer.yaml",
    ]
    
    root = Path(__file__).parent.parent
    
    for i, config in enumerate(configs, 1):
        config_path = root / config
        
        print(f"\n{'='*70}")
        print(f"ENTRENANDO MODELO {i}/9: {config}")
        print(f"{'='*70}\n")
        
        # Ejecutar entrenamiento
        cmd = [
            "python",
            str(root / "experiments" / "train_single_model.py"),
            "--config",
            str(config_path)
        ]
        
        result = subprocess.run(cmd)
        
        if result.returncode != 0:
            print(f"\n❌ ERROR en modelo {i}: {config}")
            print("Continuando con siguiente modelo...\n")
        else:
            print(f"\n✅ Modelo {i} completado: {config}\n")
    
    print(f"\n{'='*70}")
    print("✅ ENTRENAMIENTO DE 9 MODELOS COMPLETADO")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
