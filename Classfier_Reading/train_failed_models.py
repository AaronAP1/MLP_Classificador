"""Entrenar solo los 5 modelos que fallaron"""
import subprocess
from pathlib import Path

failed_configs = [
    "configs/models/adultos_ft_transformer.yaml",
    "configs/models/jovenes_mlp_ma.yaml",
    "configs/models/jovenes_tabnet.yaml",
    "configs/models/jovenes_ft_transformer.yaml",
    "configs/models/ninos_ft_transformer.yaml"
]

print(f"\n{'='*70}")
print(f"ENTRENANDO {len(failed_configs)} MODELOS QUE FALLARON")
print(f"{'='*70}\n")

for i, config_path in enumerate(failed_configs, 1):
    print(f"\n{'='*70}")
    print(f"MODELO {i}/{len(failed_configs)}: {config_path}")
    print(f"{'='*70}\n")
    
    try:
        result = subprocess.run(
            ["python", "experiments/train_single_model.py", "--config", config_path],
            check=True,
            capture_output=False,
            text=True
        )
        print(f"\n✅ Modelo {i} completado: {config_path}\n")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ ERROR en modelo {i}: {config_path}")
        print("Continuando con siguiente modelo...\n")
        continue

print(f"\n{'='*70}")
print(f"✅ ENTRENAMIENTO COMPLETADO")
print(f"{'='*70}\n")
