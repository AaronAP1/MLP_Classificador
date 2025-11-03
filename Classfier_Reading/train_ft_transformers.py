"""Entrenar solo los 3 FT-Transformer que fallaron"""
import subprocess
from pathlib import Path

ft_configs = [
    "configs/models/adultos_ft_transformer.yaml",
    "configs/models/jovenes_ft_transformer.yaml",
    "configs/models/ninos_ft_transformer.yaml"
]

print(f"\n{'='*70}")
print(f"ENTRENANDO 3 FT-TRANSFORMER")
print(f"{'='*70}\n")

for i, config_path in enumerate(ft_configs, 1):
    print(f"\n{'='*70}")
    print(f"MODELO {i}/3: {config_path}")
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
