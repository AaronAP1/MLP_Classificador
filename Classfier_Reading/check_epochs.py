"""Extraer épocas de entrenamiento de logs"""
from pathlib import Path
import re

def extract_epochs(log_path):
    """Extrae información de épocas del log"""
    if not log_path.exists():
        return None
    
    with open(log_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Buscar "Early stopping en epoch X"
    early_stop = re.search(r'Early stopping en epoch (\d+)', content)
    if early_stop:
        return int(early_stop.group(1))
    
    # Buscar "best_epoch = X" (TabNet)
    best_epoch = re.search(r'best_epoch = (\d+)', content)
    if best_epoch:
        return int(best_epoch.group(1))
    
    return None

def main():
    results_dir = Path('results/baseline')
    
    print("\n" + "="*70)
    print("ÉPOCAS DE ENTRENAMIENTO POR MODELO")
    print("="*70 + "\n")
    
    models = [
        ("adultos", "mlp_ma"),
        ("adultos", "tabnet"),
        ("adultos", "ft_transformer"),
        ("jovenes", "mlp_ma"),
        ("jovenes", "tabnet"),
        ("jovenes", "ft_transformer"),
        ("ninos", "mlp_ma"),
        ("ninos", "tabnet"),
        ("ninos", "ft_transformer"),
    ]
    
    for grupo, modelo in models:
        model_name = f"{grupo}_{modelo}"
        log_path = results_dir / model_name / "training.log"
        
        epochs = extract_epochs(log_path)
        
        if epochs:
            print(f"✅ {grupo.upper():10s} | {modelo.upper():15s} | Épocas: {epochs:3d}")
        else:
            print(f"⚠️  {grupo.upper():10s} | {modelo.upper():15s} | No encontrado")
    
    print("\n" + "="*70 + "\n")

if __name__ == "__main__":
    main()
