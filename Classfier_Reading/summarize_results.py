"""Resumen de resultados de entrenamiento baseline"""
import json
from pathlib import Path
import pandas as pd

def extract_metrics_from_log(log_path):
    """Extrae métricas del archivo de log"""
    if not log_path.exists():
        return None
    
    with open(log_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Buscar accuracy test
    if "Accuracy:" in content:
        for line in content.split('\n'):
            if "Accuracy:" in line:
                acc = float(line.split("Accuracy:")[-1].strip())
                return acc
    return None

def main():
    results_base = Path("results/baseline")
    
    configs = [
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
    
    results = []
    
    print("\n" + "="*80)
    print("RESUMEN DE RESULTADOS - ENTRENAMIENTO BASELINE")
    print("="*80 + "\n")
    
    for grupo, modelo in configs:
        model_name = f"{grupo}_{modelo}"
        model_dir = results_base / model_name
        log_path = model_dir / "training.log"
        
        if log_path.exists():
            acc = extract_metrics_from_log(log_path)
            status = "✅" if acc else "⚠️"
            acc_str = f"{acc:.4f}" if acc else "N/A"
        else:
            status = "❌"
            acc_str = "No entrenado"
        
        results.append({
            "Grupo": grupo.capitalize(),
            "Modelo": modelo.upper().replace("_", "-"),
            "Status": status,
            "Test Accuracy": acc_str
        })
        
        print(f"{status} {grupo.upper():10s} | {modelo.upper():15s} | Acc: {acc_str}")
    
    # Crear tabla resumen
    df = pd.DataFrame(results)
    
    print("\n" + "="*80)
    print("TABLA RESUMEN")
    print("="*80 + "\n")
    print(df.to_string(index=False))
    
    # Guardar CSV
    output_path = Path("results/baseline_summary.csv")
    df.to_csv(output_path, index=False)
    print(f"\n✅ Resumen guardado en: {output_path}")

if __name__ == "__main__":
    main()
