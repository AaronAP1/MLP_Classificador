"""
Script para monitorear el progreso del entrenamiento en tiempo real
"""
import os
import time
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

def find_latest_run():
    """Encuentra el run más reciente"""
    models_dir = "outputs/models"
    if not os.path.exists(models_dir):
        return None
    
    runs = [d for d in os.listdir(models_dir) if d.startswith("20250703")]
    if not runs:
        return None
    
    latest_run = max(runs)
    return os.path.join(models_dir, latest_run)

def monitor_training():
    """Monitorea el entrenamiento en tiempo real"""
    print("🔍 MONITOREANDO ENTRENAMIENTO EN TIEMPO REAL")
    print("=" * 50)
    
    latest_run = find_latest_run()
    if not latest_run:
        print("❌ No se encontró ningún run activo")
        return
    
    metrics_file = os.path.join(latest_run, "metrics_by_epoch.csv")
    print(f"📊 Monitoreando: {metrics_file}")
    
    last_epoch = 0
    while True:
        try:
            if os.path.exists(metrics_file):
                df = pd.read_csv(metrics_file)
                if len(df) > last_epoch:
                    # Mostrar nuevas épocas
                    for i in range(last_epoch, len(df)):
                        row = df.iloc[i]
                        print(f"Época {int(row['epoch']):2d}: "
                              f"Loss={row['train_loss']:.4f} "
                              f"Val_Acc={row['val_accuracy']:.4f} "
                              f"Val_F1={row['val_f1']:.4f}")
                    
                    last_epoch = len(df)
                    
                    # Mostrar resumen cada 10 épocas
                    if last_epoch % 10 == 0 and last_epoch > 0:
                        print(f"\n📈 RESUMEN DESPUÉS DE {last_epoch} ÉPOCAS:")
                        print(f"   Mejor Accuracy: {df['val_accuracy'].max():.4f}")
                        print(f"   Mejor F1-Score: {df['val_f1'].max():.4f}")
                        print(f"   Loss actual: {df['train_loss'].iloc[-1]:.4f}")
                        print("-" * 50)
            
            time.sleep(5)  # Esperar 5 segundos antes de revisar de nuevo
            
        except KeyboardInterrupt:
            print("\n⏹️  Monitoreo detenido por el usuario")
            break
        except Exception as e:
            print(f"Error en monitoreo: {e}")
            time.sleep(10)

if __name__ == "__main__":
    monitor_training()
