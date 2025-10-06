#!/usr/bin/env python3

import os
import time
import glob

print("📊 MONITOR DE ENTRENAMIENTO EN TIEMPO REAL")
print("=" * 50)

def monitor_training():
    """Monitorea los archivos de métricas del entrenamiento actual"""
    
    # Buscar el directorio más reciente
    model_dirs = glob.glob("outputs/models/202*")
    if not model_dirs:
        print("❌ No se encontró entrenamiento activo")
        return
    
    latest_dir = max(model_dirs)
    metrics_file = os.path.join(latest_dir, "metrics_by_epoch.csv")
    
    print(f"📁 Monitoreando: {latest_dir}")
    print("-" * 50)
    
    last_size = 0
    best_f1 = 0.0
    
    while True:
        try:
            if os.path.exists(metrics_file):
                current_size = os.path.getsize(metrics_file)
                
                if current_size > last_size:
                    with open(metrics_file, 'r') as f:
                        lines = f.readlines()
                        if len(lines) > 1:  # Skip header
                            last_line = lines[-1].strip().split(',')
                            if len(last_line) >= 5:
                                epoch = last_line[0]
                                loss = float(last_line[1])
                                acc = float(last_line[2])
                                f1 = float(last_line[4])
                                
                                if f1 > best_f1:
                                    best_f1 = f1
                                    status = "🎯 NUEVO MEJOR!"
                                else:
                                    status = ""
                                
                                print(f"Época {epoch:>3}: Loss={loss:.4f} | Acc={acc:.1%} | F1={f1:.4f} {status}")
                    
                    last_size = current_size
            
            time.sleep(5)  # Verificar cada 5 segundos
            
        except KeyboardInterrupt:
            print("\n👋 Monitoreo detenido")
            break
        except Exception as e:
            print(f"⚠️  Error: {e}")
            time.sleep(10)

if __name__ == "__main__":
    monitor_training()
