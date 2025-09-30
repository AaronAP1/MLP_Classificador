#!/usr/bin/env python3

import json
import os
from datetime import datetime

# Cargar configs
with open('configs/config_jovenes.json', 'r') as f:
    config = json.load(f)
with open('configs/paths.json', 'r') as f:
    paths = json.load(f)

print("🚀 ENTRENAMIENTO MLP - CLASIFICACIÓN DE HÁBITOS DE LECTURA")
print("=" * 60)

try:
    # 1. ETL
    print("📊 Ejecutando ETL...")
    from utils.etl import preprocess_and_load
    dl_tr, dl_va, dl_te, input_size, n_classes = preprocess_and_load(config, paths)
    print(f"✅ ETL completado - Input: {input_size}, Clases: {n_classes}")
    
    # 2. Modelo
    print("🧠 Inicializando modelo...")
    from models.model_jovenes import JovenesMLP
    model = JovenesMLP(
        input_dim=input_size, 
        num_classes=n_classes, 
        hidden_dims=config["model"]["hidden_dims"],
        attn_dim=config["model"]["attn_dim"],
        dropout_p=config["model"]["dropout_p"]
    )
    print(f"✅ Modelo creado - Parámetros: {sum(p.numel() for p in model.parameters()):,}")
    
    # 3. Entrenamiento 
    print("🏋️ Iniciando entrenamiento...")
    from models.trainer import train_and_evaluate
    
    # Crear directorio de salida con timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = os.path.join(config["outputs"]["model_dir"], timestamp)
    plots_dir = os.path.join(config["outputs"]["plots_dir"], timestamp)
    
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    # Actualizar config con directorios de salida
    config["outputs"]["model_dir"] = model_dir
    config["outputs"]["plots_dir"] = plots_dir
    
    # Entrenar modelo
    class_names = ["lector_frecuente", "lector_medio", "no_lector"]
    best_model = train_and_evaluate(model, dl_tr, dl_va, dl_te, config, class_names)
    
    print("🎉 ENTRENAMIENTO COMPLETADO!")
    print(f"📁 Resultados en: {model_dir}")
    print(f"📈 Gráficos en: {plots_dir}")
    
except Exception as e:
    print(f"❌ Error: {str(e)}")
    import traceback
    traceback.print_exc()
