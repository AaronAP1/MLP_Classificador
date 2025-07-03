"""
Script mejorado para entrenamiento del modelo MLP con análisis completo
"""
import json
import os
import sys
import datetime
import torch

# Agregar el directorio actual al path
sys.path.append('.')

from utils.etl import preprocess_and_load
from models.model_jovenes import JovenesMLP
from models.trainer import train_and_evaluate
from utils.plot_training_metrics import plot_training_metrics
import joblib

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def main():
    print("🚀 INICIANDO ENTRENAMIENTO MEJORADO DEL MODELO MLP")
    print("=" * 60)
    
    # 1) Cargar configuraciones
    cfg = load_json("configs/config_jovenes.json")
    paths = load_json("configs/paths.json")
    
    # 1.1) Generar un run ID
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1.2) Crear directorios de salida
    model_run = os.path.join(cfg["outputs"]["model_dir"], run_id)
    plots_run = os.path.join(cfg["outputs"]["plots_dir"], run_id)
    os.makedirs(model_run, exist_ok=True)
    os.makedirs(plots_run, exist_ok=True)
    
    # 1.3) Actualizar configuración
    cfg["outputs"]["model_dir"] = model_run
    cfg["outputs"]["plots_dir"] = plots_run
    
    print(f"📁 Guardando resultados en: {run_id}")
    print(f"📊 Configuración del modelo:")
    print(f"   - Arquitectura: {cfg['model']['hidden_dims']}")
    print(f"   - Learning Rate: {cfg['model']['lr']}")
    print(f"   - Batch Size: {cfg['model']['batch_size']}")
    print(f"   - Épocas máximas: {cfg['model']['epochs']}")
    print(f"   - Dropout: {cfg['model']['dropout_p']}")
    
    # 2) Preprocesar datos
    print("\n📥 Preprocesando datos...")
    try:
        dl_tr, dl_va, dl_te, in_dim, n_cls = preprocess_and_load(cfg, paths)
        print(f"✅ Datos procesados exitosamente:")
        print(f"   - Dimensión de entrada: {in_dim}")
        print(f"   - Número de clases: {n_cls}")
        print(f"   - Batches de entrenamiento: {len(dl_tr)}")
        print(f"   - Batches de validación: {len(dl_va)}")
        print(f"   - Batches de test: {len(dl_te)}")
    except Exception as e:
        print(f"❌ Error en preprocesamiento: {e}")
        return
    
    # 3) Cargar nombres de clases
    try:
        le = joblib.load(os.path.join(paths["processed_data_dir"], "label_encoder.pkl"))
        class_names = list(le.classes_)
        print(f"🏷️  Clases objetivo: {class_names}")
    except:
        class_names = [str(i) for i in range(n_cls)]
        print(f"🏷️  Usando nombres genéricos para clases: {class_names}")
    
    # 4) Construir modelo
    print(f"\n🧠 Construyendo modelo MLP...")
    mcfg = cfg["model"]
    model = JovenesMLP(
        input_dim=in_dim,
        hidden_dims=mcfg["hidden_dims"],
        attn_dim=mcfg["attn_dim"],
        num_classes=n_cls,
        dropout_p=mcfg.get("dropout_p", 0.0)
    )
    
    # Contar parámetros
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 Parámetros del modelo:")
    print(f"   - Total: {total_params:,}")
    print(f"   - Entrenables: {trainable_params:,}")
    
    # 5) Entrenar y evaluar
    print(f"\n🎯 INICIANDO ENTRENAMIENTO...")
    print("=" * 60)
    
    try:
        test_metrics = train_and_evaluate(model, dl_tr, dl_va, dl_te, cfg, class_names)
        
        # 6) Visualizar métricas de entrenamiento
        print(f"\n📈 Generando visualizaciones...")
        metrics_csv = os.path.join(cfg["outputs"]["model_dir"], "metrics_by_epoch.csv")
        if os.path.exists(metrics_csv):
            plot_training_metrics(metrics_csv, cfg["outputs"]["plots_dir"])
        
        print(f"\n🎉 ENTRENAMIENTO COMPLETADO EXITOSAMENTE!")
        print("=" * 60)
        print(f"📊 RESULTADOS FINALES:")
        print(f"   - Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"   - F1-Score: {test_metrics['f1']:.4f}")
        print(f"   - Precision: {test_metrics['precision']:.4f}")
        print(f"   - Recall: {test_metrics['recall']:.4f}")
        print(f"📁 Resultados guardados en: {cfg['outputs']['model_dir']}")
        print(f"📈 Gráficos guardados en: {cfg['outputs']['plots_dir']}")
        
    except Exception as e:
        print(f"❌ Error durante el entrenamiento: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
