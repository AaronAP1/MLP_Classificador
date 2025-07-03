import json, os
from utils.etl       import preprocess_and_load
from models.model_jovenes import JovenesMLP
from models.trainer  import train_and_evaluate
from utils.plot_training_metrics import plot_training_metrics
import joblib
import datetime

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def main():
    # 1) Cargar config y paths
    cfg   = load_json("configs/config_jovenes.json")
    paths = load_json("configs/paths.json")
    
    # 1.1) Generar un run ID a partir de la fecha/hora
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    model_run = os.path.join(cfg["outputs"]["model_dir"], run_id)
    plots_run = os.path.join(cfg["outputs"]["plots_dir"], run_id)
    os.makedirs(model_run, exist_ok=True)
    os.makedirs(plots_run, exist_ok=True)


   # 1.2) Construir rutas de salida específicas para este run
    model_run = os.path.join(cfg["outputs"]["model_dir"], run_id)
    plots_run = os.path.join(cfg["outputs"]["plots_dir"], run_id)
    os.makedirs(model_run, exist_ok=True)
    os.makedirs(plots_run, exist_ok=True)

   # 1.3) Inyectar esas rutas en el config para que trainer las use
    cfg["outputs"]["model_dir"] = model_run
    cfg["outputs"]["plots_dir"] = plots_run

    # 2) Preprocesar y obtener DataLoaders
    dl_tr, dl_va, dl_te, in_dim, n_cls = preprocess_and_load(cfg, paths)

    # 3) Recuperar nombres de clase
    le = joblib.load(os.path.join(paths["processed_data_dir"], "label_encoder.pkl"))
    class_names = list(le.classes_)

    # 4) Construir modelo
    mcfg = cfg["model"]
    model = JovenesMLP(input_dim=in_dim,
                       hidden_dims=mcfg["hidden_dims"],
                       attn_dim=mcfg["attn_dim"],
                       num_classes=n_cls,
                       dropout_p=mcfg.get("dropout_p", 0.0))

    # 5) Entrenar y evaluar
    test_metrics = train_and_evaluate(model, dl_tr, dl_va, dl_te, cfg, class_names)
    
    # 6) Visualizar métricas de entrenamiento
    metrics_csv = os.path.join(cfg["outputs"]["model_dir"], "metrics_by_epoch.csv")
    if os.path.exists(metrics_csv):
        plot_training_metrics(metrics_csv, cfg["outputs"]["plots_dir"])
    
    print(f"\n🎯 Entrenamiento completado exitosamente!")
    print(f"📊 Resultados guardados en: {cfg['outputs']['model_dir']}")
    print(f"📈 Gráficos guardados en: {cfg['outputs']['plots_dir']}")

if __name__ == "__main__":
    main()
