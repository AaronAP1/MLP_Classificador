import json, os, datetime, joblib, argparse
from utils.etl import preprocess_and_load
from utils.seeds import set_global_seeds
from models.mlp_ma import MLPWithInputAttention
from models.trainer import train_and_evaluate
from utils.plot_training_metrics import plot_training_metrics


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Entrenar MLP con Atención (MLP_MA)")
    parser.add_argument("--config", default="configs/config_ma.json", help="Ruta al archivo de configuración")
    parser.add_argument("--paths", default=None, help="Ruta al archivo de paths (opcional)")
    parser.add_argument("--max_epochs", type=int, default=None, help="Sobrescribir épocas para corrida rápida")
    args = parser.parse_args()

    cfg = load_json(args.config)
    paths_path = args.paths if args.paths else cfg.get("paths_override", "configs/paths.json")
    paths = load_json(paths_path)

    set_global_seeds(cfg["data"]["random_state"]) 

    if args.max_epochs is not None:
        cfg["model"]["epochs"] = int(args.max_epochs)

    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_run = os.path.join(cfg["outputs"]["model_dir"], run_id)
    plots_run = os.path.join(cfg["outputs"]["plots_dir"], run_id)
    os.makedirs(model_run, exist_ok=True)
    os.makedirs(plots_run, exist_ok=True)

    cfg["outputs"]["model_dir"] = model_run
    cfg["outputs"]["plots_dir"] = plots_run

    # Guardar nombres de features transformadas
    feat_names_path = os.path.join(model_run, "feature_names.txt")

    dl_tr, dl_va, dl_te, in_dim, n_cls = preprocess_and_load(cfg, paths, feature_names_out_path=feat_names_path)

    # Cargar nombres de clases
    le = joblib.load(os.path.join(paths["processed_data_dir"], "label_encoder.pkl"))
    class_names = list(le.classes_)

    mcfg = cfg["model"]
    model = MLPWithInputAttention(input_dim=in_dim,
                                  num_classes=n_cls,
                                  hidden_dims=mcfg["hidden_dims"],
                                  attn_dim=mcfg["attn_dim"],
                                  dropout_p=mcfg.get("dropout_p", 0.0),
                                  norm='layer', residual=False)

    test_metrics = train_and_evaluate(model, dl_tr, dl_va, dl_te, cfg, class_names)

    metrics_csv = os.path.join(cfg["outputs"]["model_dir"], "metrics_by_epoch.csv")
    if os.path.exists(metrics_csv):
        plot_training_metrics(metrics_csv, cfg["outputs"]["plots_dir"])    

    print(f"\nEntrenamiento completado. Outputs en:\n  Modelos: {cfg['outputs']['model_dir']}\n  Plots:   {cfg['outputs']['plots_dir']}")


if __name__ == "__main__":
    main()
