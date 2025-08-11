import json, os, datetime, joblib, argparse
from utils.etl import preprocess_and_load
from utils.seeds import set_global_seeds
from models.ft_transformer import FTTransformer
from models.trainer import train_and_evaluate
from utils.plot_training_metrics import plot_training_metrics


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Entrenar FT-Transformer")
    parser.add_argument("--config", required=True, help="Ruta al archivo de configuración")
    parser.add_argument("--paths", required=True, help="Ruta al archivo de paths")
    parser.add_argument("--max_epochs", type=int, default=None, help="Sobrescribir épocas para corrida rápida")
    args = parser.parse_args()

    cfg = load_json(args.config)
    paths = load_json(args.paths)

    if args.max_epochs is not None:
        cfg["model"]["epochs"] = int(args.max_epochs)

    set_global_seeds(cfg["data"]["random_state"]) 

    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_run = os.path.join(cfg["outputs"]["model_dir"], run_id)
    plots_run = os.path.join(cfg["outputs"]["plots_dir"], run_id)
    os.makedirs(model_run, exist_ok=True)
    os.makedirs(plots_run, exist_ok=True)
    cfg["outputs"]["model_dir"] = model_run
    cfg["outputs"]["plots_dir"] = plots_run

    dl_tr, dl_va, dl_te, n_num, cat_dims, n_cls = preprocess_and_load(cfg, paths)

    mcfg = cfg["model"]
    model = FTTransformer(
        num_features=n_num,
        cat_features=cat_dims,  # per-column cardinalities from ETL (includes OOV)
        emb_dim=mcfg["emb_dim"],
        n_heads=mcfg["n_heads"],
        n_layers=mcfg["n_layers"],
        num_classes=n_cls,
        dropout=mcfg.get("dropout_p", 0.2)
    )

    le = joblib.load(os.path.join(paths["processed_data_dir"], "label_encoder.pkl"))
    class_names = list(le.classes_)

    test_metrics = train_and_evaluate(model, dl_tr, dl_va, dl_te, cfg, class_names)

    metrics_csv = os.path.join(cfg["outputs"]["model_dir"], "metrics_by_epoch.csv")
    if os.path.exists(metrics_csv):
        plot_training_metrics(metrics_csv, cfg["outputs"]["plots_dir"])    

    print(f"\nEntrenamiento completado. Outputs en:\n  Modelos: {cfg['outputs']['model_dir']}\n  Plots:   {cfg['outputs']['plots_dir']}")


if __name__ == "__main__":
    main()
