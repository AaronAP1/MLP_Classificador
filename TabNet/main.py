import argparse, os, datetime, json
from utils.etl import preprocess_and_load
from models.tabnet import TabNetWrapper
from models.trainer import TabNetTrainer

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def main():
    parser = argparse.ArgumentParser(description="Entrenar TabNet")
    parser.add_argument("--config", required=True)
    parser.add_argument("--paths", required=True)
    parser.add_argument("--max_epochs", type=int, default=None)
    args = parser.parse_args()
    cfg = load_json(args.config)
    paths = load_json(args.paths)
    if args.max_epochs is not None:
        cfg["model"]["epochs"] = int(args.max_epochs)
    # Ensure outputs are under this model folder
    base_dir = os.path.dirname(os.path.abspath(__file__))
    def to_abs(p):
        return p if os.path.isabs(p) else os.path.join(base_dir, p)
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_models_base = to_abs(cfg["outputs"]["model_dir"])  # TabNet/outputs/models
    out_plots_base  = to_abs(cfg["outputs"]["plots_dir"])   # TabNet/outputs/plots
    os.makedirs(out_models_base, exist_ok=True)
    os.makedirs(out_plots_base, exist_ok=True)
    model_run = os.path.join(out_models_base, run_id)
    plots_run = os.path.join(out_plots_base, run_id)
    os.makedirs(model_run, exist_ok=True)
    os.makedirs(plots_run, exist_ok=True)
    cfg["outputs"]["model_dir"] = model_run
    cfg["outputs"]["plots_dir"] = plots_run
    X_train, y_train, X_val, y_val, X_test, y_test, cat_idxs, cat_dims, cat_emb_dim, n_cls = preprocess_and_load(cfg, paths)
    wrapper = TabNetWrapper(
        input_dim=X_train.shape[1],
        output_dim=n_cls,
        cat_idxs=cat_idxs,
        cat_dims=cat_dims,
        cat_emb_dim=cat_emb_dim,
        config=cfg["model"]
    )
    trainer = TabNetTrainer(wrapper, cfg)
    metrics = trainer.fit_eval(X_train, y_train, X_val, y_val, X_test, y_test, class_names=list(range(n_cls)))
    # Persist basic metrics for consistency
    metrics_txt = os.path.join(cfg['outputs']['model_dir'], 'final_test_metrics.txt')
    with open(metrics_txt, 'w', encoding='utf-8') as f:
        f.write('METRICAS TEST\n')
        f.write(f"Accuracy: {metrics.get('accuracy', 0):.4f}\n")
        f.write(f"Precision (weighted): {metrics.get('precision', 0):.4f}\n")
        f.write(f"Recall (weighted): {metrics.get('recall', 0):.4f}\n")
        f.write(f"F1 (weighted): {metrics.get('f1', 0):.4f}\n")
    print("Entrenamiento completado. Outputs en:")
    print("  Modelos:", cfg['outputs']['model_dir'])
    print("  Plots:", cfg['outputs']['plots_dir'])

if __name__ == "__main__":
    main()
