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
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_run = os.path.join(cfg["outputs"]["model_dir"], run_id)
    plots_run = os.path.join(cfg["outputs"]["plots_dir"], run_id)
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
    print("Entrenamiento completado. Outputs en:")
    print("  Modelos:", cfg['outputs']['model_dir'])
    print("  Plots:", cfg['outputs']['plots_dir'])

if __name__ == "__main__":
    main()
