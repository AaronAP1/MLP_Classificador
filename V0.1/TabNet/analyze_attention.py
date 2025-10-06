import joblib, numpy as np
from models.tabnet import TabNetWrapper
from utils.etl import preprocess_and_load
import argparse, os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--paths', required=True)
    args = parser.parse_args()
    cfg = joblib.load(args.config) if args.config.endswith('.pkl') else joblib.load(args.config.replace('.json','.pkl'))
    paths = joblib.load(args.paths) if args.paths.endswith('.pkl') else joblib.load(args.paths.replace('.json','.pkl'))
    X_train, y_train, X_val, y_val, X_test, y_test, cat_idxs, cat_dims, cat_emb_dim, n_cls = preprocess_and_load(cfg, paths)
    model = TabNetWrapper(
        input_dim=X_train.shape[1],
        output_dim=n_cls,
        cat_idxs=cat_idxs,
        cat_dims=cat_dims,
        cat_emb_dim=cat_emb_dim,
        config=cfg["model"]
    )
    model.model.load_model(os.path.join(cfg["outputs"]["model_dir"], "best_model.zip"))
    # Extraer importancias de máscaras
    masks = model.explain(X_test)
    np.savetxt(os.path.join(cfg["outputs"]["model_dir"], "tabnet_masks.csv"), masks, delimiter=',')
    print("Export de máscaras TabNet completado.")
