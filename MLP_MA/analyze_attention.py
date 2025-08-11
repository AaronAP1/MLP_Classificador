import os, json, torch, joblib
import numpy as np
from utils.etl import preprocess_and_load
from models.mlp_ma import MLPWithInputAttention


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def main():
    cfg = load_json("configs/config_ma.json")
    paths = load_json("configs/paths.json")

    model_dir = cfg["outputs"]["model_dir"]
    run_dirs = sorted([d for d in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir,d))])
    if not run_dirs:
        print("No hay runs en outputs/models")
        return
    last_run = os.path.join(model_dir, run_dirs[-1])
    feat_names_path = os.path.join(last_run, 'feature_names.txt')
    if not os.path.exists(feat_names_path):
        print("No se encontraron nombres de features, ejecuta main.py primero")
        return
    with open(feat_names_path, 'r', encoding='utf-8') as f:
        feature_names = [line.strip() for line in f.readlines() if line.strip()]

    dl_tr, dl_va, dl_te, in_dim, n_cls = preprocess_and_load(cfg, paths)  # para conocer dimensiones

    model = MLPWithInputAttention(input_dim=in_dim, num_classes=n_cls,
                                  hidden_dims=cfg['model']['hidden_dims'],
                                  attn_dim=cfg['model']['attn_dim'],
                                  dropout_p=cfg['model'].get('dropout_p',0.0))

    best_path = os.path.join(last_run, 'best_model.pt')
    if not os.path.exists(best_path):
        print("No se encontró best_model.pt en el último run")
        return
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(best_path, map_location=device))
    model.to(device); model.eval()

    all_scores = []
    with torch.no_grad():
        for Xb, _ in dl_te:
            Xb = Xb.to(device)
            _, scores = model(Xb, return_attn=True)
            all_scores.append(scores.cpu().numpy())
    if not all_scores:
        print("No hay batches en test")
        return
    scores_arr = np.concatenate(all_scores, axis=0)  # (N, F)
    mean_scores = scores_arr.mean(axis=0)

    out_csv = os.path.join(last_run, 'feature_attention_importance.csv')
    with open(out_csv, 'w', encoding='utf-8') as f:
        f.write("feature,mean_attention\n")
        for name, sc in zip(feature_names, mean_scores):
            f.write(f"{name},{sc}\n")
    print(f"Importancias por atención guardadas en {out_csv}")


if __name__ == '__main__':
    main()
