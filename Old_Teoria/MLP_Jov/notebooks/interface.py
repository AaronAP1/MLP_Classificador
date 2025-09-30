import os
import joblib
import pandas as pd
import numpy as np
import torch
from models.model_jovenes import JovenesMLP
import json

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def main():
    # 1) Carga de configs
    cfg   = load_json("configs/config_jovenes.json")
    paths = load_json("configs/paths.json")

    # 2) Carga preprocesador y label encoder
    preproc = joblib.load(os.path.join(paths["processed_data_dir"], "preprocessor.pkl"))
    le      = joblib.load(os.path.join(paths["processed_data_dir"], "label_encoder.pkl"))

    # 3) Instancia y carga pesos del modelo
    #    Debes usar los mismos hyperparams que en el entrenamiento
    mcfg = cfg["model"]
    # input_dim lo obtuviste al entrenar: es preproc.transform(X_train).shape[1]
    input_dim = preproc.transform(
        pd.DataFrame(columns=cfg["etl"]["cat_cols"]+cfg["etl"]["num_cols"])
    ).shape[1]
    model = JovenesMLP(input_dim=input_dim,
                       hidden_dims=mcfg["hidden_dims"],
                       attn_dim=mcfg["attn_dim"],
                       num_classes=len(le.classes_))
    model.load_state_dict(torch.load(os.path.join(cfg["outputs"]["model_dir"],
                                                  "jovenes_best.pt")))
    model.eval()

    # 4) Carga tus nuevas muestras (sin target)
    df_new = pd.read_csv("data/raw/nuevos_jovenes.csv")  
    # asegúrate de que tenga las mismas columnas que defines en config["etl"]["cat_cols"] y ["num_cols"]

    # 5) Preprocesa
    X_new = preproc.transform(df_new[cfg["etl"]["cat_cols"] + cfg["etl"]["num_cols"]])
    X_t   = torch.from_numpy(X_new.astype(np.float32))

    # 6) Forward + softmax + argmax
    with torch.no_grad():
        logits = model(X_t)                         # (N, 3)
        probs  = torch.softmax(logits, dim=1).cpu().numpy()  # (N, 3)
        preds  = np.argmax(probs, axis=1)           # (N,)

    # 7) Mapea índices a etiquetas de texto
    labels = le.inverse_transform(preds)            # e.g. ["no_lector","lector_medio",...]

    # 8) Guarda resultados
    df_new["pred_class"] = labels
    df_new["pred_prob"]  = probs.max(axis=1)
    df_new.to_csv("outputs/models/nuevas_predicciones.csv", index=False)
    print("Clasificaciones guardadas en outputs/models/nuevas_predicciones.csv")

if __name__=="__main__":
    main()
