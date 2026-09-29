"""
Tratamiento del desbalance de clases.

Se aplica dentro de cada ajuste y SOLO sobre la parte de entrenamiento:
en cada fold de la validacion cruzada y en el entrenamiento final. Nunca
sobre validacion ni prueba.

Metodos (`configs/pipeline.yaml -> balancing.method`):

    class_weights  (por defecto)
        Entropia cruzada ponderada por el inverso de la frecuencia de cada
        clase. No crea muestras. Es lo que conviene declarar en el
        articulo: "la funcion de perdida se pondero inversamente a la
        frecuencia de cada clase".

    smote
        SMOTE sobre la parte de entrenamiento, sin ponderar la perdida
        (hacer ambas cosas sobrecorrige). Si se elige, el articulo debe
        mencionarlo en Metodos.

    none
        Sin tratamiento.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from .trainer import compute_class_weights


def balance_training_set(
    X: np.ndarray, y: np.ndarray, cfg: dict, seed: int
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Dict[str, object]]:
    """
    Returns:
        (X, y, class_weights, informe) listos para entrenar. `class_weights`
        es None cuando el metodo no pondera la perdida.
    """
    bcfg = cfg["balancing"]
    method = bcfg.get("method", "class_weights")
    n_classes = cfg["n_classes"]
    before = np.bincount(np.asarray(y).astype(int), minlength=n_classes).tolist()

    if method == "class_weights":
        w = compute_class_weights(y, n_classes)
        return X, y, w, {"metodo": method, "pesos": [round(float(v), 4) for v in w],
                         "distribucion": before}

    if method == "none":
        return X, y, None, {"metodo": method, "distribucion": before}

    if method == "smote":
        smallest = int(min(before))
        if smallest < bcfg.get("min_class_count", 6):
            # Con tan pocas muestras SMOTE no puede interpolar: se cae a pesos.
            w = compute_class_weights(y, n_classes)
            return X, y, w, {"metodo": "class_weights",
                             "motivo": f"clase minoritaria con {smallest} muestras; SMOTE omitido",
                             "distribucion": before}
        from imblearn.over_sampling import SMOTE
        k = min(int(bcfg.get("k_neighbors", 5)), smallest - 1)
        Xr, yr = SMOTE(random_state=seed, k_neighbors=k).fit_resample(X, y)
        return (np.asarray(Xr, dtype=np.float32), np.asarray(yr).astype(int), None,
                {"metodo": method, "k_neighbors": k, "antes": before,
                 "despues": np.bincount(yr, minlength=n_classes).tolist()})

    raise ValueError(f"balancing.method desconocido: {method!r} "
                     "(opciones: class_weights, smote, none)")
