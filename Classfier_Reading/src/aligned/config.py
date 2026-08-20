"""Carga y validacion de `configs/pipeline.yaml`, la fuente unica de verdad."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml

REQUIRED_SECTIONS = [
    "groups", "preprocessing", "features", "split", "balancing",
    "training", "gridsearch", "models", "evaluation", "paths",
]

# Valores que el articulo fija de forma explicita y que el codigo no debe
# poder contradecir en silencio.
PAPER_INVARIANTS = {
    ("training", "optimizer"): "adamw",
    ("training", "final_epochs"): 200,
    ("training", "early_stopping_patience"): 20,
    ("preprocessing", "categorical_encoding"): "label",
    ("preprocessing", "numeric_scaling"): "minmax",
    ("features", "target_n_features"): 71,
    ("gridsearch", "cv_folds"): 5,
}


def project_root() -> Path:
    """Raiz del proyecto (carpeta `Classfier_Reading`)."""
    return Path(__file__).resolve().parents[2]


def load_config(path: Optional[Path] = None) -> dict:
    path = Path(path) if path else project_root() / "configs" / "pipeline.yaml"
    if not path.exists():
        raise FileNotFoundError(f"No se encontro la configuracion: {path}")

    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    missing = [s for s in REQUIRED_SECTIONS if s not in cfg]
    if missing:
        raise ValueError(f"Faltan secciones en {path.name}: {missing}")

    validate(cfg)
    return cfg


def validate(cfg: dict) -> None:
    """Comprueba coherencia interna y fidelidad al manuscrito."""
    s = cfg["split"]
    total = s["train_size"] + s["val_size"] + s["test_size"]
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"Las proporciones del split suman {total}, deben sumar 1.0")

    for (section, key), expected in PAPER_INVARIANTS.items():
        actual = cfg[section][key]
        if actual != expected:
            raise ValueError(
                f"cfg['{section}']['{key}'] = {actual!r} contradice el manuscrito "
                f"(esperado {expected!r}). Si el cambio es deliberado, corrige "
                f"tambien el articulo antes de tocar este archivo."
            )

    tab = cfg["models"]["tabnet"]
    if tab["n_steps"] != 3:
        raise ValueError("El articulo declara 3 decision steps en TabNet (III-C-2).")
    if tab["mask_type"] != "sparsemax":
        raise ValueError("El articulo declara mascaras sparsemax en TabNet (III-C-2).")

    ft = cfg["models"]["ft_transformer"]
    if ft["emb_dim"] != 64 or ft["n_layers"] != 4:
        raise ValueError("El articulo declara embeddings de 64 y 4 encoders (III-C-3).")

    mlp = cfg["models"]["mlp_ma"]
    if mlp["hidden_dims"] != [256, 128, 64, 32]:
        raise ValueError("El articulo declara 256->128->64->32 en MLP+MA (III-C-1).")
    if mlp["n_heads"] < 2:
        raise ValueError("MLP+MA requiere atencion multi-cabeza: n_heads >= 2 (III-C-1).")

    preset = cfg["gridsearch"]["preset"]
    if preset not in cfg["gridsearch"]["presets"]:
        raise ValueError(f"Preset desconocido: {preset!r}")
