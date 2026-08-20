"""
Pruebas del pipeline alineado.

A diferencia de la suite auditada -- que capturaba toda excepcion y
devolvia `False`, de modo que pytest la reportaba como aprobada aunque
fallara -- aqui todo se verifica con `assert` y un fallo real hace fallar
la prueba.

Ejecutar:
    python -m pytest tests/test_aligned.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.aligned import variables as V                      # noqa: E402
from src.aligned.config import load_config                  # noqa: E402
from src.aligned.evaluate import compute_metrics, copeland_ranking  # noqa: E402
from src.aligned.models import (                            # noqa: E402
    FTTransformer, MLPMultiHeadAttention, MultiHeadFeatureAttention,
)
from src.aligned.target import MAP_P403, MAP_P506, build_target  # noqa: E402


@pytest.fixture(scope="module")
def cfg():
    return load_config()


# =====================================================================
# Configuracion fiel al manuscrito
# =====================================================================
def test_config_respeta_el_manuscrito(cfg):
    assert cfg["training"]["optimizer"] == "adamw"
    assert cfg["training"]["final_epochs"] == 200
    assert cfg["training"]["early_stopping_patience"] == 20
    assert cfg["models"]["tabnet"]["n_steps"] == 3
    assert cfg["models"]["tabnet"]["mask_type"] == "sparsemax"
    assert cfg["models"]["ft_transformer"]["emb_dim"] == 64
    assert cfg["models"]["ft_transformer"]["n_layers"] == 4
    assert cfg["models"]["mlp_ma"]["hidden_dims"] == [256, 128, 64, 32]
    assert cfg["models"]["mlp_ma"]["n_heads"] >= 2
    assert cfg["gridsearch"]["cv_folds"] == 5
    assert cfg["preprocessing"]["numeric_scaling"] == "minmax"
    assert cfg["preprocessing"]["categorical_encoding"] == "label"


def test_split_suma_uno(cfg):
    s = cfg["split"]
    assert abs(s["train_size"] + s["val_size"] + s["test_size"] - 1.0) < 1e-9


def test_config_invalida_es_rechazada(cfg):
    from src.aligned.config import validate
    bad = {k: (dict(v) if isinstance(v, dict) else v) for k, v in cfg.items()}
    bad["training"] = {**cfg["training"], "optimizer": "adam"}
    with pytest.raises(ValueError, match="contradice el manuscrito"):
        validate(bad)


# =====================================================================
# Fuga del target
# =====================================================================
@pytest.mark.parametrize("col", [
    "P402", "P403", "P506", "P409", "P515", "P517", "P505", "P413_1",
    "P401_1", "P518", "P520", "lectura_score", "habito_3clases",
])
def test_variables_de_lectura_marcadas_como_fuga(col):
    assert V.is_leaky(col), f"{col} deberia bloquearse por fuga"


@pytest.mark.parametrize("col", [
    "P203", "P209", "P301", "P306", "P307_1", "P513A", "P514_1",
    "DOMINIO", "ESTRATOSOCIO", "edad_total", "idx_acceso_digital",
    "hh_nivel_educ_max",
])
def test_variables_admisibles_no_se_bloquean(col):
    assert not V.is_leaky(col), f"{col} es un predictor valido"


def test_build_feature_list_excluye_lectura_y_administrativas():
    universo = ["P203", "P306", "P402", "P403", "P506", "P413_5",
                "UBIGEO", "VIVIENDA", "P404_9_O", "edad_total"]
    feats = V.build_feature_list(universo)
    assert "P203" in feats and "P306" in feats and "edad_total" in feats
    for banned in ("P402", "P403", "P506", "P413_5", "UBIGEO", "VIVIENDA", "P404_9_O"):
        assert banned not in feats


def test_assert_no_leakage_detiene_el_pipeline():
    V.assert_no_leakage(["P203", "P306", "hh_nivel_educ_max"])   # no lanza
    with pytest.raises(AssertionError, match="FUGA DEL TARGET"):
        V.assert_no_leakage(["P203", "P517"])


def test_datos_procesados_sin_fuga(cfg):
    """Verificacion sobre los parquet reales, si el ETL ya se ejecuto."""
    base = ROOT / cfg["paths"]["processed_dir"]
    if not base.exists():
        pytest.skip("Ejecuta primero: python run_pipeline.py etl")
    for group in cfg["groups"]:
        path = base / group / "train.parquet"
        if not path.exists():
            continue
        cols = [c for c in pd.read_parquet(path).columns if c != "target"]
        V.assert_no_leakage(cols)


def test_esquema_identico_entre_splits(cfg):
    base = ROOT / cfg["paths"]["processed_dir"]
    if not base.exists():
        pytest.skip("Ejecuta primero: python run_pipeline.py etl")
    for group in cfg["groups"]:
        d = base / group
        if not (d / "train.parquet").exists():
            continue
        cols = [list(pd.read_parquet(d / f"{s}.parquet").columns)
                for s in ("train", "val", "test")]
        assert cols[0] == cols[1] == cols[2], f"esquema distinto en {group}"


def test_sin_nan_en_datos_procesados(cfg):
    base = ROOT / cfg["paths"]["processed_dir"]
    if not base.exists():
        pytest.skip("Ejecuta primero: python run_pipeline.py etl")
    for group in cfg["groups"]:
        p = base / group / "train.parquet"
        if p.exists():
            assert not pd.read_parquet(p).isna().any().any()


# =====================================================================
# Target
# =====================================================================
def test_mapeo_target_menores():
    assert MAP_P506[1.0] == 2 and MAP_P506[2.0] == 2      # frecuente
    assert MAP_P506[3.0] == 1 and MAP_P506[5.0] == 1      # medio
    assert MAP_P506[8.0] == 0                             # no leyo
    assert 9.0 not in MAP_P506                            # NS/NR descartado


def test_target_adultos_respeta_p402():
    df = pd.DataFrame({"P402": [2.0, 1.0, 1.0, 1.0, 0.0],
                       "P403": [1.0, 1.0, 3.0, 6.0, 1.0]})
    y = build_target(df, "chapter_300_400")
    assert y.iloc[0] == 0        # dice no haber leido, pese a P403=1
    assert y.iloc[1] == 2        # diariamente
    assert y.iloc[2] == 1        # una vez por semana
    assert y.iloc[3] == 0        # una vez al año
    assert pd.isna(y.iloc[4])    # capitulo no aplicado


def test_target_menores():
    df = pd.DataFrame({"P506": [1.0, 4.0, 8.0, 9.0, 0.0]})
    y = build_target(df, "chapter_500")
    assert list(y[:3]) == [2, 1, 0]
    assert pd.isna(y.iloc[3]) and pd.isna(y.iloc[4])


# =====================================================================
# Arquitecturas
# =====================================================================
def test_atencion_es_realmente_multicabeza():
    """
    El fallo central que encontro la auditoria: la 'multi-head attention'
    era una compuerta sigmoid escalar. Aqui se exige softmax sobre las
    variables, varias cabezas y pesos que sumen 1 por consulta.
    """
    attn = MultiHeadFeatureAttention(num_features=20, attn_dim=32, n_heads=4)
    out, w = attn(torch.randn(8, 20), return_attention=True)

    assert out.shape == (8, 20)
    assert w.shape == (8, 4, 20, 20), "se esperan (batch, cabezas, F, F)"
    assert w.shape[1] > 1, "debe haber mas de una cabeza"
    assert torch.allclose(w.sum(dim=-1), torch.ones(8, 4, 20), atol=1e-4), \
        "los pesos deben provenir de un softmax"


def test_mlp_ma_forward_y_arquitectura(cfg):
    m = MLPMultiHeadAttention(
        num_features=40, num_classes=3,
        hidden_dims=cfg["models"]["mlp_ma"]["hidden_dims"],
        n_heads=cfg["models"]["mlp_ma"]["n_heads"], dropout=0.3)
    assert m(torch.randn(16, 40)).shape == (16, 3)
    dims = [l.out_features for l in m.mlp if isinstance(l, torch.nn.Linear)]
    assert dims == [256, 128, 64, 32]


def test_mlp_ma_importancia_por_variable():
    m = MLPMultiHeadAttention(num_features=25, n_heads=4)
    imp = m.feature_importance(torch.randn(32, 25))
    assert imp.shape == (25,), "una importancia por variable, no un escalar"
    assert np.isfinite(imp).all()


def test_ft_transformer_forward_y_cls(cfg):
    ft = cfg["models"]["ft_transformer"]
    m = FTTransformer(num_features=30, emb_dim=ft["emb_dim"],
                      n_layers=ft["n_layers"], n_heads=ft["n_heads"])
    assert m(torch.randn(8, 30)).shape == (8, 3)
    assert m.cls_token.shape == (1, 1, 64)
    assert len(m.attn) == 4


def test_ft_transformer_expone_atencion():
    """La seccion III-G-2 necesita los pesos por cabeza y capa."""
    m = FTTransformer(num_features=15, emb_dim=32, n_layers=2, n_heads=4)
    maps = m.attention_maps(torch.randn(8, 15))
    assert maps.shape == (2, 4, 16, 16)          # (capas, cabezas, F+1, F+1)
    assert m.feature_importance(torch.randn(8, 15)).shape == (15,)


def test_tabnet_usa_sparsemax_y_tres_pasos(cfg):
    pytest.importorskip("pytorch_tabnet")
    from src.aligned.models import TabNetModel
    m = TabNetModel(num_features=20, n_steps=cfg["models"]["tabnet"]["n_steps"],
                    mask_type=cfg["models"]["tabnet"]["mask_type"])
    assert m.n_steps == 3
    assert m.mask_type == "sparsemax"


# =====================================================================
# Metricas y Copeland
# =====================================================================
def test_auc_reporta_motivo_en_lugar_de_none_mudo():
    y = np.array([0, 1, 2, 0, 1, 2])
    m = compute_metrics(y, y, None, n_classes=3)
    assert m["roc_auc_ovr"] is None
    assert m["roc_auc_ovr_error"], "un AUC ausente debe explicar por que"


def test_auc_se_calcula_con_probabilidades():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 200)
    proba = rng.dirichlet([1, 1, 1], 200)
    m = compute_metrics(y, proba.argmax(1), proba, n_classes=3)
    assert m["roc_auc_ovr"] is not None
    assert 0.0 <= m["roc_auc_ovr"] <= 1.0
    assert m["roc_auc_ovr_error"] is None


def test_copeland_sigue_la_regla_del_articulo():
    """+1 por par ganado, no +1 por metrica."""
    frame = pd.DataFrame(
        {"accuracy": [0.90, 0.85, 0.80],
         "f1_weighted": [0.90, 0.85, 0.80],
         "precision_weighted": [0.90, 0.85, 0.80]},
        index=["A", "B", "C"])
    r = copeland_ranking(frame, ["accuracy", "f1_weighted", "precision_weighted"])
    scores = r.set_index("classifier")["copeland_score"].to_dict()
    assert scores["A"] == 2 and scores["B"] == 0 and scores["C"] == -2
    assert r.iloc[0]["classifier"] == "A"


def test_copeland_empate():
    frame = pd.DataFrame(
        {"accuracy": [0.9, 0.9], "f1_weighted": [0.9, 0.9],
         "precision_weighted": [0.9, 0.9]}, index=["A", "B"])
    r = copeland_ranking(frame, ["accuracy", "f1_weighted", "precision_weighted"])
    assert set(r["copeland_score"]) == {0}
