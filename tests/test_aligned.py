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


def test_auc_macro_es_la_media_de_las_clases():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 3, 300)
    proba = rng.dirichlet([1, 1, 1], 300)
    m = compute_metrics(y, proba.argmax(1), proba, n_classes=3,
                        class_names=["A", "B", "C"])
    per = m["roc_auc_per_class"]
    assert m["roc_auc_ovr"] == pytest.approx(np.mean(list(per.values())), abs=1e-9)
    assert m["roc_auc_ovr_weighted"] is not None


# =====================================================================
# Correcciones de la revision de septiembre
# =====================================================================
def test_target_adultos_por_libros():
    """P408/P409: lectura de libros, misma escala que P506 de menores."""
    df = pd.DataFrame({"P408": [2.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                       "P409": [0.0, 1.0, 2.0, 4.0, 5.0, 0.0]})
    y = build_target(df, "chapter_400_libros")
    assert y.iloc[0] == 0          # no leyo libros
    assert y.iloc[1] == 2 and y.iloc[2] == 2
    assert y.iloc[3] == 1          # una vez al mes
    assert y.iloc[4] == 0          # cada tres meses
    assert pd.isna(y.iloc[5])      # capitulo no aplicado


def test_config_usa_libros_para_adultos(cfg):
    assert cfg["groups"]["adultos"]["target_source"] == "chapter_400_libros"
    assert cfg["groups"]["ninos"]["target_source"] == "chapter_500"


def test_config_rechaza_target_desconocido(cfg):
    from src.aligned.config import validate
    import copy
    bad = copy.deepcopy(cfg)
    bad["groups"]["adultos"]["target_source"] = "P999"
    with pytest.raises(ValueError, match="target_source"):
        validate(bad)


@pytest.mark.parametrize("col", ["P408", "P409"])
def test_fuentes_del_target_por_libros_son_fuga(col):
    assert V.is_leaky(col)


def test_factores_de_expansion_no_son_predictores():
    feats = V.build_feature_list(["FACTORFINAL_HOGAR", "FACTOR200_FINAL",
                                  "P201", "P204", "P209", "P306"])
    assert feats == ["P209", "P306"]


def test_armonizacion_une_columnas_de_merge_y_vacios():
    from src.aligned.etl import harmonize_source
    raw = pd.DataFrame({
        "P201": [np.nan, 2.0, np.nan],
        "P201_x": [1.0, np.nan, np.nan],
        "P201_y": [np.nan, np.nan, 3.0],
        "P506": [np.nan, 1.0, np.nan],
        "edad_total": [16.0, np.nan, 20.0],
    })
    out, info = harmonize_source(raw.copy(), blank_means_not_applicable=True)
    assert list(out["P201"]) == [1.0, 2.0, 3.0]
    assert not any(c.endswith(("_x", "_y")) for c in out.columns)
    assert list(out["P506"]) == [0.0, 1.0, 0.0]          # vacio -> 0 (no aplica)
    assert pd.isna(out["edad_total"].iloc[1])            # la edad nunca se rellena
    assert info["columnas_merge_unificadas"] == ["P201"]


def test_armonizacion_no_toca_fuentes_con_ceros():
    from src.aligned.etl import harmonize_source
    raw = pd.DataFrame({"P506": [np.nan, 1.0]})
    out, _ = harmonize_source(raw.copy(), blank_means_not_applicable=False)
    assert pd.isna(out["P506"].iloc[0])


def test_balanceo_por_pesos_no_crea_muestras(cfg):
    from src.aligned.balancing import balance_training_set
    X = np.random.default_rng(0).random((60, 4)).astype(np.float32)
    y = np.array([0] * 10 + [1] * 20 + [2] * 30)
    Xb, yb, w, info = balance_training_set(X, y, cfg, 42)
    assert Xb.shape == X.shape and len(yb) == len(y)
    assert w is not None and w[0] > w[1] > w[2]          # la clase rara pesa mas
    assert info["metodo"] == "class_weights"


def test_balanceo_smote_solo_sobre_lo_que_recibe(cfg):
    import copy
    from src.aligned.balancing import balance_training_set
    c = copy.deepcopy(cfg)
    c["balancing"]["method"] = "smote"
    X = np.random.default_rng(0).random((60, 4)).astype(np.float32)
    y = np.array([0] * 10 + [1] * 20 + [2] * 30)
    Xb, yb, w, info = balance_training_set(X, y, c, 42)
    assert w is None
    assert np.bincount(yb).tolist() == [30, 30, 30]


def test_entrenamiento_sin_validacion_corre_epocas_exactas():
    from src.aligned.trainer import TorchTrainer, make_loader
    m = MLPMultiHeadAttention(num_features=6, n_heads=2, attn_dim=8)
    t = TorchTrainer(m, lr=1e-3, weight_decay=1e-4, device="cpu", amp=False)
    X = np.random.default_rng(0).random((40, 6)).astype(np.float32)
    y = np.random.default_rng(1).integers(0, 3, 40)
    h = t.fit(make_loader(X, y, 16, True), None, epochs=3, patience=1, verbose=False)
    assert len(h["train_acc"]) == 3 and h["val_acc"] == []
    assert t.epochs_run == 3


def test_tabnet_registra_curva_de_entrenamiento():
    pytest.importorskip("pytorch_tabnet")
    from src.aligned.models import TabNetModel
    rng = np.random.default_rng(0)
    X = rng.random((120, 5)).astype(np.float32)
    y = np.tile([0, 1, 2], 40)
    m = TabNetModel(num_features=5, n_d=8, n_a=8, device="cpu")
    h = m.fit(X[:90], y[:90], X[90:], y[90:], epochs=3, patience=5,
              batch_size=32, virtual_batch_size=16,
              class_weights=np.array([1.0, 1.0, 1.0]), track_train=True)
    assert len(h["train_acc"]) == len(h["val_acc"]) == 3
    assert 0 <= m.best_epoch < 3
    assert m.predict_proba(X[:4]).shape == (4, 3)


def test_logloss_ponderada_igual_a_la_perdida_de_entrenamiento():
    """El early stopping de TabNet usa la misma formula que nn.CrossEntropyLoss(weight)."""
    pytest.importorskip("pytorch_tabnet")
    from src.aligned.models import weighted_logloss_metric
    rng = np.random.default_rng(3)
    proba = rng.dirichlet([1, 1, 1], 200)
    y = rng.integers(0, 3, 200)
    w = np.array([2.5, 1.2, 0.56])
    metric = weighted_logloss_metric()
    metric.weights = w
    ours = metric()(y, proba)
    ref = torch.nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float64))(
        torch.log(torch.tensor(proba)), torch.tensor(y))
    assert ours == pytest.approx(float(ref), rel=1e-9)


def test_tabnet_detiene_con_la_perdida_ponderada():
    pytest.importorskip("pytorch_tabnet")
    from src.aligned.models import TabNetModel
    rng = np.random.default_rng(0)
    X = rng.random((150, 5)).astype(np.float32)
    y = np.array([0] * 20 + [1] * 40 + [2] * 90)
    m = TabNetModel(num_features=5, n_d=8, n_a=8, device="cpu")
    m.fit(X[:120], y[:120], X[120:], y[120:], epochs=4, patience=10,
          batch_size=32, virtual_batch_size=16,
          class_weights=np.array([2.5, 1.25, 0.55]), track_train=True)
    raw = m.model.history.history
    assert "val_wlogloss" in raw and "val_logloss" not in raw
    assert m.history["val_loss"] == list(raw["val_wlogloss"])
    assert m.best_epoch == int(np.argmin(raw["val_wlogloss"]))


def test_grid_de_tabnet_usa_su_propio_learning_rate(cfg):
    from src.aligned.gridsearch import build_grid
    import copy
    c = copy.deepcopy(cfg)
    c["gridsearch"]["preset"] = "standard"
    tab = build_grid(c, "tabnet")
    mlp = build_grid(c, "mlp_ma")
    assert {hp["lr"] for hp in tab} == set(c["gridsearch"]["presets"]["standard"]["lr_tabnet"])
    assert {hp["lr"] for hp in mlp} == set(c["gridsearch"]["presets"]["standard"]["lr"])
    # Sin configuraciones repetidas: TabNet no explora dropout.
    assert len({hp["dropout"] for hp in tab}) == 1
    assert len(tab) == len({tuple(sorted(hp.items())) for hp in tab}) == 8
    assert len(mlp) == 16


def test_ensemble_promedia_probabilidades(cfg, tmp_path):
    import copy
    from src.aligned.report import build_ensemble
    c = copy.deepcopy(cfg)
    c["paths"]["results_dir"] = "res"
    c["groups"] = {"jovenes": cfg["groups"]["jovenes"]}
    rng = np.random.default_rng(7)
    y = rng.integers(0, 3, 60)
    probas = {m: rng.dirichlet([1, 1, 1], 60) for m in ("mlp_ma", "tabnet", "ft_transformer")}
    results = {}
    for m, p in probas.items():
        d = tmp_path / "res" / f"jovenes_{m}"
        d.mkdir(parents=True)
        np.savez_compressed(d / "test_predictions.npz", y_true=y, y_pred=p.argmax(1), y_proba=p)
        results[f"jovenes_{m}"] = {"test_metrics": compute_metrics(y, p.argmax(1), p, 3)}
    ens, table = build_ensemble(results, c, tmp_path)
    expected = np.mean(list(probas.values()), axis=0).argmax(1)
    cm = np.array(ens["jovenes_ensemble"]["test_metrics"]["confusion_matrix"])
    assert cm.sum() == 60
    assert ens["jovenes_ensemble"]["test_metrics"]["accuracy"] == pytest.approx(np.mean(expected == y))
    assert len(table) == 1 and 0 <= table["Desacuerdo MLP-MA / TabNet"].iloc[0] <= 1

    # Si un modelo tiene otro conjunto de prueba, el ensemble no se calcula.
    np.savez_compressed(tmp_path / "res" / "jovenes_tabnet" / "test_predictions.npz",
                        y_true=np.roll(y, 1), y_pred=y, y_proba=probas["tabnet"])
    with pytest.raises(AssertionError, match="no es el mismo"):
        build_ensemble(results, c, tmp_path)


def test_etiquetas_legibles():
    assert V.feature_label("hh_nivel_educ_mean") == "Hogar: nivel educativo promedio"
    assert V.feature_label("P306") == "Último nivel educativo aprobado"
    assert V.feature_label("ZZZ_desconocida") == "ZZZ_desconocida"


def test_datos_procesados_sin_smote(cfg):
    """El train guardado conserva la distribucion natural (el balanceo va en el ajuste)."""
    base = ROOT / cfg["paths"]["processed_dir"]
    for group in cfg["groups"]:
        d = base / group
        if not (d / "train.parquet").exists():
            pytest.skip("Ejecuta primero: python run_pipeline.py etl")
        tr = pd.read_parquet(d / "train.parquet", columns=["target"])["target"]
        va = pd.read_parquet(d / "val.parquet", columns=["target"])["target"]
        p_tr = tr.value_counts(normalize=True).sort_index().to_numpy()
        p_va = va.value_counts(normalize=True).sort_index().to_numpy()
        assert np.allclose(p_tr, p_va, atol=0.02), f"{group}: train no estratificado/natural"


# =====================================================================
# Correcciones de mapeo (oct-2026) y embeddings para nominales
# =====================================================================
def test_p305_es_texto_y_se_excluye():
    assert V.is_free_text("P305")
    assert V.build_feature_list(["P305", "P304", "P303"]) == ["P304", "P303"]


def test_nominales_van_como_categoricas():
    cols = ["CCDD", "hh_etnia_jefe", "idx_lengua_materna_tipo",
            "hh_tamano", "idx_nivel_educativo", "P314", "ESTRATO"]
    split = V.classify_columns(cols)
    assert set(split["numeric"]) == {"hh_tamano", "idx_nivel_educativo", "P314"}
    assert set(split["categorical"]) == {"CCDD", "hh_etnia_jefe", "idx_lengua_materna_tipo",
                                         "ESTRATO"}
    assert "ESTRATO" not in V.NOMINAL_COLUMNS      # ordinal: sigue como escalar


def _hogar(rows):
    base = {"UBIGEO": 1, "CONG": 1, "VIVIENDA": 1, "HOGAR": 1, "P301": 10, "P303": 7,
            "P304": 2, "P311": 1, "P314": 0, "P309": 2, "P310_N": 0, "P514_1": 0}
    return pd.DataFrame([{**base, **r} for r in rows])


def test_alfabetismo_del_hogar_no_cuenta_no_aplica_como_analfabeto():
    from src.aligned.features import household_context
    df = _hogar([
        # jefa con secundaria completa: la ENL no le pregunta P307 (0) -> alfabeta
        {"edad_total": 40, "P203": 1, "P306": 6, "P307_1": 0, "P303": 1, "P311": 1},
        # adulto con primaria incompleta que no sabe leer
        {"edad_total": 38, "P203": 2, "P306": 3, "P307_1": 2, "P303": 7, "P311": 2},
        # mayor de 64 sin capitulo 300: no entra a los promedios
        {"edad_total": 70, "P203": 5, "P306": 0, "P307_1": 0, "P303": 0, "P311": 0},
        {"edad_total": 9, "P203": 3, "P306": 0, "P307_1": 0, "P303": 0, "P311": 0},
    ])
    hh = household_context(df, pd.Index([3]))
    assert hh.loc[3, "hh_alfabetismo"] == pytest.approx(0.5)
    assert hh.loc[3, "hh_tasa_ocupacion"] == pytest.approx(0.5)   # no 1/3
    assert hh.loc[3, "hh_etnia_jefe"] == 1                        # la del jefe/a, no la del 1.er adulto con dato
    assert hh.loc[3, "hh_n_adultos"] == 3


def test_log1p_hace_que_el_ingreso_pase_el_umbral_de_varianza():
    from src.aligned.etl import AlignedPreprocessor
    rng = np.random.default_rng(0)
    ingreso = np.where(rng.random(4000) < 0.4, 0, rng.lognormal(7, 0.6, 4000))
    ingreso[0] = 35000                                  # un caso extremo, como en la ENL
    X = pd.DataFrame({"P314": ingreso})
    plano = AlignedPreprocessor(["P314"], []).fit_transform(X)["P314"]
    con_log = AlignedPreprocessor(["P314"], [], log_columns=["P314"]).fit_transform(X)["P314"]
    assert plano.var(ddof=0) < 0.005 < con_log.var(ddof=0)
    assert con_log.between(0, 1).all()


@pytest.mark.parametrize("k", [2, 3, 9, 25])
def test_codigo_categorico_ida_y_vuelta(k):
    from src.aligned.models import category_codes
    codes = torch.arange(k)
    x = torch.as_tensor(codes.numpy() / max(k - 1, 1), dtype=torch.float32)
    assert torch.equal(category_codes(x, k), codes)


def test_embeddings_en_mlp_ma_y_ft_transformer():
    cat = [(1, 5)]
    x = torch.rand(6, 4)
    x[:, 1] = torch.tensor([0, .25, .5, .75, 1., .25])              # codigos 0-4
    for model in (MLPMultiHeadAttention(4, depth=2, cat_info=cat, token_out_dim=4),
                  FTTransformer(4, emb_dim=32, n_layers=1, n_heads=4, cat_info=cat)):
        model.eval()
        a, b = x.clone(), x.clone()
        b[:, 1] = 0.0                                                # solo cambia la categoria
        assert model(a).shape == (6, 3)
        assert not torch.allclose(model(a), model(b))
        model.train()
        model(a).sum().backward()
        tables = [m for m in model.modules() if isinstance(m, torch.nn.Embedding)]
        assert len(tables) == 1 and tables[0].weight.grad.abs().sum() > 0
    # La atencion sigue siendo de una fila y columna por variable.
    att = MLPMultiHeadAttention(4, cat_info=cat).attention_maps(x)
    assert att.shape == (4, 4, 4)


def test_tabnet_con_embeddings_reporta_por_variable_original():
    from src.aligned.models import TabNetModel
    rng = np.random.default_rng(1)
    X = rng.random((240, 4)).astype(np.float32)
    X[:, 2] = rng.integers(0, 6, 240) / 5                           # nominal de 6 categorias
    y = rng.integers(0, 3, 240)
    m = TabNetModel(4, n_steps=3, device="cpu", cat_info=[(2, 6)], seed=1)
    m.fit(X[:200], y[:200], X[200:], y[200:], epochs=2, patience=5, batch_size=64)
    assert m.predict_proba(X).shape == (240, 3)
    imp, masks = m.explain(X[:50])
    assert imp.shape == (50, 4) and all(v.shape == (50, 4) for v in masks.values())


def test_ensemble_por_semilla(cfg, tmp_path):
    import copy
    from src.aligned.report import build_ensemble
    c = copy.deepcopy(cfg)
    c["paths"]["results_dir"] = "res"
    c["groups"] = {"jovenes": cfg["groups"]["jovenes"]}
    rng = np.random.default_rng(3)
    y = rng.integers(0, 3, 80)
    seeds = np.array([42, 123])
    results, all_p = {}, {}
    for m in ("mlp_ma", "tabnet", "ft_transformer"):
        p = rng.dirichlet([1, 1, 1], (2, 80))
        all_p[m] = p
        d = tmp_path / "res" / f"jovenes_{m}"
        d.mkdir(parents=True)
        np.savez_compressed(d / "test_predictions.npz", y_true=y, y_pred=p[0].argmax(1), y_proba=p[0])
        np.savez_compressed(d / "test_predictions_seeds.npz", y_true=y, seeds=seeds, y_proba=p)
        runs = [{"seed": int(s), **compute_metrics(y, p[i].argmax(1), p[i], 3)} for i, s in enumerate(seeds)]
        results[f"jovenes_{m}"] = {"test_metrics": compute_metrics(y, p[0].argmax(1), p[0], 3),
                                   "estabilidad": {"runs": runs}}
    ens, table = build_ensemble(results, c, tmp_path)
    est = ens["jovenes_ensemble"]["estabilidad"]
    assert est["n_runs"] == 2
    acc = [np.mean(np.mean([all_p[m][i] for m in all_p], axis=0).argmax(1) == y) for i in range(2)]
    assert est["accuracy_mean"] == pytest.approx(np.mean(acc))
    assert table["Semillas en que el ensemble gana"].iloc[0].endswith("/2")


def test_copeland_usa_la_media_de_las_semillas(cfg):
    import copy
    from src.aligned.report import ranking_copeland
    c = copy.deepcopy(cfg)
    c["groups"] = {"ninos": cfg["groups"]["ninos"]}
    keys = cfg["evaluation"]["copeland_metrics"]

    def r(seed42, media):
        return {"test_metrics": {k: seed42 for k in keys},
                "estabilidad": {f"{k}_mean": media for k in keys}}
    # En la semilla 42 gana TabNet; en la media, FT-Transformer.
    results = {"ninos_mlp_ma": r(0.40, 0.40), "ninos_tabnet": r(0.60, 0.41),
               "ninos_ft_transformer": r(0.50, 0.45)}
    _, global_rank = ranking_copeland(results, c)
    assert global_rank.iloc[0]["classifier"] == "FT-Transformer"


def test_entrenamiento_rechaza_datos_de_un_etl_anterior(cfg, tmp_path):
    import copy, json
    from src.aligned.pipeline import load_group
    c = copy.deepcopy(cfg)
    c["paths"]["processed_dir"] = "proc"
    d = tmp_path / "proc" / "ninos"
    d.mkdir(parents=True)
    pd.DataFrame({"P209": [0.0, 1.0], "target": [0, 1]}).to_parquet(d / "train.parquet")
    (d / "config.json").write_text(json.dumps({"grupo": "ninos"}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="version anterior del ETL"):
        load_group(c, "ninos", tmp_path)


def test_adultos_son_de_18_a_64(cfg):
    assert (cfg["groups"]["adultos"]["edad_min"], cfg["groups"]["adultos"]["edad_max"]) == (18, 64)
