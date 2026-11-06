"""
Construccion de la variable objetivo `comportamiento_lector`.

Tres clases, segun la seccion III del articulo:
    0 = No Lector
    1 = Lector Medio
    2 = Lector Frecuente

A diferencia del ETL anterior, aqui la definicion es explicita, esta
anclada a las escalas reales del cuestionario ENL 2022 y no depende de
umbrales arbitrarios sobre un score compuesto sin documentar.

Escalas usadas (verificadas contra los diccionarios de datos):

  P506 - "En los ultimos 12 meses, con que frecuencia leyo o le leyeron
          libros impresos o digitales" (poblacion 0-17)
      1 Diariamente              -> 2 Lector Frecuente
      2 Varias veces a la semana -> 2 Lector Frecuente
      3 Una vez a la semana      -> 1 Lector Medio
      4 Varias veces al mes      -> 1 Lector Medio
      5 Una vez al mes           -> 1 Lector Medio
      6 Una vez cada tres meses  -> 0 No Lector
      7 Por lo menos una vez al año -> 0 No Lector
      8 No leyo / no le leyeron  -> 0 No Lector
      9 NS/NR                    -> descartado

  P408 - "Ha leido algun libro impreso y/o digital"  (18+)   [por defecto]
      1 Si  -> se resuelve con P409
      2 No  -> 0 No Lector
  P409 - "Con que frecuencia" (lectura de libros, 18+)
      1 Diariamente              -> 2 Lector Frecuente
      2 Varias veces a la semana -> 2 Lector Frecuente
      3 Una vez a la semana      -> 1 Lector Medio
      4 Una vez al mes           -> 1 Lector Medio
      5 Una vez cada tres meses  -> 0 No Lector
      6 Por lo menos una vez al año -> 0 No Lector

  P402/P403 - misma escala, pero sobre "libros, periodicos, revistas u
  otros contenidos" (cualquier lectura). Disponible como alternativa
  (`chapter_300_400`).

Por que P408/P409 por defecto: P506 (0-17) pregunta por LIBROS. Para que
las tres clases signifiquen lo mismo en los tres grupos etarios, los
adultos tambien se clasifican por lectura de libros. Es ademas el
indicador oficial que reporta la ENL 2022 ("leyo libros": 47,3 %).
Con P402/P403, el 67 % de los adultos cae en "Lector Frecuente" porque
cuenta cualquier contenido leido, y la clase deja de ser comparable con
la de los menores.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

NO_LECTOR, LECTOR_MEDIO, LECTOR_FRECUENTE = 0, 1, 2

# P506 (0-17)
MAP_P506: Dict[float, int] = {
    1.0: LECTOR_FRECUENTE,
    2.0: LECTOR_FRECUENTE,
    3.0: LECTOR_MEDIO,
    4.0: LECTOR_MEDIO,
    5.0: LECTOR_MEDIO,
    6.0: NO_LECTOR,
    7.0: NO_LECTOR,
    8.0: NO_LECTOR,
    # 9 = NS/NR -> NaN
}

# P403 (18+), condicionado a P402 == 1
MAP_P403: Dict[float, int] = {
    1.0: LECTOR_FRECUENTE,
    2.0: LECTOR_FRECUENTE,
    3.0: LECTOR_MEDIO,
    4.0: LECTOR_MEDIO,
    5.0: NO_LECTOR,
    6.0: NO_LECTOR,
}

# P409 (18+), condicionado a P408 == 1. Misma escala que P403.
MAP_P409: Dict[float, int] = dict(MAP_P403)

# Fuentes admitidas en configs/pipeline.yaml -> groups.<grupo>.target_source
TARGET_SOURCES = {
    "chapter_500": "P506 · frecuencia de lectura de libros (0-17)",
    "chapter_400_libros": "P408 + P409 · lectura de libros (18+)",
    "chapter_300_400": "P402 + P403 · cualquier lectura (18+)",
}


def _clean(series: pd.Series) -> pd.Series:
    """0 en la ENL significa 'no aplica el capitulo'; se trata como nulo."""
    return series.replace(0, np.nan)


def build_target(df: pd.DataFrame, source: str) -> pd.Series:
    """
    Devuelve la Serie `comportamiento_lector` (float con NaN donde no se
    puede determinar). El llamador decide si descarta esas filas.

    Args:
        df: DataFrame con las columnas crudas de la ENL.
        source: una de las claves de `TARGET_SOURCES`.
    """
    if source == "chapter_500":
        if "P506" not in df.columns:
            raise KeyError("Falta P506, requerida para el target de 0-17.")
        return _clean(df["P506"]).map(MAP_P506).astype("float64")

    if source == "chapter_400_libros":
        return _leyo_y_frecuencia(df, "P408", "P409", MAP_P409)

    if source == "chapter_300_400":
        return _leyo_y_frecuencia(df, "P402", "P403", MAP_P403)

    raise ValueError(f"target_source desconocido: {source!r}. "
                     f"Opciones: {sorted(TARGET_SOURCES)}")


def _leyo_y_frecuencia(df: pd.DataFrame, leyo_col: str, freq_col: str,
                       mapping: Dict[float, int]) -> pd.Series:
    """Clase a partir de 'leyo en 12 meses' (1 Si / 2 No) + su frecuencia."""
    for col in (leyo_col, freq_col):
        if col not in df.columns:
            raise KeyError(f"Falta {col}, requerida para el target de 18+.")

    leyo_raw = _clean(df[leyo_col])
    freq = _clean(df[freq_col])

    target = pd.Series(np.nan, index=df.index, dtype="float64")
    # Declara no haber leido en 12 meses -> No Lector, sin importar la frecuencia.
    target[leyo_raw == 2.0] = NO_LECTOR
    # Declara haber leido -> la clase la define la frecuencia.
    leyo = leyo_raw == 1.0
    target[leyo] = freq[leyo].map(mapping)
    return target


def target_report(y: pd.Series, class_names) -> Dict[str, object]:
    """Resumen de la distribucion de clases, para el log y el config JSON."""
    valid = y.dropna().astype(int)
    counts = valid.value_counts().reindex(range(len(class_names)), fill_value=0)
    total = int(counts.sum())
    return {
        "n_valid": total,
        "n_discarded": int(y.isna().sum()),
        "counts": {class_names[i]: int(counts[i]) for i in range(len(class_names))},
        "ratios": {
            class_names[i]: round(float(counts[i]) / total, 6) if total else 0.0
            for i in range(len(class_names))
        },
        "imbalance_ratio": (
            round(float(counts.max()) / float(counts.min()), 2)
            if counts.min() > 0 else None
        ),
        "min_class_count": int(counts.min()),
    }
