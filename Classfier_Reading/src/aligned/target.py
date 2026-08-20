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

  P402 - "Ha leido libros, periodicos, revistas u otros contenidos"  (18+)
      1 Si  -> se resuelve con P403
      2 No  -> 0 No Lector
  P403 - "Con que frecuencia"  (18+)
      1 Diariamente              -> 2 Lector Frecuente
      2 Varias veces a la semana -> 2 Lector Frecuente
      3 Una vez a la semana      -> 1 Lector Medio
      4 Una vez al mes           -> 1 Lector Medio
      5 Una vez cada tres meses  -> 0 No Lector
      6 Por lo menos una vez al año -> 0 No Lector
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


def _clean(series: pd.Series) -> pd.Series:
    """0 en la ENL significa 'no aplica el capitulo'; se trata como nulo."""
    return series.replace(0, np.nan)


def build_target(df: pd.DataFrame, source: str) -> pd.Series:
    """
    Devuelve la Serie `comportamiento_lector` (float con NaN donde no se
    puede determinar). El llamador decide si descarta esas filas.

    Args:
        df: DataFrame con las columnas crudas de la ENL.
        source: "chapter_500" (0-17) o "chapter_300_400" (18+).
    """
    if source == "chapter_500":
        if "P506" not in df.columns:
            raise KeyError("Falta P506, requerida para el target de 0-17.")
        return _clean(df["P506"]).map(MAP_P506).astype("float64")

    if source == "chapter_300_400":
        for col in ("P402", "P403"):
            if col not in df.columns:
                raise KeyError(f"Falta {col}, requerida para el target de 18+.")

        p402 = _clean(df["P402"])
        p403 = _clean(df["P403"])

        target = pd.Series(np.nan, index=df.index, dtype="float64")
        # Declara no haber leido en 12 meses -> No Lector, sin importar P403.
        target[p402 == 2.0] = NO_LECTOR
        # Declara haber leido -> la clase la define la frecuencia.
        leyo = p402 == 1.0
        target[leyo] = p403[leyo].map(MAP_P403)
        return target

    raise ValueError(f"target_source desconocido: {source!r}")


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
