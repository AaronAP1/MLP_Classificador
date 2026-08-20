"""
Registro de variables de la ENL 2022.

Este modulo es la unica autoridad sobre que columna puede ser predictor y
cual no. La auditoria detecto que el ETL anterior declaraba una lista
`target_related` que nunca se aplicaba, dejando las variables que definen
la etiqueta dentro de la matriz de features. Aqui la exclusion es
estructural: `build_feature_list()` parte del universo de columnas y
elimina todo lo que caiga en la lista negra, y `assert_no_leakage()`
vuelve a comprobarlo despues del preprocesamiento.

Criterio de corte, derivado de los diccionarios de datos de la ENL 2022:

  Capitulo 200  (P201-P213)  demografico          -> PREDICTOR
  Capitulo 300  (P301-P315)  socioeconomico,
                             linguistico, educativo -> PREDICTOR
  Capitulo 400  (P401-P440)  practicas lectoras 18+ -> EXCLUIDO (define el target)
  Capitulo 500  (P501-P520)  practicas lectoras 0-17 -> EXCLUIDO salvo
                             P513/P513A/P514_* que son edad, lengua materna
                             y nivel educativo, no comportamiento lector.

Nota de correccion respecto al ETL anterior: P315_1..P315_6 NO son
"tipos de lectura" sino limitaciones permanentes (discapacidad), y P311
no es "lee por placer" sino si trabajo la semana pasada. El ETL previo
construia el target de adultos a partir de esas variables.
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Sequence

# ---------------------------------------------------------------------
# Identificadores y campos administrativos: ni predictores ni target.
# ---------------------------------------------------------------------
ADMIN_COLUMNS: List[str] = [
    "FECHA_CORTE", "PER", "CONG", "NSELV", "CCPP", "NOMBREPP", "CCDI",
    "NOMBREDI", "CCCP", "NOMBRECP", "UBIGEO", "VIVIENDA", "HOGAR",
    "RESFIN", "RESFIN_O", "RESFIN_V", "RESFIN_V_O", "INF_200", "INF_300",
    "INF_400", "INF_500", "PERS_NRO", "PERS_300", "PERS_400", "PERS_500",
    "TIPO_ENT", "TIPO_ENT_O", "RES_PER", "NOMBREDD",
    "P213",            # numero de celular: identificador, no atributo
    "_src",            # marca interna de procedencia del CSV
]

# ---------------------------------------------------------------------
# Variables que definen el comportamiento lector.
# Son la fuente del target y por tanto NUNCA pueden ser predictores.
# ---------------------------------------------------------------------
TARGET_SOURCE_COLUMNS: List[str] = [
    "P402", "P403",    # adultos: leyo en 12 meses + frecuencia
    "P506",            # 0-17: frecuencia de lectura de libros
]

# Patrones de exclusion por fuga. Se aplican sobre el nombre de columna.
LEAKAGE_PATTERNS: List[str] = [
    r"^P4\d{2}",       # capitulo 400 completo: practicas lectoras adultos
    r"^P50[1-9]",      # 501-509 practicas lectoras 0-17
    r"^P51[0257-9]",   # 510,512,515,517,518,519 practicas lectoras
    r"^P516",
    r"^P520",
    r"^P504A?_?\d*",   # 504/504A actividades de lectura del hogar
    r"^P60[12]",       # capitulo 600 (equipamiento/libros) si llegara a estar
    r"^lectura_",      # scores derivados de lectura
    r"^habito_",
]

# Excepciones explicitas dentro del rango 500: no son comportamiento lector.
LEAKAGE_EXCEPTIONS: List[str] = [
    "P513",            # edad de residentes 3-17
    "P513A",           # lengua materna (entorno linguistico)
    "P514_1",          # nivel educativo al que asiste
    "P514_2",          # año de estudios
    "P514_3",          # grado de estudios
]

# Columnas de texto libre / "otro (especifique)": alta cardinalidad, sin
# valor predictivo y con riesgo de filtrar la respuesta literal.
FREE_TEXT_PATTERN = r"_O$|_O_|_9_O$|_1_O$"

# ---------------------------------------------------------------------
# Agrupacion tematica de los predictores admitidos. Se usa para construir
# los indicadores derivados de la seccion III-B del articulo.
# ---------------------------------------------------------------------
DIMENSIONS: Dict[str, List[str]] = {
    "geografico": [
        "DOMINIO", "ESTRATO", "ESTRATOSOCIO", "CCDD", "TOT_HOGAR",
        "FACTORFINAL_HOGAR", "FACTOR200_FINAL",
    ],
    "demografico": [
        "P201", "P203", "P204", "P205", "P206", "P207", "P208", "P209",
        "P210_A", "P210_M", "P211", "P211_C", "P212", "edad_total",
    ],
    "linguistico": [
        "P301", "P302_1", "P302_2", "P302_3", "P302_4", "P302_5", "P302_6",
        "P302_7", "P302_8", "P302_9", "P302_10", "P302_11", "P302_12",
        "P302_13", "P302_14", "P302_15", "P303", "P307_1", "P307_2",
        "P308_1", "P308_2_1", "P308_2_2", "P308_2_3", "P308_2_4",
        "P308_2_5", "P308_2_6", "P308_2_7", "P308_2_8", "P308_2_9",
        "P513A",
    ],
    "educativo": [
        "P304", "P305", "P305_C", "P306", "P306_A", "P306_G", "P306_C",
        "P309", "P310_N", "P310_A", "P310_G", "P310_C",
        "P514_1", "P514_2", "P514_3",
    ],
    "socioeconomico": [
        "P311", "P312_1", "P312_2", "P312_3", "P312_4", "P312_5",
        "P312_6", "P312_7_8", "P313", "P314", "P314_1",
    ],
    "salud": [
        "P315_1", "P315_2", "P315_3", "P315_4", "P315_5", "P315_6",
    ],
}

# Variables tratadas como numericas continuas. El resto es categorico y
# recibe Label Encoding segun la seccion III-A.
NUMERIC_COLUMNS: List[str] = [
    "edad_total", "P210_A", "P210_M", "TOT_HOGAR", "FACTORFINAL_HOGAR",
    "FACTOR200_FINAL", "P314_1", "P306_A", "P310_A", "P514_2", "P513",
]


# =====================================================================
# API
# =====================================================================
def _matches_any(name: str, patterns: Iterable[str]) -> bool:
    return any(re.search(p, name) for p in patterns)


def is_leaky(column: str) -> bool:
    """True si la columna revela el comportamiento lector."""
    if column in LEAKAGE_EXCEPTIONS:
        return False
    if column in TARGET_SOURCE_COLUMNS:
        return True
    return _matches_any(column, LEAKAGE_PATTERNS)


def is_admin(column: str) -> bool:
    return column in ADMIN_COLUMNS


def is_free_text(column: str) -> bool:
    return bool(re.search(FREE_TEXT_PATTERN, column))


def build_feature_list(all_columns: Sequence[str]) -> List[str]:
    """
    Universo de columnas -> lista de predictores admitidos.

    El orden de los filtros importa: primero se descartan administrativas
    y texto libre, luego todo lo que huela a comportamiento lector.
    """
    features: List[str] = []
    for col in all_columns:
        if is_admin(col) or is_free_text(col):
            continue
        if is_leaky(col):
            continue
        if col.endswith(".1"):          # duplicados del merge de capitulos
            continue
        features.append(col)
    return features


def classify_columns(columns: Sequence[str]) -> Dict[str, List[str]]:
    """Separa los predictores en numericos y categoricos."""
    numeric = [c for c in columns if c in NUMERIC_COLUMNS]
    categorical = [c for c in columns if c not in NUMERIC_COLUMNS]
    return {"numeric": numeric, "categorical": categorical}


def dimension_of(column: str) -> str:
    """Dimension tematica de una variable, o 'otro'."""
    base = column.split("__")[0]
    for dim, cols in DIMENSIONS.items():
        if base in cols:
            return dim
    return "otro"


def assert_no_leakage(columns: Sequence[str]) -> None:
    """
    Verificacion dura. Se invoca despues del preprocesamiento y despues
    de la seleccion de features. Si algo se cuela, el pipeline se detiene
    en lugar de producir metricas infladas.
    """
    offenders = [c for c in columns if is_leaky(c.split("__")[0])]
    if offenders:
        raise AssertionError(
            "FUGA DEL TARGET detectada en la matriz de features: "
            f"{sorted(offenders)[:20]}"
            + (" ..." if len(offenders) > 20 else "")
            + "\nEstas variables definen el comportamiento lector y no "
              "pueden usarse como predictores."
        )
