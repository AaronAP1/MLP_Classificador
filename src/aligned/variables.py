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
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

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
    "P201",            # numero de orden de la persona en el hogar
    "P204",            # numero del nucleo familiar
    "P211_E",          # nacionalidad (texto libre)
    # Factores de expansion del diseño muestral: corrigen la representatividad
    # de la muestra, no describen a la persona. Usarlos como predictores
    # mete el diseño de la encuesta dentro del modelo.
    "FACTORFINAL_HOGAR", "FACTOR200_FINAL",
    "_src",            # marca interna de procedencia del CSV
]

# ---------------------------------------------------------------------
# Variables que definen el comportamiento lector.
# Son la fuente del target y por tanto NUNCA pueden ser predictores.
# ---------------------------------------------------------------------
TARGET_SOURCE_COLUMNS: List[str] = [
    "P402", "P403",    # adultos: leyo cualquier contenido en 12 meses + frecuencia
    "P408", "P409",    # adultos: leyo libros en 12 meses + frecuencia
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

# Respuestas alfanumericas sin codificar. P305 guarda el NOMBRE del pueblo
# indigena ("QUECHUAS", "AIMARA"...): al convertirla a numero el texto se
# volvia "sin dato" (-1), y como jovenes.csv la trae vacia, en la practica
# marcaba el archivo de origen de cada persona. La pertenencia a un pueblo
# ya la recoge P304 (si / no / no sabe).
TEXT_COLUMNS: List[str] = ["P305"]

# ---------------------------------------------------------------------
# Agrupacion tematica de los predictores admitidos. Se usa para construir
# los indicadores derivados de la seccion III-B del articulo.
# ---------------------------------------------------------------------
DIMENSIONS: Dict[str, List[str]] = {
    "geografico": [
        "DOMINIO", "ESTRATO", "ESTRATOSOCIO", "CCDD", "TOT_HOGAR",
    ],
    "demografico": [
        "P203", "P205", "P206", "P207", "P208", "P209",
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
# recibe Label Encoding segun la seccion III-A. Los indicadores derivados
# (idx_*, hh_*) son numericos salvo los que figuran en NOMINAL_COLUMNS.
NUMERIC_COLUMNS: List[str] = [
    "edad_total", "P210_A", "P210_M", "TOT_HOGAR",
    "P314",            # monto del ingreso mensual (soles). P314_1 es solo
                       # la marca "no sabe / no tuvo ganancia": categorica.
    "P306_A", "P310_A", "P514_2", "P513",
]

# Categoricas SIN orden: el codigo es una etiqueta (Amazonas=1, Ancash=2...),
# no una magnitud. Con Label Encoding entran como un numero de 0 a 1 y la
# red tendria que "deshacer" un orden arbitrario; por eso cada una recibe un
# embedding propio (un vector aprendido por categoria) en las tres
# arquitecturas. En una prueba con un MLP simple, representarlas como
# categorias subio el F1-Macro en validacion +0,05 (niños), +0,06
# (jovenes) y +0,01 (adultos). Las categoricas ordinales (estrato por
# tamaño de centro poblado, grado, nivel educativo) y las binarias siguen
# como escalar, que ya conserva toda su informacion.
NOMINAL_COLUMNS: List[str] = [
    "CCDD",            # departamento
    "DOMINIO",         # dominio geografico
    "ESTRATOSOCIO",    # estrato socioeconomico (incluye la categoria rural)
    "P203", "P205",    # parentesco con el jefe de hogar / de nucleo
    "P212",            # estado civil
    "P301", "P513A",   # lengua materna (18+ / 3-17)
    "P303",            # autoidentificacion etnica
    "P304",            # pertenencia a pueblo indigena (si / no / no sabe)
    "P306_C", "P310_C",  # tipo de centro de estudios
    "P307_1", "P307_2",  # sabe leer y escribir (no aplica / si / no)
    "P310_N", "P514_1",  # nivel que cursa (mezcla niveles y modalidades)
    "P313",            # categoria ocupacional
    "hh_etnia_jefe",   # etnia del jefe de hogar
    "idx_lengua_materna_tipo",  # castellano / originaria / extranjera / otra
]

# Etiquetas legibles para tablas y figuras. Las variables de la ENL que no
# esten aqui se rotulan con su descripcion del diccionario de datos
# (`feature_label`), y si tampoco estan alli, con su codigo.
FEATURE_LABELS: Dict[str, str] = {
    "edad_total": "Edad",
    "P210_A": "Edad (años)",
    "P210_M": "Edad (meses)",
    "P203": "Parentesco con jefe/a de hogar",
    "P205": "Parentesco con jefe/a de núcleo",
    "P209": "Sexo",
    "P211": "Nacionalidad",
    "P212": "Estado civil",
    "DOMINIO": "Dominio geográfico",
    "ESTRATO": "Estrato geográfico (tamaño de centro poblado)",
    "ESTRATOSOCIO": "Estrato socioeconómico",
    "CCDD": "Departamento",
    "TOT_HOGAR": "Número de hogares en la vivienda",
    "P301": "Lengua materna",
    "P303": "Autoidentificación étnica",
    "P304": "Pertenencia a pueblo indígena u originario",
    "P306": "Último nivel educativo aprobado",
    "P306_A": "Último año aprobado",
    "P306_G": "Último grado aprobado",
    "P306_C": "Tipo de centro de estudios",
    "P307_1": "Sabe leer y escribir (espontánea)",
    "P307_2": "Sabe leer y escribir (cartilla)",
    "P308_1": "Lee y escribe en lengua originaria",
    "P305": "Pueblo indígena u originario",
    "P309": "Asiste a educación actualmente",
    "P310_N": "Nivel que cursa actualmente",
    "P310_A": "Año que cursa actualmente",
    "P310_G": "Grado que cursa actualmente",
    "P310_C": "Tipo de centro (estudios actuales)",
    "P311": "Trabajó la semana pasada",
    "P313": "Categoría ocupacional",
    "P314": "Ingreso mensual (ocupación principal)",
    "P314_1": "Sin ganancia / no sabe su ingreso",
    "P513A": "Lengua materna (3-17 años)",
    "P514_1": "Nivel educativo al que asiste",
    "P514_2": "Año de estudios",
    "P514_3": "Grado de estudios",
    "idx_acceso_digital": "Índice de acceso digital (aprox.)",
    "idx_nivel_educativo": "Nivel educativo (índice)",
    "idx_lengua_materna_tipo": "Tipo de lengua materna",
    "idx_n_lenguas_habladas": "N.º de lenguas que habla",
    "hh_nivel_educ_max": "Hogar: nivel educativo máximo",
    "hh_nivel_educ_mean": "Hogar: nivel educativo promedio",
    "hh_nivel_educ_jefe": "Hogar: nivel educativo del jefe/a",
    "hh_alfabetismo": "Hogar: alfabetismo adulto",
    "hh_lengua_indigena": "Hogar: adultos con lengua originaria",
    "hh_pueblo_indigena": "Hogar: adultos de pueblo originario",
    "hh_etnia_jefe": "Hogar: etnia del jefe/a",
    "hh_tasa_ocupacion": "Hogar: tasa de ocupación",
    "hh_ingreso_mean": "Hogar: ingreso promedio",
    "hh_ingreso_max": "Hogar: ingreso máximo",
    "hh_estudia": "Hogar: adultos que estudian",
    "hh_limitacion": "Hogar: alguna limitación permanente",
    "hh_n_adultos": "Hogar: n.º de adultos",
    "hh_tamano": "Hogar: tamaño",
    "hh_n_menores": "Hogar: n.º de menores",
    "hh_ratio_menores": "Hogar: proporción de menores",
}

# Preguntas de opcion multiple (una columna por opcion).
_LENGUAS = {1: "quechua", 2: "aimara", 3: "asháninka", 4: "awajún", 5: "shipibo-konibo",
            6: "shawi", 7: "matsigenka", 8: "achuar", 9: "otra originaria",
            10: "castellano", 11: "portugués", 12: "inglés", 13: "otra extranjera",
            14: "no escucha ni habla", 15: "lengua de señas"}
_OPTION_FAMILIES = {
    "P302": ("Habla", _LENGUAS),
    "P308_2": ("Lee y escribe en", _LENGUAS),
    "P312": ("Semana pasada", {1: "tenía trabajo sin trabajar", 2: "negocio propio",
                               3: "trabajo ocasional", 4: "chacra o crianza",
                               5: "ayudó en negocio familiar", 6: "otra actividad"}),
    "P315": ("Limitación permanente para", {1: "ver", 2: "oír", 3: "comunicarse",
                                            4: "moverse o caminar", 5: "entender o aprender",
                                            6: "relacionarse"}),
}
FEATURE_LABELS["P312_7_8"] = "Semana pasada: cuidado del hogar / no trabajó"
for _prefix, (_stem, _opts) in _OPTION_FAMILIES.items():
    for _k, _name in _opts.items():
        FEATURE_LABELS.setdefault(f"{_prefix}_{_k}", f"{_stem}: {_name}")

_DICTIONARY_CACHE: Optional[Dict[str, str]] = None


def _load_dictionary_labels() -> Dict[str, str]:
    """Descripcion corta de cada codigo ENL, leida de data/metadata/*.xlsx."""
    global _DICTIONARY_CACHE
    if _DICTIONARY_CACHE is not None:
        return _DICTIONARY_CACHE
    labels: Dict[str, str] = {}
    meta = Path(__file__).resolve().parents[2] / "data" / "metadata"
    try:
        import pandas as pd
        for path in sorted(meta.glob("Diccionario*.xlsx")):
            frame = pd.read_excel(path, header=None).fillna("")
            for row in frame.itertuples(index=False):
                code, desc = str(row[0]).strip(), str(row[1]).strip()
                if re.fullmatch(r"P\d{3}[A-Z]?(_\w+)?", code) and desc:
                    # "306. ¿CUAL ES EL ULTIMO ...? 1. Sin nivel ..." -> pregunta sola
                    text = re.split(r"\?|\n|\s\d\.\s", desc)[0]
                    text = re.sub(r"^\d+\s*[\.\-]?\s*[A-Z]?\.?\s*", "", text).strip(" ¿:.")
                    labels.setdefault(code, text.capitalize()[:48])
    except Exception:                                     # noqa: BLE001
        pass
    _DICTIONARY_CACHE = labels
    return labels


def feature_label(column: str) -> str:
    """Nombre legible de una variable para tablas y figuras."""
    base = column.split("__")[0]
    if base in FEATURE_LABELS:
        return FEATURE_LABELS[base]
    desc = _load_dictionary_labels().get(base)
    return f"{desc} ({base})" if desc else base


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
    return column in TEXT_COLUMNS or bool(re.search(FREE_TEXT_PATTERN, column))


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


def is_numeric_column(column: str) -> bool:
    """Numerica continua: la lista explicita y los indicadores derivados no nominales."""
    if column in NOMINAL_COLUMNS:
        return False
    return column in NUMERIC_COLUMNS or column.startswith(("idx_", "hh_"))


def classify_columns(columns: Sequence[str]) -> Dict[str, List[str]]:
    """Separa los predictores en numericos y categoricos."""
    numeric = [c for c in columns if is_numeric_column(c)]
    categorical = [c for c in columns if not is_numeric_column(c)]
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
