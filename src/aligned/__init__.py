"""
Pipeline alineado al manuscrito de la tesis.

Este paquete reemplaza al conjunto de scripts auditado. Sus garantias:

  * Ningun hiperparametro incrustado: todo sale de `configs/pipeline.yaml`.
  * Ninguna ruta genera datos, metricas ni predicciones sinteticas.
  * La fuga del target se bloquea de forma estructural en `variables.py`
    y se vuelve a verificar tras el preprocesamiento y antes de entrenar.
  * `config.validate()` falla si la configuracion contradice una
    afirmacion explicita del articulo.

Uso:
    python run_pipeline.py etl
    python run_pipeline.py train
    python run_pipeline.py report
"""

from .config import load_config, project_root, validate

__all__ = ["load_config", "project_root", "validate"]
__version__ = "1.0.0"
