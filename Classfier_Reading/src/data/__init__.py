"""
Módulo de procesamiento de datos.

Este módulo contiene todas las funcionalidades para:
- Carga y limpieza de datos
- Creación de variables objetivo
- Split de datasets
- Preprocessing (encoding + scaling)
- Gestión de dataloaders
"""

from .etl import (
    load_metadata_mappings,
    select_features,
    create_target_variable,
    split_data,
    preprocess_features,
    save_processed_data,
    run_etl_pipeline
)

__all__ = [
    'load_metadata_mappings',
    'select_features',
    'create_target_variable',
    'split_data',
    'preprocess_features',
    'save_processed_data',
    'run_etl_pipeline'
]
