#!/usr/bin/env python
"""
Punto de entrada unico del pipeline alineado al manuscrito.

    python run_pipeline.py check       Verifica entorno, config y datos
    python run_pipeline.py etl         Prepara los datos (rapido, sin GPU)
    python run_pipeline.py benchmark   Estima el tiempo de entrenamiento
    python run_pipeline.py train       Grid Search + CV + reentreno + test
    python run_pipeline.py report      Tablas y figuras del articulo
    python run_pipeline.py all         etl -> train -> report

Opciones utiles:
    --preset {fast,standard,paper}   Tamaño del Grid Search (Tabla 2)
    --groups ninos jovenes adultos   Subconjunto de grupos
    --models mlp_ma tabnet ft_transformer
    --no-gridsearch                  Salta la busqueda (solo para pruebas)
    --no-multiseed                   Salta las 5 semillas de estabilidad
"""

from __future__ import annotations

import argparse
import io
import sys
import time
from pathlib import Path

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.aligned.config import load_config           # noqa: E402
from src.aligned.models import MODEL_NAMES           # noqa: E402


def banner(text: str) -> None:
    print(f"\n{'=' * 70}\n  {text}\n{'=' * 70}")


# ---------------------------------------------------------------------
def cmd_check(cfg, args) -> int:
    banner("VERIFICACION DEL ENTORNO")
    ok = True

    print("\n  Dependencias:")
    for mod, label in [("torch", "PyTorch"), ("pytorch_tabnet", "pytorch-tabnet"),
                       ("sklearn", "scikit-learn"), ("imblearn", "imbalanced-learn"),
                       ("pandas", "pandas"), ("pyarrow", "pyarrow"),
                       ("matplotlib", "matplotlib"), ("yaml", "PyYAML")]:
        try:
            m = __import__(mod)
            print(f"    OK   {label:<18} {getattr(m, '__version__', '?')}")
        except ImportError:
            print(f"    FALTA {label:<18} -> pip install -r requirements.txt")
            ok = False

    try:
        import torch
        if torch.cuda.is_available():
            print(f"\n  GPU: {torch.cuda.get_device_name(0)} "
                  f"({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)")
        else:
            print("\n  GPU: no disponible; se entrenara en CPU (mas lento).")
    except ImportError:
        pass

    print("\n  Datos crudos:")
    raw = ROOT / cfg["paths"]["raw_dir"]
    for name in ("ninos", "jovenes", "adultos"):
        p = raw / f"{name}.csv"
        print(f"    {'OK  ' if p.exists() else 'FALTA'} {p.relative_to(ROOT)}")
        ok &= p.exists()

    print("\n  Datos procesados:")
    proc = ROOT / cfg["paths"]["processed_dir"]
    for group in cfg["groups"]:
        done = all((proc / group / f"{s}.parquet").exists()
                   for s in ("train", "val", "test"))
        print(f"    {'OK  ' if done else '--  '} {group}"
              f"{'' if done else '  (ejecuta: python run_pipeline.py etl)'}")

    print(f"\n  Configuracion validada contra el manuscrito: OK")
    print(f"    optimizador={cfg['training']['optimizer']} · "
          f"epocas={cfg['training']['final_epochs']} · "
          f"patience={cfg['training']['early_stopping_patience']}")
    print(f"    TabNet n_steps={cfg['models']['tabnet']['n_steps']} · "
          f"mask={cfg['models']['tabnet']['mask_type']}")
    print(f"    MLP+MA n_heads={cfg['models']['mlp_ma']['n_heads']} (multi-cabeza real)")
    print(f"    features objetivo={cfg['features']['target_n_features']}")

    print(f"\n  {'LISTO' if ok else 'FALTAN REQUISITOS'}")
    return 0 if ok else 1


def cmd_etl(cfg, args) -> int:
    from src.aligned.etl import run_etl
    banner("ETL · PREPARACION DE DATOS")
    started = time.time()
    run_etl(cfg, ROOT)
    print(f"\n  ETL completado en {time.time() - started:.1f} s")
    print(f"  Salida: {ROOT / cfg['paths']['processed_dir']}")
    return 0


def cmd_benchmark(cfg, args) -> int:
    from src.aligned.benchmark import run_benchmark
    banner("BENCHMARK · ESTIMACION DE TIEMPOS")
    run_benchmark(cfg, ROOT, groups=args.groups, models=args.models)
    return 0


def cmd_train(cfg, args) -> int:
    from src.aligned.pipeline import run_all
    banner("ENTRENAMIENTO · GRID SEARCH + CV + REENTRENO + TEST")
    print(f"  preset Grid Search : {cfg['gridsearch']['preset']}")
    print(f"  multi-semilla      : {cfg['evaluation']['stability']['multiseed']}")
    started = time.time()
    results = run_all(cfg, ROOT, args.groups, args.models,
                      do_gridsearch=not args.no_gridsearch)
    ok = sum(1 for v in results.values() if "error" not in v)
    print(f"\n  {ok}/{len(results)} combinaciones completadas "
          f"en {(time.time() - started) / 3600:.2f} h")
    for k, v in results.items():
        if "error" in v:
            print(f"    FALLO {k}: {v['error']}")
    return 0 if ok == len(results) else 1


def cmd_report(cfg, args) -> int:
    from src.aligned.report import build_report
    banner("REPORTE · TABLAS Y FIGURAS")
    info = build_report(cfg, ROOT)
    print(f"\n  Salida: {info['output_dir']}")
    return 0


def cmd_all(cfg, args) -> int:
    for fn in (cmd_etl, cmd_train, cmd_report):
        code = fn(cfg, args)
        if code != 0:
            return code
    return 0


# ---------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Pipeline alineado al manuscrito de la tesis.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("command",
                        choices=["check", "etl", "benchmark", "train", "report", "all"])
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--preset", choices=["fast", "standard", "paper"])
    parser.add_argument("--groups", nargs="+", default=None,
                        choices=["ninos", "jovenes", "adultos"])
    parser.add_argument("--models", nargs="+", default=None, choices=MODEL_NAMES)
    parser.add_argument("--no-gridsearch", action="store_true")
    parser.add_argument("--no-multiseed", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.preset:
        cfg["gridsearch"]["preset"] = args.preset
    if args.no_multiseed:
        cfg["evaluation"]["stability"]["multiseed"] = False
    if args.seed is not None:
        cfg["seed"] = args.seed

    return {
        "check": cmd_check, "etl": cmd_etl, "benchmark": cmd_benchmark,
        "train": cmd_train, "report": cmd_report, "all": cmd_all,
    }[args.command](cfg, args)


if __name__ == "__main__":
    raise SystemExit(main())
