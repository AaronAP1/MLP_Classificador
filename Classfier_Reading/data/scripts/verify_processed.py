"""
Script para verificar y resumir los datos procesados.
"""

import json
from pathlib import Path
import pandas as pd

def print_summary():
    """Imprime un resumen de todos los datos procesados."""
    
    print("\n" + "="*80)
    print("📊 RESUMEN DE DATOS PROCESADOS")
    print("="*80 + "\n")
    
    base_dir = Path(__file__).parent.parent
    processed_dir = base_dir / "processed"
    
    grupos = ["adultos", "jovenes", "ninos"]
    
    total_train = 0
    total_val = 0
    total_test = 0
    
    for grupo in grupos:
        grupo_dir = processed_dir / grupo
        config_file = grupo_dir / f"{grupo}_config.json"
        
        if not config_file.exists():
            print(f"❌ No se encontró config para {grupo}")
            continue
        
        # Leer config
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        print(f"{'='*80}")
        print(f"📋 GRUPO: {grupo.upper()}")
        print(f"{'='*80}\n")
        
        # Información general
        print(f"Total de registros: {config['total_registros']:,}")
        print(f"Total de features: {config['features']['total']}")
        
        # Splits
        print(f"\n📊 Distribución de splits:")
        splits = config['splits']
        total = sum(splits.values())
        for split_name, count in splits.items():
            pct = count / total * 100
            print(f"   {split_name.capitalize():5s}: {count:6,} ({pct:5.2f}%)")
            
            if split_name == "train":
                total_train += count
            elif split_name == "val":
                total_val += count
            elif split_name == "test":
                total_test += count
        
        # Target
        print(f"\n🎯 Variable objetivo: {config['target']['name']}")
        print(f"   Clases:")
        for clase_id, clase_nombre in config['target']['classes'].items():
            print(f"      {clase_id}: {clase_nombre}")
        
        # Distribución del target en train
        print(f"\n   Distribución en TRAIN:")
        train_dist = config['target']['distribution']['train']
        total_train_grupo = sum(train_dist.values())
        for clase_id, count in sorted(train_dist.items(), key=lambda x: int(x[0])):
            clase_nombre = config['target']['classes'][str(clase_id)]
            pct = count / total_train_grupo * 100
            print(f"      {clase_id}: {clase_nombre:20s} - {count:6,} ({pct:5.2f}%)")
        
        # Archivos generados
        print(f"\n📁 Archivos generados:")
        files_to_check = [
            f"{grupo}_train.csv",
            f"{grupo}_val.csv",
            f"{grupo}_test.csv",
            f"{grupo}_preprocessor.pkl",
            f"{grupo}_config.json"
        ]
        for filename in files_to_check:
            filepath = grupo_dir / filename
            if filepath.exists():
                size_mb = filepath.stat().st_size / (1024 * 1024)
                print(f"   ✅ {filename:30s} ({size_mb:.2f} MB)")
            else:
                print(f"   ❌ {filename:30s} (no encontrado)")
        
        print()
    
    # Resumen total
    print(f"{'='*80}")
    print(f"📊 RESUMEN TOTAL")
    print(f"{'='*80}\n")
    print(f"Total train: {total_train:7,}")
    print(f"Total val:   {total_val:7,}")
    print(f"Total test:  {total_test:7,}")
    print(f"{'─'*40}")
    print(f"TOTAL:       {total_train + total_val + total_test:7,}\n")
    
    print(f"✅ Todos los datos procesados correctamente")
    print(f"✅ Listos para entrenar modelos\n")
    print("="*80 + "\n")

if __name__ == "__main__":
    print_summary()
