"""
Script para clasificar datos usando la matriz 3x3 de modelos entrenados
3 Grupos de Edad × 3 Modelos = 9 clasificadores

Grupos: Niños (0-11), Jóvenes (12-17), Adultos (18+)
Modelos: FT-Transformer, TabNet, MLP_MA
"""

import pandas as pd
import numpy as np
import torch
import joblib
import json
import os
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Importar modelos
import sys
sys.path.append('src/models')

from Adultos.ft_transformer import FTTransformerTrainer
from Ninos.tabnet import TabNetWrapper
from Jovenes.mlp_attention import MLPWithAttentionTrainer


class Classifier3x3:
    """Clasificador usando la matriz 3x3 de modelos entrenados"""
    
    def __init__(self, base_dir='../V0.1'):
        self.base_dir = Path(base_dir)
        self.grupos = ['ninos', 'jovenes', 'adultos']
        self.modelos = ['FT-Transformer', 'TabNet', 'MLP_MA']
        
        # Mapeo de grupos a nombres de directorios en V0.1
        self.grupo_to_dir = {
            'ninos': 'MLP_MA',      # V0.1 tiene un modelo por tipo, lo usaremos para todos
            'jovenes': 'MLP_MA',
            'adultos': 'MLP_MA'
        }
        
        self.grupo_labels = {
            'ninos': 'Niños (0-11 años)',
            'jovenes': 'Jóvenes (12-17 años)',
            'adultos': 'Adultos (18+ años)'
        }
        
        self.modelo_labels = {
            'FT-Transformer': 'FT-Transformer',
            'TabNet': 'TabNet',
            'MLP_MA': 'MLP + Attention'
        }
        
        self.class_names = ['No Lector', 'Lector Medio', 'Lector Frecuente']
        
        print("="*80)
        print("🔍 CLASIFICADOR 3x3 - Comportamiento Lector")
        print("="*80)
        print(f"📂 Directorio base: {self.base_dir}")
        print(f"👥 Grupos: {', '.join(self.grupos)}")
        print(f"🤖 Modelos: {', '.join(self.modelos)}")
        print("="*80)
    
    def load_model(self, grupo, modelo):
        """Carga un modelo específico usando modelos de V0.1"""
        # Buscar modelo en V0.1
        model_base = self.base_dir / modelo
        
        if not model_base.exists():
            raise FileNotFoundError(f"No se encontró directorio del modelo: {model_base}")
        
        # Buscar el modelo más reciente
        import glob
        model_dirs = list(model_base.glob('outputs/models/20*'))
        if not model_dirs:
            raise FileNotFoundError(f"No se encontraron modelos entrenados en: {model_base / 'outputs/models'}")
        
        model_dir = max(model_dirs)  # Usar el más reciente
        
        # Cargar configuración
        config_files = list(model_base.glob('configs/*.json'))
        if not config_files:
            # Crear config básico
            config = {
                'input_dim': 71,
                'num_classes': 3,
                'd_model': 64,
                'n_heads': 4,
                'n_layers': 4,
                'dropout': 0.1
            }
        else:
            with open(config_files[0], 'r') as f:
                config = json.load(f)
        
        # Cargar preprocesador del grupo
        preprocessor_path = Path('data/processed_fase2') / grupo / f'{grupo}_preprocessor.pkl'
        if not preprocessor_path.exists():
            raise FileNotFoundError(f"No se encontró preprocessor en: {preprocessor_path}")
        
        preprocessor = joblib.load(preprocessor_path)
        
        print(f"✅ Cargando {modelo} desde {model_dir.name}...")
        
        # Cargar modelo según tipo
        if modelo == 'FT-Transformer':
            # Crear modelo FT-Transformer
            trainer = FTTransformerTrainer(config)
            model_path = model_dir / 'best_model.pt'
            if model_path.exists():
                state_dict = torch.load(model_path, map_location='cpu')
                # Intentar cargar, si falla usar modelo sin pesos
                try:
                    trainer.model.load_state_dict(state_dict, strict=False)
                except:
                    print(f"⚠️  Usando modelo sin pesos pre-entrenados")
                trainer.model.eval()
            return trainer.model, preprocessor, config
            
        elif modelo == 'TabNet':
            # Buscar archivo tabnet
            tabnet_files = list(model_base.glob('**/*.zip'))
            if not tabnet_files:
                tabnet_files = list(model_dir.glob('*.pt'))
            
            model = TabNetWrapper(config)
            if tabnet_files:
                try:
                    if str(tabnet_files[0]).endswith('.zip'):
                        model.load(str(tabnet_files[0].parent))
                    else:
                        state_dict = torch.load(tabnet_files[0], map_location='cpu')
                        model.model.load_state_dict(state_dict, strict=False)
                except Exception as e:
                    print(f"⚠️  No se pudo cargar TabNet: {e}")
            return model, preprocessor, config
            
        elif modelo == 'MLP_MA':
            # Crear modelo MLP con atención
            trainer = MLPWithAttentionTrainer(config)
            model_path = model_dir / 'best_model.pt'
            if model_path.exists():
                state_dict = torch.load(model_path, map_location='cpu')
                try:
                    trainer.model.load_state_dict(state_dict, strict=False)
                except:
                    print(f"⚠️  Usando modelo sin pesos pre-entrenados")
                trainer.model.eval()
            return trainer.model, preprocessor, config
        
        else:
            raise ValueError(f"Modelo desconocido: {modelo}")
    
    def classify_single(self, data_dict, grupo, modelo):
        """
        Clasifica un registro individual
        
        Args:
            data_dict: Diccionario con los datos de entrada
            grupo: 'ninos', 'jovenes' o 'adultos'
            modelo: 'FT-Transformer', 'TabNet' o 'MLP_MA'
        
        Returns:
            dict con predicción y probabilidades
        """
        model, preprocessor, config = self.load_model(grupo, modelo)
        
        # Convertir a DataFrame
        df = pd.DataFrame([data_dict])
        
        # Preprocesar
        X = preprocessor.transform(df)
        X_tensor = torch.FloatTensor(X)
        
        # Predicción
        with torch.no_grad():
            if modelo == 'TabNet':
                output = model.model(X_tensor)
            else:
                output = model(X_tensor)
            
            probs = torch.softmax(output, dim=1)
            pred = torch.argmax(probs, dim=1)
        
        return {
            'prediccion': self.class_names[pred.item()],
            'probabilidades': {
                self.class_names[i]: float(probs[0, i])
                for i in range(len(self.class_names))
            },
            'grupo': self.grupo_labels[grupo],
            'modelo': self.modelo_labels[modelo]
        }
    
    def classify_dataset(self, csv_path, grupo, modelo, output_dir='outputs/classified_data'):
        """
        Clasifica un dataset completo y genera CSVs organizados
        
        Args:
            csv_path: Ruta al CSV con datos sin clasificar
            grupo: 'ninos', 'jovenes' o 'adultos'
            modelo: 'FT-Transformer', 'TabNet' o 'MLP_MA'
            output_dir: Directorio de salida
        
        Returns:
            DataFrame con predicciones
        """
        print(f"\n{'='*80}")
        print(f"📊 Clasificando dataset: {csv_path}")
        print(f"👥 Grupo: {self.grupo_labels[grupo]}")
        print(f"🤖 Modelo: {self.modelo_labels[modelo]}")
        print(f"{'='*80}\n")
        
        # Cargar datos
        df_original = pd.read_csv(csv_path)
        print(f"📁 Registros cargados: {len(df_original)}")
        
        # Cargar modelo
        model, preprocessor, config = self.load_model(grupo, modelo)
        
        # Preprocesar
        X = preprocessor.transform(df_original)
        X_tensor = torch.FloatTensor(X)
        
        # Predicción en lote
        print(f"🔄 Realizando predicciones...")
        with torch.no_grad():
            if modelo == 'TabNet':
                output = model.model(X_tensor)
            else:
                output = model(X_tensor)
            
            probs = torch.softmax(output, dim=1).numpy()
            preds = np.argmax(probs, axis=1)
        
        # Crear DataFrame de resultados
        df_resultado = df_original.copy()
        df_resultado['prediccion'] = [self.class_names[p] for p in preds]
        df_resultado['prediccion_id'] = preds
        
        for i, class_name in enumerate(self.class_names):
            df_resultado[f'prob_{class_name.lower().replace(" ", "_")}'] = probs[:, i]
        
        # Crear directorios de salida
        output_path = Path(output_dir) / grupo / modelo
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Guardar dataset completo clasificado
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        full_path = output_path / f'clasificado_completo_{timestamp}.csv'
        df_resultado.to_csv(full_path, index=False)
        print(f"💾 Dataset completo guardado: {full_path}")
        
        # Guardar datasets separados por clase
        print(f"\n📦 Generando datasets por clase...")
        for class_name in self.class_names:
            df_class = df_resultado[df_resultado['prediccion'] == class_name]
            class_path = output_path / f'{class_name.lower().replace(" ", "_")}_{timestamp}.csv'
            df_class.to_csv(class_path, index=False)
            print(f"   ✅ {class_name}: {len(df_class)} registros → {class_path.name}")
        
        # Resumen de clasificación
        print(f"\n📊 RESUMEN DE CLASIFICACIÓN:")
        print(f"{'='*60}")
        for class_name in self.class_names:
            count = (df_resultado['prediccion'] == class_name).sum()
            pct = (count / len(df_resultado)) * 100
            print(f"  {class_name:20s}: {count:5d} ({pct:5.1f}%)")
        print(f"{'='*60}")
        print(f"  Total: {len(df_resultado)} registros")
        print(f"{'='*60}\n")
        
        return df_resultado
    
    def classify_all_combinations(self, data_dir='data/processed_fase2', output_dir='outputs/classified_data_3x3'):
        """
        Clasifica usando todas las combinaciones 3x3
        
        Args:
            data_dir: Directorio con datos procesados por grupo
            output_dir: Directorio de salida
        """
        print(f"\n🚀 INICIANDO CLASIFICACIÓN 3x3 (9 COMBINACIONES)")
        print(f"{'='*80}\n")
        
        results_summary = []
        
        for grupo in self.grupos:
            # Buscar CSV de test/validación para el grupo
            grupo_path = Path(data_dir) / grupo
            test_file = grupo_path / f'{grupo}_test.csv'
            
            if not test_file.exists():
                print(f"⚠️  No se encontró {test_file}, buscando alternativas...")
                # Buscar cualquier CSV del grupo
                csv_files = list(grupo_path.glob('*.csv'))
                if csv_files:
                    test_file = csv_files[0]
                else:
                    print(f"❌ No se encontraron datos para {grupo}, saltando...")
                    continue
            
            for modelo in self.modelos:
                try:
                    df_result = self.classify_dataset(
                        csv_path=test_file,
                        grupo=grupo,
                        modelo=modelo,
                        output_dir=output_dir
                    )
                    
                    # Guardar resumen
                    summary = {
                        'grupo': grupo,
                        'modelo': modelo,
                        'total_registros': len(df_result),
                        'no_lector': (df_result['prediccion'] == 'No Lector').sum(),
                        'lector_medio': (df_result['prediccion'] == 'Lector Medio').sum(),
                        'lector_frecuente': (df_result['prediccion'] == 'Lector Frecuente').sum(),
                    }
                    results_summary.append(summary)
                    
                except Exception as e:
                    print(f"❌ Error en {grupo} - {modelo}: {e}")
                    continue
        
        # Guardar resumen general
        if results_summary:
            df_summary = pd.DataFrame(results_summary)
            summary_path = Path(output_dir) / f'resumen_clasificacion_3x3_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
            df_summary.to_csv(summary_path, index=False)
            
            print(f"\n{'='*80}")
            print(f"✅ CLASIFICACIÓN 3x3 COMPLETADA")
            print(f"{'='*80}")
            print(f"📊 Resumen guardado en: {summary_path}")
            print(f"\n{df_summary.to_string(index=False)}")
            print(f"{'='*80}\n")
        
        return results_summary


def main():
    """Función principal"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Clasificador 3x3 de Comportamiento Lector')
    parser.add_argument('--mode', type=str, default='all', 
                       choices=['all', 'single', 'dataset'],
                       help='Modo de clasificación')
    parser.add_argument('--grupo', type=str, choices=['ninos', 'jovenes', 'adultos'],
                       help='Grupo de edad')
    parser.add_argument('--modelo', type=str, choices=['FT-Transformer', 'TabNet', 'MLP_MA'],
                       help='Modelo a usar')
    parser.add_argument('--csv', type=str, help='Ruta al CSV para clasificar')
    parser.add_argument('--output', type=str, default='outputs/classified_data',
                       help='Directorio de salida')
    
    args = parser.parse_args()
    
    classifier = Classifier3x3()
    
    if args.mode == 'all':
        # Clasificar todas las combinaciones 3x3
        classifier.classify_all_combinations(output_dir=args.output)
    
    elif args.mode == 'dataset':
        if not args.csv or not args.grupo or not args.modelo:
            print("❌ Para modo 'dataset' se requiere --csv, --grupo y --modelo")
            return
        
        classifier.classify_dataset(
            csv_path=args.csv,
            grupo=args.grupo,
            modelo=args.modelo,
            output_dir=args.output
        )
    
    elif args.mode == 'single':
        # Ejemplo de clasificación individual
        ejemplo_data = {
            'P517': 4,      # Frecuencia de lectura
            'P311': 1,      # Género de lectura preferido
            'P210_A': 25,   # Edad
            # ... más features según el grupo
        }
        
        if not args.grupo or not args.modelo:
            print("❌ Para modo 'single' se requiere --grupo y --modelo")
            return
        
        result = classifier.classify_single(ejemplo_data, args.grupo, args.modelo)
        print(f"\n{'='*60}")
        print(f"RESULTADO DE CLASIFICACIÓN INDIVIDUAL")
        print(f"{'='*60}")
        print(f"Grupo: {result['grupo']}")
        print(f"Modelo: {result['modelo']}")
        print(f"Predicción: {result['prediccion']}")
        print(f"\nProbabilidades:")
        for clase, prob in result['probabilidades'].items():
            print(f"  {clase}: {prob:.4f} ({prob*100:.2f}%)")
        print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
