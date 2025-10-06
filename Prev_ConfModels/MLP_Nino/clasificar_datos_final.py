#!/usr/bin/env python3
"""
Script para clasificar hábitos de lectura usando modelo entrenado
y generar datasets organizados por categoría.
"""

import pandas as pd
import numpy as np
import torch
import joblib
import json
import os
from datetime import datetime
import glob

def clasificar_y_organizar_datos(csv_path="data/raw/jovenes.csv", model_path=None, 
                                generar_datasets_separados=True):
    """
    Clasifica datos y organiza en datasets separados por categoria de lectura.
    
    Args:
        csv_path: Ruta al CSV con los datos a clasificar
        model_path: Ruta al modelo entrenado (si None, usa el mas reciente)
        generar_datasets_separados: Si True, genera CSVs separados por categoria
    """
    
    # Cargar config
    with open('configs/config_jovenes.json', 'r') as f:
        config = json.load(f)
    
    # Si no se especifica modelo, usar el mas reciente
    if model_path is None:
        import glob
        model_dirs = glob.glob("outputs/models/202*")
        latest_dir = max(model_dirs)
        model_path = f"{latest_dir}/best_model.pt"
    
    print(f"*** CLASIFICADOR Y ORGANIZADOR DE HABITOS DE LECTURA ***")
    print(f"Modelo: {model_path}")
    print("=" * 70)
    
    # Cargar datos originales
    df_original = pd.read_csv(csv_path)
    print(f"Datos cargados: {df_original.shape[0]} personas, {df_original.shape[1]} columnas")
    
    # Cargar preprocesadores
    preprocessor = joblib.load("data/processed/preprocessor.pkl")
    label_encoder = joblib.load("data/processed/label_encoder.pkl")
    print(f"Preprocesadores cargados correctamente")
    
    # Crear target de lectura
    from utils.etl import crear_target_lectura
    df_with_target = crear_target_lectura(df_original)
    print(f"Target de lectura creado: {len(df_with_target)} muestras")
    
    # Extraer características
    feature_cols = config["features"]["selected_features"]
    X = df_with_target[feature_cols].copy()
    
    # Limpiar datos (reemplazar NaN e Inf)
    # Para numericas: usar la mediana
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if X[col].isna().any():
            median_val = X[col].median()
            if pd.isna(median_val):  # Si toda la columna es NaN
                median_val = 0
            X[col] = X[col].fillna(median_val)
            print(f"Columna {col}: NaN reemplazados con {median_val}")
        
        # Reemplazar infinitos
        X[col] = X[col].replace([np.inf, -np.inf], X[col].median())
    
    # Para categoricas: usar 'Desconocido'
    categorical_cols = X.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        X[col] = X[col].fillna('Desconocido')
        print(f"Columna {col}: NaN reemplazados con 'Desconocido'")
    
    print(f"Datos limpios: {len(X)} muestras")
    
    # Aplicar preprocesamiento
    X_processed = preprocessor.transform(X)
    print(f"Datos preprocesados: {X_processed.shape}")
    
    # Configurar modelo
    input_size = X_processed.shape[1]
    n_classes = len(label_encoder.classes_)
    
    print(f"Input size: {input_size}")
    print(f"Número de clases: {n_classes}")
    print(f"Clases: {label_encoder.classes_}")
    
    # Cargar modelo
    from models.model_jovenes import MLPModel
    
    model_config = config["model"]
    model = MLPModel(
        input_dim=input_size,
        num_classes=n_classes,
        hidden_dims=model_config["hidden_dims"],
        dropout_rate=model_config["dropout_rate"],
        attention_dim=model_config.get("attention_dim", 64)
    )
    
    # Cargar pesos
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    print(f"Modelo cargado en dispositivo: {device}")
    
    # Hacer predicciones
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_processed).to(device)
        outputs = model(X_tensor)
        probabilities = torch.softmax(outputs, dim=1).cpu().numpy()
        predicted_indices = torch.argmax(outputs, dim=1).cpu().numpy()
        
        # Convertir a etiquetas
        predicted_labels = label_encoder.inverse_transform(predicted_indices)
        confidence_scores = np.max(probabilities, axis=1)
    
    print(f"Predicciones completadas para {len(predicted_labels)} muestras")
    
    # Crear DataFrame con resultados
    df_clasificado = df_original.copy()
    df_clasificado['prediccion_habito'] = predicted_labels
    df_clasificado['confianza'] = confidence_scores
    
    # Agregar probabilidades por clase
    for i, clase in enumerate(label_encoder.classes_):
        df_clasificado[f'prob_{clase}'] = probabilities[:, i]
    
    # Mostrar resumen de clasificación
    print(f"\n*** RESULTADOS DE CLASIFICACION ***")
    distribucion = pd.Series(predicted_labels).value_counts()
    print(f"Distribución de predicciones:")
    for habito, count in distribucion.items():
        porcentaje = (count / len(predicted_labels)) * 100
        print(f"  {habito}: {count} personas ({porcentaje:.1f}%)")
    
    print(f"\nConfianza promedio: {confidence_scores.mean():.3f}")
    print(f"Confianza mínima: {confidence_scores.min():.3f}")
    print(f"Confianza máxima: {confidence_scores.max():.3f}")
    
    # Crear directorio de salida
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"data/processed/clasificacion_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Guardar dataset completo clasificado
    output_path = f"{output_dir}/datos_clasificados_completo.csv"
    df_clasificado.to_csv(output_path, index=False)
    print(f"\n✅ Dataset completo guardado: {output_path}")
    
    # Generar datasets separados por categoría
    if generar_datasets_separados:
        print(f"\n*** GENERANDO DATASETS SEPARADOS ***")
        
        for habito in label_encoder.classes_:
            df_categoria = df_clasificado[df_clasificado['prediccion_habito'] == habito].copy()
            categoria_path = f"{output_dir}/datos_{habito}.csv"
            df_categoria.to_csv(categoria_path, index=False)
            
            print(f"  {habito}: {len(df_categoria)} personas → {categoria_path}")
        
        # Generar resumen estadístico
        resumen_path = f"{output_dir}/resumen_clasificacion.txt"
        with open(resumen_path, 'w', encoding='utf-8') as f:
            f.write("RESUMEN DE CLASIFICACION DE HABITOS DE LECTURA\n")
            f.write("=" * 50 + "\n")
            f.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Modelo usado: {model_path}\n")
            f.write(f"Datos clasificados: {len(df_clasificado)} personas\n\n")
            
            f.write("DISTRIBUCION POR CATEGORIA:\n")
            for habito, count in distribucion.items():
                porcentaje = (count / len(predicted_labels)) * 100
                f.write(f"  {habito}: {count} personas ({porcentaje:.1f}%)\n")
            
            f.write(f"\nCONFIANZA DEL MODELO:\n")
            f.write(f"  Promedio: {confidence_scores.mean():.3f}\n")
            f.write(f"  Mínima: {confidence_scores.min():.3f}\n")
            f.write(f"  Máxima: {confidence_scores.max():.3f}\n")
            
            f.write(f"\nARCHIVOS GENERADOS:\n")
            f.write(f"  - datos_clasificados_completo.csv: Dataset completo con predicciones\n")
            for habito in label_encoder.classes_:
                f.write(f"  - datos_{habito}.csv: Solo personas clasificadas como {habito}\n")
        
        print(f"  Resumen guardado: {resumen_path}")
    
    print(f"\n🎯 Clasificación completada exitosamente!")
    print(f"📁 Archivos guardados en: {output_dir}")
    
    return df_clasificado, output_dir

if __name__ == "__main__":
    # Ejecutar clasificación
    df_resultado, carpeta_salida = clasificar_y_organizar_datos(
        csv_path="data/raw/jovenes.csv",
        generar_datasets_separados=True
    )
    
    print(f"\n✨ ¡Listo! Revisa los archivos en: {carpeta_salida}")
