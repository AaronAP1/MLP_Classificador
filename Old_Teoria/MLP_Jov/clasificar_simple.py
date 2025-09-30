#!/usr/bin/env python3
"""
Script simple para clasificar datos usando el modelo entrenado.
"""

import pandas as pd
import numpy as np
import torch
import joblib
import json
import os
from datetime import datetime

def crear_target_lectura_simple(df):
    """Crear target de lectura simplificado."""
    df = df.copy()
    
    # Crear score compuesto de lectura
    df['lectura_score'] = 0.0
    
    # 1. Base: P517 (frecuencia directa) - normalizado 0-1
    if 'P517' in df.columns:
        mask_p517 = df['P517'].notna()
        df.loc[mask_p517, 'lectura_score'] = (df.loc[mask_p517, 'P517'] - 1) / 4
    
    # 2. Para el resto: variables P315 (tipos de material)
    mask_no_p517 = ~mask_p517 if 'P517' in df.columns else df.index
    lectura_vars = ['P315_1', 'P315_2', 'P315_3', 'P315_4', 'P315_5', 'P315_6']
    for var in lectura_vars:
        if var in df.columns:
            mask_valid = mask_no_p517 & df[var].notna()
            df.loc[mask_valid, 'lectura_score'] += df.loc[mask_valid, var] * 0.1
    
    # 3. P311: saber leer (base fundamental)
    if 'P311' in df.columns:
        mask_valid = mask_no_p517 & df['P311'].notna()
        df.loc[mask_valid & (df['P311'] == 0), 'lectura_score'] = 0.0
    
    # 4. P409: frecuencia alternativa
    if 'P409' in df.columns:
        mask_valid = mask_no_p517 & df['P409'].notna()
        df.loc[mask_valid, 'lectura_score'] += (df.loc[mask_valid, 'P409'] - 1) / 5 * 0.3
    
    # 5. Normalizar y categorizar
    df['lectura_score'] = np.clip(df['lectura_score'], 0, 1)
    
    def score_to_category(score):
        if pd.isna(score) or score < 0.2:
            return "no_lector"
        elif score < 0.6:
            return "lector_medio"
        else:
            return "lector_frecuente"
    
    df["habito_lectura_completo"] = df['lectura_score'].apply(score_to_category)
    return df

def main():
    """Función principal para clasificar datos."""
    
    print("*** CLASIFICADOR DE HABITOS DE LECTURA ***")
    print("=" * 50)
    
    # Cargar configuración
    with open('configs/config_jovenes.json', 'r') as f:
        config = json.load(f)
    
    # Usar modelo más reciente
    model_path = "outputs/models/20250703_122801/best_model.pt"
    print(f"Usando modelo: {model_path}")
    
    # Cargar datos
    df = pd.read_csv("data/raw/jovenes.csv")
    print(f"Datos cargados: {df.shape[0]} filas, {df.shape[1]} columnas")
    
    # Cargar preprocesadores
    preprocessor = joblib.load("data/processed/preprocessor.pkl")
    label_encoder = joblib.load("data/processed/label_encoder.pkl")
    print("Preprocesadores cargados")
    
    # Crear target de lectura (usando la lógica del ETL)
    df_with_target = crear_target_lectura_simple(df)
    print(f"Target creado: {len(df_with_target)} muestras")
    
    # Preparar features
    feature_cols = config["features"]["selected_features"]
    X = df_with_target[feature_cols].copy()
    
    # Limpiar datos simples
    for col in X.columns:
        if X[col].dtype in ['float64', 'int64']:
            X[col] = X[col].fillna(X[col].median()).replace([np.inf, -np.inf], X[col].median())
        else:
            X[col] = X[col].fillna('Desconocido')
    
    print(f"Datos preparados: {X.shape}")
    
    # Aplicar preprocesamiento
    X_processed = preprocessor.transform(X)
    print(f"Datos procesados: {X_processed.shape}")
    
    # Cargar modelo
    from models.model_jovenes import MLPModel
    
    input_size = X_processed.shape[1]
    n_classes = len(label_encoder.classes_)
    
    model = MLPModel(
        input_dim=input_size,
        num_classes=n_classes,
        hidden_dims=config["model"]["hidden_dims"],
        dropout_rate=config["model"]["dropout_rate"],
        attention_dim=config["model"].get("attention_dim", 64)
    )
    
    # Cargar pesos entrenados
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    print(f"Modelo cargado en: {device}")
    print(f"Clases: {label_encoder.classes_}")
    
    # Hacer predicciones
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_processed).to(device)
        outputs = model(X_tensor)
        probabilities = torch.softmax(outputs, dim=1).cpu().numpy()
        predicted_indices = torch.argmax(outputs, dim=1).cpu().numpy()
        
        predicted_labels = label_encoder.inverse_transform(predicted_indices)
        confidence_scores = np.max(probabilities, axis=1)
    
    print(f"Predicciones completadas: {len(predicted_labels)} muestras")
    
    # Mostrar distribución
    print("\\nDistribución de predicciones:")
    distribucion = pd.Series(predicted_labels).value_counts()
    for habito, count in distribucion.items():
        porcentaje = (count / len(predicted_labels)) * 100
        print(f"  {habito}: {count} ({porcentaje:.1f}%)")
    
    print(f"\\nConfianza promedio: {confidence_scores.mean():.3f}")
    
    # Crear resultado
    df_resultado = df.copy()
    df_resultado['prediccion_habito'] = predicted_labels
    df_resultado['confianza'] = confidence_scores
    
    # Agregar probabilidades
    for i, clase in enumerate(label_encoder.classes_):
        df_resultado[f'prob_{clase}'] = probabilities[:, i]
    
    # Crear directorio de salida
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"data/processed/clasificacion_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Guardar dataset completo
    output_path = f"{output_dir}/datos_clasificados_completo.csv"
    df_resultado.to_csv(output_path, index=False)
    print(f"\\n✅ Dataset completo guardado: {output_path}")
    
    # Guardar datasets separados
    for habito in label_encoder.classes_:
        df_categoria = df_resultado[df_resultado['prediccion_habito'] == habito]
        categoria_path = f"{output_dir}/datos_{habito}.csv"
        df_categoria.to_csv(categoria_path, index=False)
        print(f"  {habito}: {len(df_categoria)} personas → {categoria_path}")
    
    # Guardar resumen
    resumen_path = f"{output_dir}/resumen.txt"
    with open(resumen_path, 'w', encoding='utf-8') as f:
        f.write("RESUMEN DE CLASIFICACION\\n")
        f.write("=" * 30 + "\\n")
        f.write(f"Fecha: {datetime.now()}\\n")
        f.write(f"Total clasificado: {len(df_resultado)}\\n\\n")
        f.write("Distribución:\\n")
        for habito, count in distribucion.items():
            porcentaje = (count / len(predicted_labels)) * 100
            f.write(f"  {habito}: {count} ({porcentaje:.1f}%)\\n")
    
    print(f"  Resumen: {resumen_path}")
    print(f"\\n🎯 ¡Clasificación completada exitosamente!")
    print(f"📁 Archivos en: {output_dir}")

if __name__ == "__main__":
    main()
