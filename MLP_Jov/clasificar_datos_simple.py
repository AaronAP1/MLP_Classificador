#!/usr/bin/env python3

import torch
import pandas as pd
import joblib
import json
import os
from datetime import datetime

def clasificar_y_organizar_datos(csv_path, model_path=None, generar_datasets_separados=True):
    """
    Clasifica habitos de lectura y organiza datos por categorias
    
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
    
    # Cargar preprocesadores (están en data/processed/)
    preprocessor = joblib.load("data/processed/preprocessor.pkl")
    label_encoder = joblib.load("data/processed/label_encoder.pkl")
    
    # Aplicar el ETL mínimo para clasificación
    from utils.etl import crear_target_lectura, clean_data
    
    # Crear target de lectura si no existe
    df_with_target = crear_target_lectura(df_original)
    print(f"Target creado. Muestras: {len(df_with_target)}")
    
    # Usar las mismas features que en entrenamiento
    feature_cols = config["features"]["selected_features"]
    target_col = config["data"]["target_column"]
    
    # Seleccionar features y limpiar datos
    X = df_with_target[feature_cols].copy()
    y = df_with_target[target_col].copy()
    
    # Limpiar datos
    X_clean, y_clean = clean_data(X, y, feature_cols)
    print(f"Datos limpios: {len(X_clean)} muestras")
    
    # Cargar modelo
    from models.model_jovenes import JovenesMLP
    model = JovenesMLP(
        input_dim=input_size,
        num_classes=n_classes,
        hidden_dims=config["model"]["hidden_dims"],
        attn_dim=config["model"]["attn_dim"],
        dropout_p=config["model"]["dropout_p"]
    )
    
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    # Clasificar todos los datos
    all_predictions = []
    all_probabilities = []
    
    with torch.no_grad():
        for X_batch, _ in dl_tr:
            logits = model(X_batch)
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            
            all_predictions.extend(preds.numpy())
            all_probabilities.extend(probs.numpy())
    
    # Convertir predicciones a etiquetas
    pred_labels = label_encoder.inverse_transform(all_predictions)
    
    # Crear DataFrame completo con datos originales + predicciones
    df_clasificado = df_original.copy()
    df_clasificado['prediccion_habito'] = pred_labels
    df_clasificado['prob_lector_frecuente'] = [p[0] for p in all_probabilities]
    df_clasificado['prob_lector_medio'] = [p[1] for p in all_probabilities]
    df_clasificado['prob_no_lector'] = [p[2] for p in all_probabilities]
    df_clasificado['confianza'] = [max(p) for p in all_probabilities]
    
    # Mostrar resumen
    print(f"\n*** RESULTADOS DE CLASIFICACION ***")
    print(f"Distribucion de predicciones:")
    pred_counts = pd.Series(pred_labels).value_counts()
    for categoria, count in pred_counts.items():
        print(f"  - {categoria}: {count:,} personas ({count/len(pred_labels)*100:.1f}%)")
    
    print(f"\nConfianza promedio: {pd.Series([max(p) for p in all_probabilities]).mean():.1%}")
    
    # Crear directorio de salida con timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"data/processed/clasificados_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Guardar dataset completo clasificado
    output_file_completo = f"{output_dir}/dataset_completo_clasificado.csv"
    df_clasificado.to_csv(output_file_completo, index=False)
    print(f"\nDataset completo clasificado guardado en: {output_file_completo}")
    
    if generar_datasets_separados:
        # Generar datasets separados por categoria
        print(f"\nGenerando datasets separados por categoria...")
        
        categorias = df_clasificado['prediccion_habito'].unique()
        for categoria in categorias:
            df_categoria = df_clasificado[df_clasificado['prediccion_habito'] == categoria]
            archivo_categoria = f"{output_dir}/dataset_{categoria.lower().replace(' ', '_')}.csv"
            df_categoria.to_csv(archivo_categoria, index=False)
            print(f"  - {categoria}: {len(df_categoria):,} personas -> {archivo_categoria}")
        
        # Crear resumen estadistico
        resumen = {
            'fecha_clasificacion': timestamp,
            'total_personas': len(df_clasificado),
            'modelo_usado': model_path,
            'confianza_promedio': float(pd.Series([max(p) for p in all_probabilities]).mean()),
            'distribucion_categorias': pred_counts.to_dict()
        }
        
        with open(f"{output_dir}/resumen_clasificacion.json", 'w') as f:
            json.dump(resumen, f, indent=2, ensure_ascii=False)
        
        print(f"  - Resumen estadistico -> {output_dir}/resumen_clasificacion.json")
    
    return df_clasificado, output_dir

if __name__ == "__main__":
    # Clasificar y organizar el dataset original completo
    df_clasificado, output_dir = clasificar_y_organizar_datos('data/raw/jovenes.csv')
    
    print(f"\n*** CLASIFICACION Y ORGANIZACION COMPLETADA ***")
    print(f"   Archivos generados en: {output_dir}")
    print(f"   Dataset completo con {len(df_clasificado):,} personas clasificadas")
    print(f"   Datasets separados por categoria de habito de lectura")
    print(f"   Resumen estadistico de la clasificacion")
    
    # Mostrar estructura de archivos generados
    print(f"\n*** ESTRUCTURA DE ARCHIVOS GENERADOS ***")
    import glob
    archivos = glob.glob(f"{output_dir}/*")
    for archivo in sorted(archivos):
        nombre = archivo.split("\\")[-1]
        if nombre.endswith('.csv'):
            df_temp = pd.read_csv(archivo)
            print(f"  {nombre}: {len(df_temp):,} registros")
        else:
            print(f"  {nombre}")
