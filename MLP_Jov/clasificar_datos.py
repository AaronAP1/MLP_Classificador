#!/usr/bin/env python3

import torch
import pandas as pd
import joblib
import json

def clasificar_nuevos_datos(csv_path, model_path=None):
    """
    Clasifica hábitos de lectura de nuevos datos usando el modelo entrenado
    """
    
    # Cargar config
    with open('configs/config_jovenes.json', 'r') as f:
        config = json.load(f)
    
    # Si no se especifica modelo, usar el más reciente
    if model_path is None:
        import glob
        model_dirs = glob.glob("outputs/models/202*")
        latest_dir = max(model_dirs)
        model_path = f"{latest_dir}/best_model.pt"
    
    print(f"🔮 CLASIFICADOR DE HÁBITOS DE LECTURA")
    print(f"📁 Modelo: {model_path}")
    print("=" * 60)
    
    # Cargar datos
    df = pd.read_csv(csv_path)
    print(f"📊 Datos cargados: {df.shape[0]} personas")
    
    # Cargar preprocesadores
    latest_dir = "/".join(model_path.split("/")[:-1])
    preprocessor = joblib.load(f"{latest_dir}/../processed/preprocessor.pkl")
    label_encoder = joblib.load(f"{latest_dir}/../processed/label_encoder.pkl")
    
    # Aplicar el mismo ETL que en entrenamiento
    from utils.etl import preprocess_and_load
    
    # Simular config para solo procesar sin dividir
    temp_config = config.copy()
    temp_config["data"]["raw_csv"] = csv_path
    
    # Procesar datos (esto creará automáticamente el target si no existe)
    dl_tr, dl_va, dl_te, input_size, n_classes = preprocess_and_load(temp_config, {})
    
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
    
    # Crear DataFrame de resultados
    results = pd.DataFrame({
        'prediccion_habito': pred_labels,
        'prob_lector_frecuente': [p[0] for p in all_probabilities],
        'prob_lector_medio': [p[1] for p in all_probabilities], 
        'prob_no_lector': [p[2] for p in all_probabilities],
        'confianza': [max(p) for p in all_probabilities]
    })
    
    # Mostrar resumen
    print(f"\n📈 RESULTADOS DE CLASIFICACIÓN:")
    print(f"Distribución de predicciones:")
    print(results['prediccion_habito'].value_counts())
    
    print(f"\n📊 Confianza promedio: {results['confianza'].mean():.1%}")
    
    # Guardar resultados
    output_file = csv_path.replace('.csv', '_clasificado.csv')
    results.to_csv(output_file, index=False)
    print(f"\n✅ Resultados guardados en: {output_file}")
    
    return results

if __name__ == "__main__":
    # Clasificar el dataset original completo
    results = clasificar_nuevos_datos('data/raw/jovenes.csv')
    
    print(f"\n🎯 ¡Clasificación completada!")
    print(f"   Cada persona ha sido clasificada según sus hábitos de lectura")
