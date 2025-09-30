#!/usr/bin/env python3
"""
Script de prueba para clasificar hábitos de lectura de personas individuales.
"""

import pandas as pd
import numpy as np
import torch
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

def clasificar_persona(datos_persona):
    """
    Clasifica los hábitos de lectura de una persona.
    
    Args:
        datos_persona: dict con variables de la persona
        
    Returns:
        dict con la predicción y probabilidades
    """
    
    # Cargar configuración
    with open('configs/config_jovenes.json', 'r') as f:
        config = json.load(f)
    
    # Cargar preprocesadores y modelo
    preprocessor = joblib.load("data/processed/preprocessor.pkl")
    label_encoder = joblib.load("data/processed/label_encoder.pkl")
    
    model_path = "outputs/models/20250703_122801/best_model.pt"
    
    # Crear DataFrame con los datos de la persona
    df_persona = pd.DataFrame([datos_persona])
    
    # Agregar target de lectura
    df_persona = crear_target_lectura_simple(df_persona)
    
    # Mapear las columnas categóricas a texto (simulado)
    cat_cols = ["P505","P506","P507","P508","P509","P201","P203","P206","P207","P208","P311","P315_1","P315_2","P315_3","P315_4","P315_5","P315_6"]
    for col in cat_cols:
        if col in df_persona.columns:
            df_persona[f"{col}_txt"] = df_persona[col].astype(str)
    
    # Seleccionar features según la configuración
    try:
        feature_cols = config["features"]["selected_features"]
    except KeyError:
        # Si no existe, usar las columnas de texto
        feature_cols = [f"{col}_txt" for col in cat_cols if f"{col}_txt" in df_persona.columns]
        feature_cols += ["FACTORFINAL_HOGAR","P210_A","P210_M","P310_N","P310_A","P409","P517","lectura_score"]
        feature_cols = [col for col in feature_cols if col in df_persona.columns]
    
    X = df_persona[feature_cols].copy()
    
    # Limpiar datos
    for col in X.columns:
        if X[col].dtype in ['float64', 'int64']:
            X[col] = X[col].fillna(0).replace([np.inf, -np.inf], 0)
        else:
            X[col] = X[col].fillna('Desconocido')
    
    # Aplicar preprocesamiento
    X_processed = preprocessor.transform(X)
    
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
    
    # Cargar pesos
    device = torch.device('cpu')  # Usar CPU para predicciones individuales
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Hacer predicción
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_processed)
        outputs = model(X_tensor)
        probabilities = torch.softmax(outputs, dim=1).numpy()[0]
        predicted_index = torch.argmax(outputs, dim=1).numpy()[0]
        
        predicted_label = label_encoder.inverse_transform([predicted_index])[0]
        confidence = np.max(probabilities)
    
    # Preparar resultado
    resultado = {
        'prediccion': predicted_label,
        'confianza': float(confidence),
        'probabilidades': {
            clase: float(prob) 
            for clase, prob in zip(label_encoder.classes_, probabilities)
        }
    }
    
    return resultado

def crear_target_lectura_simple(df):
    """Crear target de lectura simplificado."""
    df = df.copy()
    
    # Crear score compuesto de lectura
    df['lectura_score'] = 0.0
    
    # 1. Base: P517 (frecuencia directa) - normalizado 0-1
    if 'P517' in df.columns:
        mask_p517 = df['P517'].notna()
        if mask_p517.any():
            df.loc[mask_p517, 'lectura_score'] = (df.loc[mask_p517, 'P517'] - 1) / 4
    
    # 2. Para el resto: variables P315 (tipos de material)
    lectura_vars = ['P315_1', 'P315_2', 'P315_3', 'P315_4', 'P315_5', 'P315_6']
    for var in lectura_vars:
        if var in df.columns:
            mask_valid = df[var].notna()
            if mask_valid.any():
                df.loc[mask_valid, 'lectura_score'] += df.loc[mask_valid, var] * 0.1
    
    # 3. P311: saber leer (base fundamental)
    if 'P311' in df.columns:
        mask_valid = df['P311'].notna()
        if mask_valid.any():
            df.loc[mask_valid & (df['P311'] == 0), 'lectura_score'] = 0.0
    
    # 4. P409: frecuencia alternativa
    if 'P409' in df.columns:
        mask_valid = df['P409'].notna()
        if mask_valid.any():
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
    """Función principal para probar clasificaciones."""
    
    print("🔍 CLASIFICADOR DE HABITOS DE LECTURA - MODO PRUEBA")
    print("=" * 60)
    
    # Ejemplos de personas para probar
    ejemplos = [
        {
            "nombre": "Persona Lectora Frecuente",
            "datos": {
                "P517": 5,      # Lee muy frecuentemente
                "P311": 1,      # Sabe leer
                "P315_1": 3,    # Lee libros
                "P315_2": 2,    # Lee revistas
                "P409": 4,      # Lee frecuentemente (alternativo)
                "P210_A": 25,   # 25 años
                "FACTORFINAL_HOGAR": 300
            }
        },
        {
            "nombre": "Persona No Lectora",
            "datos": {
                "P517": 1,      # Nunca lee
                "P311": 1,      # Sabe leer pero no lo hace
                "P315_1": 0,    # No lee libros
                "P409": 1,      # No lee nada
                "P210_A": 30,   # 30 años
                "FACTORFINAL_HOGAR": 250
            }
        },
        {
            "nombre": "Persona Lectora Media",
            "datos": {
                "P517": 3,      # Lee ocasionalmente
                "P311": 1,      # Sabe leer
                "P315_1": 1,    # Lee pocos libros
                "P315_2": 2,    # Lee algunas revistas
                "P409": 3,      # Lee de vez en cuando
                "P210_A": 22,   # 22 años
                "FACTORFINAL_HOGAR": 400
            }
        },
        {
            "nombre": "Persona con Datos Mínimos",
            "datos": {
                "P517": 4,      # Solo frecuencia de lectura
                "P210_A": 28    # Solo edad
            }
        }
    ]
    
    # Probar cada ejemplo
    for i, ejemplo in enumerate(ejemplos, 1):
        print(f"\n{i}. {ejemplo['nombre']}")
        print("-" * 40)
        
        # Mostrar datos de entrada
        print("Datos de entrada:")
        for variable, valor in ejemplo['datos'].items():
            print(f"  {variable}: {valor}")
        
        try:
            # Clasificar
            resultado = clasificar_persona(ejemplo['datos'])
            
            # Mostrar resultado
            print(f"\n✅ RESULTADO:")
            print(f"  Predicción: {resultado['prediccion']}")
            print(f"  Confianza: {resultado['confianza']:.3f}")
            print(f"  Probabilidades:")
            for clase, prob in resultado['probabilidades'].items():
                print(f"    {clase}: {prob:.3f}")
                
        except Exception as e:
            print(f"❌ Error al clasificar: {e}")
    
    print(f"\n" + "=" * 60)
    print("🎯 Pruebas completadas!")
    print("\nPara clasificar una persona nueva, usa:")
    print("resultado = clasificar_persona({'P517': 4, 'P311': 1, 'P210_A': 25})")

if __name__ == "__main__":
    main()
