#!/usr/bin/env python3
"""
Script ultra-simple para probar clasificación individual.
"""

import pandas as pd
import numpy as np
import torch
import joblib
import warnings
warnings.filterwarnings('ignore')

def predecir_habito_lectura(frecuencia_lectura=None, sabe_leer=None, edad=None):
    """
    Predice hábito de lectura con variables mínimas.
    
    Args:
        frecuencia_lectura: 1-5 (1=nunca, 5=muy frecuente)
        sabe_leer: 0 o 1 (0=no sabe, 1=sabe)
        edad: años
    
    Returns:
        tupla (prediccion, confianza)
    """
    
    try:
        # Preparar datos básicos
        datos = {}
        
        if frecuencia_lectura is not None:
            datos['P517'] = frecuencia_lectura
        if sabe_leer is not None:
            datos['P311'] = sabe_leer
        if edad is not None:
            datos['P210_A'] = edad
            
        # Valores por defecto para completar
        defaults = {
            'P517': 2,  # Lee poco por defecto
            'P311': 1,  # Sabe leer por defecto
            'P210_A': 25,  # 25 años por defecto
            'FACTORFINAL_HOGAR': 300,
            'P210_M': 0,
            'P310_N': 5,
            'P310_A': 2,
            'P409': 2
        }
        
        # Completar datos faltantes
        for key, default_val in defaults.items():
            if key not in datos:
                datos[key] = default_val
        
        # Crear DataFrame
        df = pd.DataFrame([datos])
        
        # Crear score de lectura simple
        score = 0.0
        if 'P517' in datos:
            score = (datos['P517'] - 1) / 4  # Normalizar 1-5 a 0-1
        
        if 'P311' in datos and datos['P311'] == 0:
            score = 0.0  # Si no sabe leer, score = 0
            
        df['lectura_score'] = score
        
        # Crear target
        if score < 0.2:
            target = "no_lector"
        elif score < 0.6:
            target = "lector_medio"
        else:
            target = "lector_frecuente"
            
        df['habito_lectura_completo'] = target
        
        # Agregar columnas de texto dummy
        for col in ['P505','P506','P507','P508','P509','P201','P203','P206','P207','P208','P311','P315_1','P315_2','P315_3','P315_4','P315_5','P315_6']:
            df[f'{col}_txt'] = str(datos.get(col, 'Desconocido'))
        
        # Features mínimas
        feature_cols = [
            'P505_txt', 'P506_txt', 'P507_txt', 'P508_txt', 'P509_txt',
            'P201_txt', 'P203_txt', 'P206_txt', 'P207_txt', 'P208_txt',
            'P311_txt', 'P315_1_txt', 'P315_2_txt', 'P315_3_txt', 'P315_4_txt', 'P315_5_txt', 'P315_6_txt',
            'FACTORFINAL_HOGAR', 'P210_A', 'P210_M', 'P310_N', 'P310_A', 'P409', 'P517', 'lectura_score'
        ]
        
        X = df[feature_cols].copy()
        
        # Cargar preprocesadores
        preprocessor = joblib.load("data/processed/preprocessor.pkl")
        label_encoder = joblib.load("data/processed/label_encoder.pkl")
        
        # Procesar
        X_processed = preprocessor.transform(X)
        
        # Cargar modelo
        from models.model_jovenes import MLPModel
        model = MLPModel(
            input_dim=X_processed.shape[1],
            num_classes=3,
            hidden_dims=[256, 128, 64, 32],
            dropout_rate=0.3,
            attention_dim=64
        )
        
        model.load_state_dict(torch.load("outputs/models/20250703_122801/best_model.pt", map_location='cpu'))
        model.eval()
        
        # Predecir
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_processed)
            outputs = model(X_tensor)
            probs = torch.softmax(outputs, dim=1).numpy()[0]
            pred_idx = torch.argmax(outputs, dim=1).numpy()[0]
            
            pred_label = label_encoder.inverse_transform([pred_idx])[0]
            confidence = np.max(probs)
        
        return pred_label, confidence
        
    except Exception as e:
        return f"Error: {e}", 0.0

def main():
    """Pruebas rápidas."""
    
    print("🚀 CLASIFICADOR RAPIDO DE HABITOS DE LECTURA")
    print("=" * 50)
    
    # Pruebas simples
    tests = [
        ("Lector frecuente", 5, 1, 25),
        ("No lector", 1, 1, 30),
        ("Lector medio", 3, 1, 22),
        ("Solo frecuencia", 4, None, None),
        ("Mínimo", None, None, None)
    ]
    
    for desc, freq, sabe, edad in tests:
        pred, conf = predecir_habito_lectura(freq, sabe, edad)
        print(f"\n{desc}:")
        print(f"  Entrada: frecuencia={freq}, sabe_leer={sabe}, edad={edad}")
        print(f"  Resultado: {pred} (confianza: {conf:.3f})")
    
    print(f"\n" + "=" * 50)
    print("✅ Pruebas completadas!")
    print("\nUso: predecir_habito_lectura(frecuencia_lectura=4, sabe_leer=1, edad=25)")

if __name__ == "__main__":
    main()
