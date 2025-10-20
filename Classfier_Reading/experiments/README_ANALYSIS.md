# 🔬 Análisis Experimental - Guía de Uso

## ¿Qué hace este script?

Este script responde a las preguntas más importantes **ANTES** de entrenar todos tus modelos:

### 1️⃣ **¿Es suficiente seed=42?**
- Entrena el mismo modelo con 5 seeds diferentes (42, 123, 456, 789, 2024)
- Mide la **variabilidad** entre resultados
- Te dice si necesitas promediar múltiples seeds o si uno solo es confiable

**Resultado esperado:**
- Si CV < 3%: ✅ Seed 42 es suficiente
- Si CV > 5%: ⚠️ Necesitas promediar 5 seeds

---

### 2️⃣ **¿Cuántas épocas necesito realmente?**
- Prueba diferentes configuraciones: 50, 100, 150, 200, 250 épocas
- Mide el **costo-beneficio** (tiempo vs mejora)
- Identifica el punto óptimo

**Resultado esperado:**
- Encontrar que 150 épocas dan F1=0.85 en 120s
- Pero 200 épocas solo mejoran a F1=0.851 en 180s
- → **Recomendación:** Usar 150 épocas (ahorra 60s sin perder calidad)

---

### 3️⃣ **¿Hay overfitting?**
- Grafica curvas de **training loss** vs **validation loss**
- Detecta si el modelo está "memorizando" (overfitting)
- Valida que early stopping funciona correctamente

**Resultado esperado:**
- Ver que val_loss deja de mejorar después de ~100 épocas
- Confirmar que patience=30 es adecuado

---

## 🚀 Cómo usar

### Paso 1: Ejecutar análisis
```bash
cd "c:\Users\xdxdxdxd\OneDrive - Universidad Peruana Unión\Escritorio\MLP_Py_In01"
python Classfier_Reading/experiments/analyze_configurations.py
```

**Tiempo estimado:** 15-30 minutos (entrena ~15 modelos pequeños)

---

### Paso 2: Revisar resultados

El script genera:

```
experiments/analysis/
├── exp1_multiple_seeds.csv          # Resultados de 5 seeds
├── exp1_seed_variability.png        # Gráfico de variabilidad
├── exp2_learning_curves.png         # Curvas de entrenamiento
├── exp3_epoch_comparison.csv        # Comparación de épocas
├── exp3_epoch_comparison.png        # Gráfico épocas vs F1
└── recommendations.json             # ⭐ RECOMENDACIONES FINALES
```

---

### Paso 3: Interpretar recommendations.json

Ejemplo de salida:

```json
{
    "seed_strategy": "single",
    "seed_variability_cv": 1.85,
    "recommended_epochs": 150,
    "recommended_patience": 30,
    "expected_f1": 0.8523,
    "time_per_model_seconds": 120.5,
    "total_models": 9,
    "total_time_hours": 0.30,
    "recommendations": {
        "seed": "Usar seed=42",
        "epochs": "150 épocas con patience=30",
        "class_weights": "Aplicar en jóvenes y adultos",
        "metrics": "Priorizar F1-Macro"
    }
}
```

**Interpretación:**

✅ **Seed único es suficiente** (CV=1.85% < 3%)
- No necesitas promediar 5 seeds
- Puedes usar seed=42 con confianza

✅ **150 épocas son óptimas**
- F1 esperado: 0.8523
- Tiempo por modelo: 120s (~2 minutos)

✅ **Tiempo total: 18 minutos**
- 9 modelos × 120s = 1,080s = 18 minutos
- ¡Puedes entrenar todo en media hora!

---

## 📊 Ejemplos de Gráficos Generados

### Gráfico 1: Variabilidad entre Seeds
![Seed Variability](docs/example_seed_plot.png)

**Qué buscar:**
- Puntos muy juntos = Modelo estable ✅
- Puntos dispersos = Modelo inestable ⚠️

---

### Gráfico 2: Curvas de Aprendizaje
![Learning Curves](docs/example_learning_curves.png)

**Qué buscar:**
- Val Loss deja de bajar = Convergencia alcanzada ✅
- Val Loss sube mientras Train Loss baja = Overfitting 🔴
- Ambas curvas bajan juntas = Entrenamiento saludable ✅

---

### Gráfico 3: Épocas vs F1-Score
![Epoch Comparison](docs/example_epoch_plot.png)

**Qué buscar:**
- Curva se aplana = No vale la pena más épocas
- Curva sigue subiendo = Puede beneficiarse de más épocas

---

## 🎓 Para tu Tesis

### Si seed=42 es suficiente (CV < 3%):

**Escribir en Metodología:**
```
Para garantizar reproducibilidad, se fijó la semilla aleatoria (seed) en 42
para todos los experimentos. Análisis preliminares con 5 seeds diferentes
mostraron una variabilidad menor al 3% (CV=1.85%), confirmando la estabilidad
del modelo y validando el uso de un único seed.
```

---

### Si necesitas múltiples seeds (CV > 5%):

**Escribir en Metodología:**
```
Debido a la variabilidad inherente del proceso de entrenamiento (CV=6.2%),
todos los resultados se reportan como promedio ± desviación estándar de
5 ejecuciones con diferentes semillas aleatorias [42, 123, 456, 789, 2024].
```

**En Resultados:**
```
TabNet (Niños): F1-Macro = 0.852 ± 0.012
```

---

### Sobre las Épocas:

**Escribir en Metodología:**
```
Los modelos se entrenaron con un máximo de 150 épocas, utilizando early
stopping con paciencia de 30 épocas para prevenir overfitting. Esta
configuración fue determinada mediante análisis empírico que mostró que
150 épocas son suficientes para convergencia, con mejoras marginales
(<0.5%) al extender el entrenamiento.
```

---

## 💡 Tips Adicionales

### Si el script tarda mucho:
```python
# Reducir epochs en EXPERIMENT_CONFIG:
'epoch_tests': [50, 100, 150],  # En lugar de [50, 100, 150, 200, 250]
'seeds': [42, 123, 456],         # En lugar de 5 seeds
```

### Si quieres analizar otro grupo:
```python
# Cambiar en EXPERIMENT_CONFIG:
'grupo': 'jovenes',  # o 'adultos'
'num_features': 139,  # o 71 para adultos
'data_dir': project_root / 'data' / 'processed' / 'jovenes',
```

---

## ❓ FAQ

**P: ¿Por qué empezar con niños?**
R: Porque es el dataset balanceado, sin complicaciones de class weights. Si funciona bien aquí, funcionará en los otros grupos.

**P: ¿Debo ejecutar esto 3 veces (una por grupo)?**
R: NO es estrictamente necesario. Los patrones suelen ser similares. Pero si tienes tiempo, sí vale la pena.

**P: ¿Qué hago si CV > 10% (muy inestable)?**
R: Posibles causas:
1. Learning rate muy alto
2. Dataset muy pequeño
3. Arquitectura inadecuada
→ Considera ajustar hiperparámetros o usar ensemble de modelos

**P: ¿Cuánto espacio en disco necesito?**
R: ~500MB (los modelos intermedios se descartan)

---

## 🎯 Siguiente Paso

Después de ejecutar esto:

1. ✅ Leer `recommendations.json`
2. ✅ Ajustar `train_ninos.py`, `train_jovenes.py`, `train_adultos.py` si es necesario
3. ✅ Ejecutar entrenamiento completo con confianza

**¡Ya sabes que tu configuración es sólida!** 🚀
