# 🔍 ANÁLISIS CRÍTICO: Target Variable y Data Leakage

**Fecha**: 20 de Octubre, 2025  
**Universidad Peruana Unión**  
**Investigador**: Aaron

---

## 📊 HALLAZGO PRINCIPAL

**Los 3 grupos (Niños, Jóvenes, Adultos) presentan 99.9-100% de accuracy con modelos simples**

| Grupo | LogReg Accuracy | DT(depth=3) Accuracy | Clasificación |
|-------|----------------|---------------------|---------------|
| **NIÑOS** | 100.0% | 100.0% | 🔴 TRIVIAL |
| **JÓVENES** | 99.9% | 99.9% | 🔴 TRIVIAL |
| **ADULTOS** | 100.0% | 100.0% | 🔴 TRIVIAL |

---

## 🎯 ORIGEN DEL TARGET: `comportamiento_lector`

### **Cómo se construye el target** (desde `src/data/etl.py`)

```python
def create_target_variable(df: pd.DataFrame, grupo: str) -> pd.DataFrame:
    """
    Crea la variable objetivo 'comportamiento_lector' con 3 clases.
    
    Clases:
    - 0: No lector
    - 1: Lector medio
    - 2: Lector frecuente
    """
```

### **Estrategia 1: Usar P517 directamente**

```python
# Mapeo directo de P517 (frecuencia de lectura)
mapping = {
    1.0: 0,  # Nunca → No lector
    2.0: 1,  # De vez en cuando → Lector medio
    3.0: 1,  # 1-2 veces por semana → Lector medio
    4.0: 2,  # 3-4 veces por semana → Lector frecuente
    5.0: 2   # Todos los días → Lector frecuente
}
df["comportamiento_lector"] = df["P517"].map(mapping)
```

**⚠️ PROBLEMA**: Si P517 es una pregunta del cuestionario ("¿Con qué frecuencia lee?"), entonces:
- **El target ES literalmente la respuesta a una pregunta**
- **NO es una variable derivada independiente**
- **Es una categorización directa de P517**

---

### **Estrategia 2: Derivar desde múltiples variables**

Si P517 no está disponible, el código deriva un `lectura_score`:

```python
# Sumar puntos por tipos de lectura
for var in ['P514_1', 'P514_2', 'P514_3']:  # Tipos de lectura
    df['lectura_score'] += df[var] * 0.1

# Ajustar por frecuencia
df['lectura_score'] += (df['P515'] - 1) / 5 * 0.3

# Convertir a categorías
if score < 0.2:
    return 0  # No lector
elif score < 0.6:
    return 1  # Lector medio
else:
    return 2  # Lector frecuente
```

**⚠️ PROBLEMA**: 
- P514_1, P514_2, P514_3 son **respuestas del cuestionario sobre tipos de lectura**
- P515 es **frecuencia de lectura** (respuesta directa del encuestado)
- El target es **una combinación lineal de las features de entrada**

---

## 🔴 DIAGNÓSTICO: DATA LEAKAGE POR DISEÑO

### **Tipo de Leakage**: **TARGET LEAKAGE (Circular Reasoning)**

El problema es que:

1. **Las features de entrada incluyen**:
   - P514_1, P514_2, P514_3 (tipos de lectura)
   - P515 (frecuencia de lectura)
   - P311 (¿lee por placer/información?)
   - P409 (frecuencia de lectura adultos)

2. **El target se calcula como**:
   - Transformación directa de P517 (frecuencia)
   - O suma ponderada de P514_*, P515, P311, P409

3. **Resultado**: 
   - El modelo aprende: "Si P515 = 5 → target = 2"
   - **No es aprendizaje, es memorización de la fórmula**

---

## 📋 EVIDENCIA DEL LEAKAGE

### **1. Accuracy Perfecta con Modelos Triviales**

```
NIÑOS:
   DT(depth=1): 88.5% (aprende 1 regla simple)
   DT(depth=3): 100.0% (aprende 3-7 reglas → suficiente)
   LogReg: 100.0% (modelo lineal más simple)

JÓVENES:
   DT(depth=1): 99.8%
   LogReg: 99.9%

ADULTOS:
   DT(depth=1): 100.0%
   LogReg: 100.0%
```

**Conclusión**: Si un Decision Tree de profundidad 1 alcanza 88-100%, significa que **UNA SOLA variable predice perfectamente el target**.

---

### **2. Concentración de Feature Importance**

```
NIÑOS:
   Top 3 features: 69.4%
   Top 5 features: 90.7%
   
JÓVENES:
   Top 3 features: 29.4%
   Top 10 features: 69.0%

ADULTOS:
   Top 3 features: 41.8%
   Top 10 features: 88.7%
```

**Conclusión**: Pocas features dominan → probablemente P514, P515, P517 que están directamente relacionadas con el target.

---

### **3. Distribuciones Extremadamente Desbalanceadas**

```
NIÑOS:    7% / 80% / 12%   (Clase 1 domina)
JÓVENES:  80% / 0.2% / 19%  (Clase 0 domina, Clase 1 casi inexistente)
ADULTOS:  20% / 0.02% / 79% (Clase 2 domina, Clase 1 casi inexistente)
```

**Pregunta crítica**: ¿Por qué las distribuciones son TAN diferentes entre grupos?

**Posible respuesta**: 
- Si P517 (o variables similares) tienen códigos diferentes por grupo
- O si las preguntas del cuestionario son diferentes por edad
- Entonces el target refleja **artefactos del diseño de la encuesta**, no comportamiento real

---

## 🎓 IMPLICACIONES PARA LA TESIS

### **Opción A: RECONOCER EL PROBLEMA** ✅ RECOMENDADO

**Narrativa Honesta**:

> *"Durante el desarrollo del modelo, identificamos que la variable objetivo (`comportamiento_lector`) se derivó directamente de variables del cuestionario que también se utilizaron como features de entrada (P514, P515, P517). Esto genera **target leakage**, donde el modelo aprende la fórmula de construcción del target en lugar de patrones subyacentes del comportamiento lector.*
>
> *Esta situación es común en estudios de encuestas donde no existe una variable objetivo independiente (ej. evaluación externa de comprensión lectora). Los resultados de 99-100% de accuracy con modelos simples confirman este diagnóstico.*
>
> *Por tanto, reformulamos el objetivo del trabajo hacia la **interpretabilidad** de patrones de lectura mediante técnicas de Deep Learning, específicamente:*
> 1. *Análisis de attention mechanisms para identificar variables más relevantes*
> 2. *Comparación de arquitecturas (TabNet, MLP+Attention, FT-Transformer) en su capacidad explicativa*
> 3. *Generación de perfiles de lectores basados en características sociodemográficas*"

**Ventajas**:
- ✅ Demuestra rigor científico
- ✅ Mantiene la validez de los modelos de Deep Learning (para interpretabilidad)
- ✅ Evita críticas de revisores que detectarían el leakage
- ✅ Redefine el aporte: de "predicción" a "análisis interpretativo"

---

### **Opción B: RE-DISEÑAR EL TARGET** 🔧

**Alternativa 1: Target Independiente**
- Usar **SOLO variables demográficas/sociodemográficas** como features:
  - P201 (edad), P203 (sexo), P206 (educación), P207 (ocupación)
  - CCDD (departamento), ESTRATOSOCIO (nivel socioeconómico)
- Target: P517 o derivación de P514/P515
- **NO incluir** P514, P515, P311, P409 como features

**Alternativa 2: Clustering No Supervisado**
- Aplicar K-Means, DBSCAN o Gaussian Mixture sobre **todas las variables de lectura**
- Generar 3 clusters naturales
- Analizar características de cada cluster
- **Ventaja**: No hay target → no hay leakage

---

### **Opción C: MANTENER COMO ESTÁ** ❌ NO RECOMENDADO

**Argumentación forzada**:
> *"Los resultados de 100% accuracy validan que las características sociodemográficas y de acceso a lectura son predictores perfectos del comportamiento lector"*

**Problemas**:
- ❌ Revisores técnicos detectarán el leakage
- ❌ Aporta poco valor científico
- ❌ No permite generalización a nuevos datos
- ❌ Contradice el objetivo de "modelo predictivo" en la tesis

---

## 💡 RECOMENDACIÓN FINAL

### **ACCIÓN INMEDIATA**:

1. **DETENER entrenamientos de modelos complejos** (TabNet, MLP+Attention, FT-Transformer con 50-250 epochs)
   - Ya sabemos que obtendrán 99-100% accuracy
   - No aportarán información nueva

2. **REUNIÓN CON ASESOR** para discutir:
   - ¿El diseño del dataset permite análisis predictivo?
   - ¿Hay evaluaciones independientes de comprensión lectora?
   - ¿Se puede acceder a datos de la ENAHO sin variables de lectura filtradas?

3. **RE-DISEÑAR PIPELINE** según decisión:
   - **Si Opción A**: Enfoque en interpretabilidad (1 semana)
   - **Si Opción B**: Re-hacer ETL + entrenar (2-3 semanas)
   - **Si Opción C**: Continuar pero con argumentación clara de limitaciones

---

## 📚 REFERENCIAS TÉCNICAS

### **Data Leakage en ML**:

1. **Kaggle - Data Leakage**: https://www.kaggle.com/discussions/getting-started/65186
2. **Kaufman et al. (2011)**: "Leakage in Data Mining: Formulation, Detection, and Avoidance"
3. **Kapoor & Narayanan (2023)**: "Leakage and the Reproducibility Crisis in ML-based Science"

### **Target Leakage Específico**:

> *"Target leakage occurs when features are derived from the target variable itself, creating a circular dependency that inflates performance metrics but prevents generalization"*

**Tu caso exacto**:
```python
# Feature P515 (frecuencia de lectura)
X['P515'] = [1, 2, 3, 4, 5]

# Target derivado de P515
y['comportamiento_lector'] = map_frequency_to_category(X['P515'])

# Resultado: modelo aprende la función map(), no patrones reales
```

---

## ✅ CHECKLIST DE DECISIÓN

Marca lo que aplica a tu situación:

- [ ] **He confirmado con mi asesor que el dataset tiene este problema**
- [ ] **Tengo acceso a evaluaciones independientes de comprensión lectora**
- [ ] **Puedo re-hacer el ETL excluyendo variables de lectura de las features**
- [ ] **Mi asesor aprueba el enfoque de interpretabilidad en lugar de predicción**
- [ ] **He actualizado los objetivos de la tesis para reflejar el cambio**
- [ ] **Tengo tiempo (2-3 semanas) para re-diseñar el pipeline**

---

## 🚀 PRÓXIMOS PASOS

### **Si decides Opción A (Interpretabilidad)**:

1. ✅ Ya tienes los análisis de complejidad
2. ✅ Entrenar modelos con énfasis en attention/feature importance
3. ✅ Generar visualizaciones de attention weights
4. ✅ Crear perfiles de lectores por grupo etario
5. ✅ Escribir sección de metodología reconociendo el target leakage

### **Si decides Opción B (Re-diseño)**:

1. 🔧 Modificar `select_features()` en ETL para excluir variables de lectura
2. 🔧 Mantener solo P517/P515 como target puro (no como feature)
3. 🔧 Re-entrenar modelos con features solo demográficas
4. 🔧 Esperar accuracy realista: 60-75% (mucho más valioso científicamente)

---

**Última actualización**: 2025-10-20  
**Siguiente revisión**: Después de reunión con asesor
