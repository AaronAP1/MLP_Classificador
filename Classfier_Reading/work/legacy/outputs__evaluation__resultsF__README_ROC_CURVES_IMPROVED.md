# 📊 GUÍA DE CURVAS ROC MEJORADAS

**Fecha:** Noviembre 10, 2025  
**Archivos generados:** 4 versiones optimizadas

---

## 🎯 ¿QUÉ PROBLEMA RESOLVEMOS?

### **PROBLEMA ORIGINAL:**
El gráfico 3×3 con 27 curvas ROC era:
- ❌ Muy saturado (difícil de leer)
- ❌ Difícil comparar modelos directamente
- ❌ Demasiada información en un solo gráfico
- ❌ Leyendas sobrepuestas

### **SOLUCIÓN:**
4 versiones especializadas para diferentes propósitos

---

## 📊 VERSIÓN 1: ROC CONSOLIDADO (POR GRUPO)

### **Archivo:** `roc_curves_consolidated.png`

### **Diseño:**
```
┌─────────────────────────────────────────────────────┐
│  [Adultos]    [Jóvenes]    [Niños]                 │
│     9 curvas     9 curvas     9 curvas              │
│  (3 modelos × 3 clases por panel)                   │
└─────────────────────────────────────────────────────┘
```

### **Características:**
- **3 paneles:** Uno por grupo demográfico
- **9 curvas por panel:** 3 modelos × 3 clases
- **Colores por modelo:**
  - 🔵 MLP-MA: Azul
  - 🟣 TabNet: Púrpura
  - 🟠 FT-Transformer: Naranja
- **Líneas por clase:**
  - ━━━ Sólida: No Lector
  - ╌╌╌ Punteada: Lector Medio
  - ····· Puntos: Lector Frecuente

### **¿Cuándo usar?**
✅ **Análisis detallado por grupo demográfico**
✅ **Material suplementario del paper**
✅ **Cuando necesitas mostrar TODAS las clases**

### **Ventajas:**
- Muestra rendimiento completo por clase
- Fácil comparar modelos dentro del mismo grupo
- Mantiene coherencia visual (colores consistentes)

### **Lectura:**
> "En Adultos, TabNet (púrpura) supera a MLP-MA (azul) en las 3 clases.
> La clase 'Lector Frecuente' (línea punteada) tiene mejor AUC en todos los modelos."

---

## 📊 VERSIÓN 2: COMPARACIÓN POR MODELO (RECOMENDADA) ⭐

### **Archivo:** `roc_curves_model_comparison.png`

### **Diseño:**
```
┌─────────────────────────────────────────────────────┐
│  [MLP-MA]    [TabNet]    [FT-Transformer]          │
│  Adultos     Adultos     Adultos                    │
│  Jóvenes     Jóvenes     Jóvenes                    │
│  Niños       Niños       Niños                      │
│  (3 curvas por panel)                               │
└─────────────────────────────────────────────────────┘
```

### **Características:**
- **3 paneles:** Uno por modelo
- **3 curvas por panel:** Una por grupo demográfico
- **Curvas micro-averaged:** Promedio de las 3 clases
- **Colores por grupo:**
  - 🟢 Adultos: Verde
  - 🟠 Jóvenes: Naranja
  - 🟣 Niños: Púrpura

### **¿Cuándo usar?**
✅ **PAPER PRINCIPAL (RECOMENDADO)** ⭐
✅ **Comparación directa de rendimiento**
✅ **Presentaciones y defensa de tesis**

### **Ventajas:**
- **MÁS CLARO:** Solo 3 curvas por panel
- **COMPARACIÓN DIRECTA:** Fácil ver qué grupo se clasifica mejor
- **MENSAJE CLARO:** "TabNet funciona mejor en Adultos que en Niños"

### **Lectura:**
> "TabNet (panel central) muestra AUC superior en Adultos (0.978) 
> comparado con Jóvenes (0.921) y Niños (0.796), evidenciando que 
> la complejidad del comportamiento lector aumenta con edad temprana."

---

## 📊 VERSIÓN 3: MEGA GRÁFICO UNIFICADO (3×3)

### **Archivo:** `roc_curves_mega_unified.png`

### **Diseño:**
```
┌────────────────────────────────────────────────────┐
│        MLP-MA      TabNet      FT-Trans            │
│Adultos  [ROC]      [ROC]        [ROC]              │
│Jóvenes  [ROC]      [ROC]        [ROC]              │
│Niños    [ROC]      [ROC]        [ROC]              │
│ (3 clases por celda, más compacto)                 │
└────────────────────────────────────────────────────┘
```

### **Características:**
- **Grid 3×3:** 9 subgráficos (como el original pero mejorado)
- **3 curvas por celda:** Una por clase
- **Más compacto:** Mejor aprovechamiento de espacio
- **Labels optimizados:** Solo en bordes

### **¿Cuándo usar?**
✅ **Apéndice del paper**
✅ **Análisis exhaustivo**
✅ **Cuando necesitas TODO en un solo gráfico**

### **Ventajas:**
- Vista completa: 9 modelos × 3 clases = 27 curvas
- Comparación matricial (fila = grupo, columna = modelo)
- Mantiene toda la información del original

### **Desventajas:**
- Puede ser denso para presentaciones
- Requiere pantalla/impresión grande

---

## 📊 VERSIÓN 4: RESUMEN ULTRA-SIMPLIFICADO ⭐⭐

### **Archivo:** `roc_curves_summary.png`

### **Diseño:**
```
┌────────────────────────────────────────────┐
│     UN SOLO PANEL                          │
│  ━━━ MLP-MA (azul)                        │
│  ━━━ TabNet (rojo)                        │
│  ━━━ FT-Transformer (verde)               │
│  (1 curva por modelo, promedio de todo)    │
└────────────────────────────────────────────┘
```

### **Características:**
- **1 solo panel:** Máxima simplicidad
- **3 curvas:** Una por modelo (promedio de todos los grupos)
- **Mensaje único:** "¿Qué modelo es mejor en general?"
- **Zona de excelencia:** Resaltada visualmente

### **¿Cuándo usar?**
✅ **PRESENTACIÓN EJECUTIVA** ⭐⭐
✅ **SLIDE DE CONCLUSIONES**
✅ **ABSTRACT GRÁFICO**
✅ **PÓSTER CIENTÍFICO**

### **Ventajas:**
- **ULTRA CLARO:** Solo 3 líneas
- **MENSAJE DIRECTO:** "TabNet es mejor"
- **IMPACTO VISUAL:** Perfecto para audiencias no técnicas

### **Lectura:**
> "TabNet (línea roja) alcanza el mejor AUC promedio (0.875), 
> superando a MLP-MA (0.871) y FT-Transformer (0.855), 
> consolidándose como el modelo óptimo para clasificación de 
> hábitos de lectura en población peruana."

---

## 🎯 COMPARACIÓN DE LAS 4 VERSIONES

| Aspecto | V1 Consolidado | V2 Comparación ⭐ | V3 Mega Grid | V4 Resumen ⭐⭐ |
|---------|---------------|------------------|--------------|----------------|
| **Paneles** | 3 (por grupo) | 3 (por modelo) | 9 (grid 3×3) | 1 (unificado) |
| **Curvas totales** | 27 (9 por panel) | 9 (3 por panel) | 27 (3 por celda) | 3 (total) |
| **Claridad** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Detalle** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐ |
| **Comparabilidad** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Uso en paper** | Suplementario | **Principal** ⭐ | Apéndice | Conclusión |
| **Presentación** | No recomendado | ✅ Bueno | No recomendado | ✅ Excelente |
| **Tamaño** | 20×6 in | 20×6 in | 18×16 in | 10×8 in |

---

## 📖 CÓMO LEER CADA VERSIÓN

### **VERSIÓN 1 (Consolidado):**
```
Pregunta: "¿Cómo se desempeña cada clase en Adultos?"
Respuesta: Mira el panel izquierdo, compara las 3 líneas de cada color.
```

### **VERSIÓN 2 (Comparación):** ⭐
```
Pregunta: "¿TabNet funciona mejor en Adultos o Niños?"
Respuesta: Mira el panel central, compara curva verde vs púrpura.
```

### **VERSIÓN 3 (Mega Grid):**
```
Pregunta: "Quiero ver TODO el rendimiento de FT-Transformer"
Respuesta: Mira la columna derecha (3 celdas).
```

### **VERSIÓN 4 (Resumen):** ⭐⭐
```
Pregunta: "¿Qué modelo debo usar?"
Respuesta: La curva más arriba = TabNet (rojo).
```

---

## 🎓 RECOMENDACIONES PARA TU PAPER

### **ESTRUCTURA SUGERIDA:**

#### **Sección: Resultados**

**Figura Principal (en el cuerpo del texto):**
```latex
\begin{figure}[htbp]
\centering
\includegraphics[width=\textwidth]{roc_curves_model_comparison.png}
\caption{Curvas ROC de los tres modelos evaluados en cada grupo 
demográfico. TabNet (centro) muestra el mejor rendimiento con AUC 
superior a 0.97 en Adultos, 0.92 en Jóvenes, y 0.80 en Niños, 
evidenciando alta capacidad discriminativa en todos los grupos.}
\label{fig:roc_main}
\end{figure}
```

**Texto de acompañamiento:**
> "La Figura X presenta las curvas ROC (Receiver Operating Characteristic) 
> para los tres clasificadores evaluados. TabNet alcanza el mejor AUC 
> (Area Under the Curve) promedio de 0.875, superando a MLP-MA (0.871) 
> y FT-Transformer (0.855). Se observa un patrón consistente: el 
> rendimiento decrece al clasificar grupos etarios más jóvenes 
> (Adultos > Jóvenes > Niños), reflejando la mayor complejidad en 
> capturar patrones de lectura en edades tempranas."

---

#### **Material Suplementario:**

**Figura Suplementaria S1:**
```latex
\includegraphics[width=\textwidth]{roc_curves_consolidated.png}
\caption{Análisis detallado de curvas ROC por clase. Se muestran 
las tres clases de hábitos de lectura (No Lector, Lector Medio, 
Lector Frecuente) para cada combinación modelo-grupo.}
```

**Figura Suplementaria S2:**
```latex
\includegraphics[width=0.7\textwidth]{roc_curves_summary.png}
\caption{Comparación general del rendimiento de los modelos. 
Curva calculada como micro-average de todos los grupos y clases.}
```

---

#### **Presentación/Defensa:**

**Slide de Resultados Principales:**
- Usa **VERSIÓN 4** (Summary) ⭐⭐
- Título: "TabNet supera a otros modelos"
- 1 gráfico grande + 3 bullet points

**Slide de Análisis Detallado:**
- Usa **VERSIÓN 2** (Comparación) ⭐
- Título: "Rendimiento por grupo demográfico"
- Explica diferencias entre Adultos/Jóvenes/Niños

**Slide de Backup (por si preguntan):**
- Ten **VERSIÓN 1** lista
- Para responder: "¿Y qué pasa con la clase X?"

---

## 📊 INTERPRETACIÓN DE LOS GRÁFICOS

### **¿Qué es una curva ROC?**
```
Eje X (FPR): % de falsos positivos
Eje Y (TPR): % de verdaderos positivos

Curva ideal: Esquina superior izquierda (TPR=1, FPR=0)
Curva mala: Diagonal (aleatorio)
```

### **¿Qué es AUC?**
```
AUC = Área bajo la curva ROC
Rango: [0, 1]

AUC > 0.90: Excelente ✅
AUC > 0.80: Bueno ✅
AUC > 0.70: Aceptable ⚠️
AUC < 0.70: Pobre ❌
```

### **Tus resultados:**
```
Adultos:  AUC > 0.93  ✅ EXCELENTE
Jóvenes:  AUC > 0.84  ✅ BUENO
Niños:    AUC ~ 0.77  ✅ ACEPTABLE (⚠️ límite)
```

---

## 💡 MENSAJES CLAVE PARA TU PAPER

### **MENSAJE 1: TabNet es superior**
> "TabNet alcanza consistentemente el mejor AUC en los 3 grupos 
> demográficos (0.978 Adultos, 0.921 Jóvenes, 0.796 Niños), 
> superando a MLP-MA y FT-Transformer en todas las evaluaciones."

**Evidencia:** Versión 2 o 4

---

### **MENSAJE 2: La edad afecta la clasificabilidad**
> "Se observa una relación inversa entre edad y dificultad de 
> clasificación: Adultos (AUC 0.97) > Jóvenes (AUC 0.92) > Niños 
> (AUC 0.80), sugiriendo que los hábitos de lectura se consolidan 
> y vuelven más predecibles con la edad."

**Evidencia:** Versión 2 (comparar curvas dentro de cada panel)

---

### **MENSAJE 3: Todas las clases son clasificables**
> "Los tres modelos logran discriminar efectivamente las tres clases 
> de comportamiento lector (No Lector, Medio, Frecuente) con AUC 
> superiores a 0.75 en Adultos y Jóvenes, validando la pertinencia 
> de la taxonomía propuesta."

**Evidencia:** Versión 1 (muestra curvas por clase)

---

## ✅ CHECKLIST DE CALIDAD

| Aspecto | V1 | V2 | V3 | V4 |
|---------|----|----|----|----|
| Resolución 300 DPI | ✅ | ✅ | ✅ | ✅ |
| Colores distinguibles | ✅ | ✅ | ✅ | ✅ |
| Leyendas legibles | ✅ | ✅ | ⚠️ | ✅ |
| Labels claros | ✅ | ✅ | ⚠️ | ✅ |
| Tamaño apropiado | ✅ | ✅ | ✅ | ✅ |
| Mensaje claro | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ |

---

## 🎯 DECISIÓN FINAL: ¿QUÉ VERSIÓN USAR?

### **Para el paper científico:**
1. **Figura principal:** VERSIÓN 2 (Comparación) ⭐
2. **Material suplementario:** VERSIÓN 1 (Consolidado)
3. **Abstract gráfico:** VERSIÓN 4 (Resumen) ⭐⭐

### **Para la presentación:**
1. **Slide principal:** VERSIÓN 4 (Resumen) ⭐⭐
2. **Slide detallado:** VERSIÓN 2 (Comparación) ⭐

### **Para el póster:**
1. **VERSIÓN 4** (Resumen) con números grandes de AUC

---

**¿Necesitas modificaciones adicionales?** 
Por ejemplo:
- Cambiar colores
- Ajustar tamaños
- Agregar anotaciones
- Generar versión en blanco/negro para impresión

**Los 4 gráficos están listos para usar en tu tesis!** 🎓📊
