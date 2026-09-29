# -*- coding: utf-8 -*-
# Índices = párrafos de nivel superior en "Manuscript IJIET - Propuesta revision Turnitin.docx" (versión guardada por Word).

EDITS = [
    # Resumen: SMOTE no se describe en Métodos
    (9, "sub", "SMOTE, ", ""),
    # Introducción: dato ENL 2022 por lengua materna (informe INEI/MINCUL, ref. [1], p. 84)
    (15, "ins", "es amplia [1].", " La brecha también aparece según la lengua materna: entre los adultos alfabetizados de 18 a 64 años, leyó libros el 49,6 % de quienes aprendieron a hablar en castellano, frente al 33,1 % de quienes lo hicieron en una lengua nativa [1]."),
    # Introducción: no hay comparación con métodos tradicionales en Resultados
    (17, "sub", " y contrastadas con métodos tradicionales de aprendizaje automático", ""),
    # Trabajos relacionados: ya no se afirma balanceo de clases
    (21, "sub", "balancear las clases", "tener en cuenta el desbalance de clases"),
    # Métodos: fuente de los datos
    (45, "sub", "que recogen", "que el Ministerio de Cultura y el INEI publican en la Plataforma Nacional de Datos Abiertos del Perú [31]. Estos microdatos recogen"),
    # Métodos: la partición 70/15/15 queda solo en "División de datos"
    (48, "sub", " Cada subgrupo se partió en 70 % para entrenamiento, 15 % para validación y 15 % para prueba [11], [15], [16].", ""),
    (198, "sub", "El dataset final fue dividido en", "Cada subconjunto etario (niños, jóvenes y adultos) se dividió en"),
    # Métodos: no existe Apéndice A
    (86, "sub", " Para que el experimento pueda reproducirse, el Apéndice A presenta el pseudocódigo del pipeline completo, desde el preprocesamiento hasta la comparación final de los modelos.", ""),
    # Métodos: hardware indicado por los autores
    (200, "sub", "utilizando GPU para", "utilizando una GPU NVIDIA GeForce RTX 4060 de 8 GB para"),
    # Resultados: conteo según la Table 3
    (346, "sub", "11 de las 15", "14 de las 15"),
    (377, "sub", "11", "14"),
    (378, "sub", "MLP+MA lideró en 4,", "MLP+MA lideró en 1 (Recall-Weighted en adultos),"),
    # Resultados: FT-Transformer no supera a MLP+MA en F1-Macro en adultos
    (391, "sub", "El transformer supera ligeramente a MLP+MA en adultos en F1-Macro, pero:", "En adultos, FT-Transformer queda por debajo de MLP+MA en F1-Macro (0.8478 frente a 0.8671), y además:"),
    (393, "sub", "cae hasta 0.686 en niños,", "su F1-Weighted cae hasta 0.686 en niños,"),
    (397, "sub", "desempeño óptimo.", "desempeño óptimo [32]."),
    # Resultados: clases con nombre inexistente
    (425, "sub", "Las confusiones más frecuentes ocurren entre Lector Ocasional y Lector Medio, particularmente en MLP+MA y FT-Transformer.", "Los errores se reparten entre las tres clases, sin que un par concentre la mayoría de las confusiones (Fig. 8)."),
    (434, "sub", "0.6786", "0.5540"),
    (368, "sub", "necesitan muestras más grandes y datos más homogéneos para abstraer patrones de forma adecuada.", "necesitan muestras más grandes y son más sensibles a las variables poco informativas [32]."),
    # Discusión
    (508, "sub", "11 de las 15", "14 de las 15"),
    (509, "sub", "sin estructuras complejas ni un costo computacional alto.", "con una estructura más simple que la de los otros dos modelos y entrenándose, como ellos, en una GPU de consumo (NVIDIA GeForce RTX 4060 de 8 GB)."),
    (510, "sub", "Esto coincide con estudios previos según los cuales los Transformers tabulares necesitan muestras grandes, distribuciones de baja entropía y datos homogéneos para estabilizar sus embeddings.", "Esto coincide con estudios previos que muestran que los modelos profundos para datos tabulares pierden ventaja con muestras de tamaño mediano y con variables poco informativas [32]."),
    (536, "sub", "identificar a los grupos con menor hábito lector y priorizarlos.", "identificar a los grupos con menor hábito lector y priorizarlos en iniciativas de promoción de la lectura como El Perú Lee, del Ministerio de Educación."),
    (537, "sub", "el uso de Grid Search, validación cruzada estratificada y varias métricas multiclase hace que el experimento sea reproducible y comparable con trabajos futuros. Su transferencia a otros contextos educativos, sin embargo, todavía tendría que comprobarse con datos externos.", "el estudio trabaja con microdatos públicos de la ENL, de modo que otros equipos pueden contrastar sus resultados con los de trabajos futuros. Que los modelos funcionen igual de bien en otros contextos educativos todavía debe comprobarse con datos externos."),
    # Conclusiones
    (540, "sub", "11 de las 15", "14 de las 15"),
]

# "Finalmente, cada conjunto fue dividido en: 70 % / 15 % / 15 % … manteniendo la proporción…" (repetido)
DELETE = [77, 78, 79, 80, 81, 82, 83]

NEW_REFS = (578, [
    "Ministerio de Cultura del Perú and Instituto Nacional de Estadística e Informática, “Encuesta Nacional de Lectura,” *Plataforma Nacional de Datos Abiertos*, 2023. [Online]. Available: https://www.datosabiertos.gob.pe/dataset/encuesta-nacional-de-lectura",
    "L. Grinsztajn, E. Oyallon, and G. Varoquaux, “Why do tree-based models still outperform deep learning on typical tabular data?,” in *Proc. Advances in Neural Information Processing Systems (NeurIPS)*, vol. 35, 2022, pp. 507–520.",
])

REPLIES = [
    (1, "Hecho: quité SMOTE del resumen. También cambié 'balancear las clases' por 'tener en cuenta el desbalance de clases' al final de Trabajos relacionados, porque el artículo ya no dice que se haya aplicado un balanceo."),
    (9, "Añadí un dato del mismo informe de la ENL 2022 (su referencia [1], p. 84): leyó libros el 49,6 % de los adultos con lengua materna castellana frente al 33,1 % de quienes tienen una lengua nativa (población alfabetizada de 18 a 64 años). Conecta con su hallazgo sobre el idioma materno."),
    (13, "Quité la frase. No agregué la comparación: habría que entrenar Random Forest o XGBoost con sus datos y describir cómo se hizo. En el respaldo legacy_backup_MLP_Py_In01.zip hay imágenes 'comparison_traditional_vs_dl' (nov-2025), pero sin el código ni las métricas no puedo verificar de dónde salen."),
    (25, "Añadí la fuente en el texto y como nueva referencia [31] (Plataforma Nacional de Datos Abiertos; Ministerio de Cultura e INEI, 2023). Queda pendiente el número de registros usados en cada grupo."),
    (28, "Hecho: la partición 70/15/15 queda solo en 'División de datos y configuración experimental'; eliminé la repetición aquí y en 'Construcción del dataset final'. Aparte, revisen los rangos de edad: los CSV usados (adultos.csv, jovenes.csv, ninos.csv) tienen niños de 0–15 años, jóvenes de 16–27 y adultos de 28–99; el texto dice 3–13, 14–17 y 18+, y la Fig. 9 dice 0–11, 12–17 y 18+. Deben confirmar cuáles se usaron realmente."),
    (31, "No los agregué. La configuración guardada del experimento de nov-2025 (data/processed/*_config.json) usa One-Hot y StandardScaler, no Label Encoding ni Min-Max, y no registra umbral de varianza ni de correlación; en niños quedan 131 variables, no 71. Los valores 0.005 y 0.95 del código src/aligned se escribieron en agosto a partir del artículo y no produjeron la Table 3. Esta sección debería describir lo que realmente se hizo."),
    (33, "No lo agregué: ninguno de los cuatro índices aparece entre las variables del experimento de nov-2025 (solo hay preguntas originales codificadas). Además, las notas del código src/aligned dicen que el índice de acceso digital es aproximado y que la frecuencia lectora semanal no puede usarse como variable. Si los índices no se usaron, conviene quitar esta subsección."),
    (35, "Hecho: eliminé la referencia al Apéndice A."),
    (39, "Corregido a 14 de 15, también en la lista de más abajo (TabNet 14, MLP+MA 1), en la Discusión y en las Conclusiones."),
    (41, "El texto ya está corregido. No puedo agregar media ± desviación estándar: en los archivos no hay resultados de las 5 semillas ni de los 5 folds (la 'Std' de statistical_summary.csv es entre grupos, no entre ejecuciones). Si esa evaluación no se hizo, habría que quitar 'Evaluación de estabilidad' de Métodos."),
    (43, "Verifiqué las cifras con la Table 3: están bien. Se queda así."),
    (45, "Añadí la cita [32] (Grinsztajn et al., NeurIPS 2022) y ajusté la frase a lo que ese trabajo muestra: con muestras medianas y variables poco informativas, los modelos profundos pierden frente a los de árboles."),
    (47, "Corregido: en adultos FT-Transformer queda por debajo de MLP+MA en F1-Macro. También corregí la viñeta '0.686 en niños', que corresponde a F1-Weighted y no a F1-Macro (el F1-Macro de FT en niños es 0.5174)."),
    (49, "Corregido. En la Fig. 8 los errores de los jóvenes se reparten entre las tres clases (entre 2 % y 6 % por celda) sin un par dominante; lo describí así y quité 'Lector Ocasional'."),
    (51, "Corregido a 0.5540, como en la Table 3."),
    (53, "No lo cambié porque hay tres versiones distintas. (1) Los AUC del texto (0.971–0.978, 0.937–0.954, 0.762–0.807) coinciden con all_roc_curves.png de la carpeta resultsF, la misma de la Table 3. (2) La Fig. 9 del artículo es otra imagen (roc_curves_consolidated.png, 10-nov) con AUC de 0.958–0.998 y otras edades. (3) La Table 4 (0.99 / 0.98 / 0.87) no coincide con ninguna figura que encontré. Además, según all_roc_curves.png, FT-Transformer no tiene el AUC más bajo en niños (lo tiene MLP+MA), así que también habría que revisar las frases que lo afirman. Tienen que decidir qué corrida es la válida y usar su figura y su tabla."),
    (55, "No lo agregué: tendría que leer valores a ojo en la Fig. 11, cuyo origen está en duda (ver comentario en la Table 3). Si tienen el registro del entrenamiento, el dato sale de ahí."),
    (57, "Corregido a 14 de 15."),
    (59, "Añadí la GPU aquí y en Métodos ('Todos los modelos se implementaron en PyTorch 2.6.0…'). Ojo: los registros de entrenamiento de nov-2025 que encontré dicen 'Usando device: cpu'. Confirmen en qué equipo se hizo la corrida final."),
    (61, "Corregido: añadí la cita [32] y quité 'baja entropía' y 'estabilizar sus embeddings', que no estaban respaldados."),
    (63, "Revisado: 'Accuracy > 0.96' coincide con la Table 3 y 'AUC ≈ 0.97' con las curvas ROC de resultsF (0.958–0.978 por clase). Si eligen otra versión del AUC, hay que ajustarlo (ver comentario en Curvas ROC)."),
    (65, "Listo: ya no queda ninguna mención a la entropía de los datos (también la quité del párrafo sobre FT-Transformer). Solo queda 'entropía cruzada', que es la función de pérdida y es correcta."),
    (69, "Ojo: 'Perú sí lee' es la campaña de la Iglesia Adventista que reparte el libro 'Esperanza para un mundo en crisis'; no es un programa de promoción de la lectura, y en un artículo sobre política educativa podría generar objeciones. Usé 'El Perú Lee', la estrategia del Ministerio de Educación para la educación básica. Si prefieren otra, cámbienla."),
    (71, "Corregido: ya no dice que el experimento sea reproducible, porque la metodología descrita no coincide con la configuración guardada (ver comentarios en Métodos). Ahora destaca que los datos son públicos y deja la transferencia a otros contextos como algo por comprobar."),
    (73, "Revisado: ya no queda ninguna afirmación de que se superen los métodos tradicionales (también la quité de la Introducción). En Trabajos relacionados, 'tradicionales' solo aparece al describir otros estudios, lo cual es correcto."),
]

NEW_COMMENTS = [
    (342, "Verificar el origen de esta tabla antes de enviar. Sus valores coinciden exactamente con outputs/evaluation/resultsF/comprehensive_metrics.csv (legacy_backup_MLP_Py_In01.zip, 3-nov-2025), cuyo README indica que se regenera con generate_resultsF.py ajustando 'Metric ranges' y la semilla aleatoria. Los registros de entrenamiento del mismo día (results/baseline) dan valores muy distintos (p. ej., MLP+MA en niños: accuracy 0.488) y el entrenamiento 'fase2' del 19-nov no generó modelos. Si existe un entrenamiento real que produzca estas cifras, conviene guardar su script y sus registros junto al manuscrito."),
    (69, "Posible fuga de información: si la clase (No lector / Lector medio / Lector frecuente) se define a partir de la frecuencia de lectura, usar la frecuencia lectora semanal como variable de entrada le da al modelo parte de la respuesta. Las notas del código src/aligned la excluyen por ese motivo. Revisen cómo se definió la clase y si esta variable entró en los modelos; en la Discusión aparece como una de las más influyentes."),
]
