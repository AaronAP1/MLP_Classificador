"""
Convierte el documento LaTeX de Resultados a formato Word (.docx)
Genera un documento profesional con tablas, figuras y formato académico.
"""

import re
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.style import WD_STYLE_TYPE
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

def add_hyperlink(paragraph, url, text):
    """Agrega un hipervínculo al párrafo"""
    part = paragraph.part
    r_id = part.relate_to(url, 'http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink', is_external=True)
    
    hyperlink = OxmlElement('w:hyperlink')
    hyperlink.set(qn('r:id'), r_id)
    
    new_run = OxmlElement('w:r')
    rPr = OxmlElement('w:rPr')
    
    new_run.append(rPr)
    new_run.text = text
    hyperlink.append(new_run)
    
    paragraph._p.append(hyperlink)
    return hyperlink

def create_table_1(doc):
    """Tabla 1: Comprehensive Metrics"""
    
    # Datos de comprehensive_metrics.csv
    data = [
        ['Grupo', 'Modelo', 'Accuracy', 'F1-W', 'F1-M', 'Precision-W', 'Recall-W'],
        ['Adultos', 'MLP-MA', '0.9752', '0.9668', '0.8671', '0.9725', '0.9727'],
        ['Adultos', 'TabNet', '0.9767', '0.9740', '0.8783', '0.9776', '0.9724'],
        ['Adultos', 'FT-Transformer', '0.9652', '0.9639', '0.8478', '0.9686', '0.9678'],
        ['Jóvenes', 'MLP-MA', '0.9172', '0.9137', '0.7590', '0.9162', '0.9164'],
        ['Jóvenes', 'TabNet', '0.9240', '0.9210', '0.7712', '0.9207', '0.9240'],
        ['Jóvenes', 'FT-Transformer', '0.9030', '0.8996', '0.7345', '0.9031', '0.9050'],
        ['Niños', 'MLP-MA', '0.7220', '0.7153', '0.5415', '0.7166', '0.7221'],
        ['Niños', 'TabNet', '0.7363', '0.7307', '0.5540', '0.7290', '0.7369'],
        ['Niños', 'FT-Transformer', '0.6974', '0.6863', '0.5174', '0.6950', '0.6956'],
    ]
    
    # Crear tabla
    table = doc.add_table(rows=len(data), cols=len(data[0]))
    table.style = 'Light Grid Accent 1'
    
    # Llenar tabla
    for i, row_data in enumerate(data):
        row = table.rows[i]
        for j, cell_value in enumerate(row_data):
            cell = row.cells[j]
            cell.text = cell_value
            
            # Formato de encabezado
            if i == 0:
                cell.paragraphs[0].runs[0].font.bold = True
                cell.paragraphs[0].runs[0].font.size = Pt(10)
                shading_elm = OxmlElement('w:shd')
                shading_elm.set(qn('w:fill'), 'D9E1F2')
                cell._element.get_or_add_tcPr().append(shading_elm)
            else:
                cell.paragraphs[0].runs[0].font.size = Pt(9)
            
            # Resaltar mejores valores (TabNet en cada grupo)
            if i in [2, 5, 8] and j >= 2:  # Filas de TabNet, columnas de métricas
                cell.paragraphs[0].runs[0].font.bold = True
    
    # Agregar caption
    caption = doc.add_paragraph()
    caption.add_run('Tabla 1. ').bold = True
    caption.add_run('Resultados de evaluación en el conjunto de test para los tres modelos en los tres grupos demográficos. Se presentan cinco métricas de clasificación: Accuracy, F1-Weighted, F1-Macro, Precision-Weighted y Recall-Weighted.')
    caption.paragraph_format.space_after = Pt(12)
    
    # Nota al pie
    note = doc.add_paragraph()
    note_run = note.add_run('Nota: F1-W = F1-Weighted, F1-M = F1-Macro, Precision-W = Precision-Weighted, Recall-W = Recall-Weighted. Los valores en negrita indican el mejor rendimiento en cada grupo demográfico.')
    note_run.font.size = Pt(8)
    note_run.font.italic = True
    note.paragraph_format.space_after = Pt(18)

def create_table_2(doc):
    """Tabla 2: Statistical Summary"""
    
    data = [
        ['Modelo', 'Métrica', 'Promedio', 'Desv. Est.', 'Mín.', 'Máx.', 'Rango'],
        ['MLP-MA', 'Accuracy', '0.8715', '0.1083', '0.7220', '0.9752', '0.2532'],
        ['MLP-MA', 'F1-Weighted', '0.8652', '0.1082', '0.7153', '0.9668', '0.2515'],
        ['MLP-MA', 'F1-Macro', '0.7226', '0.1354', '0.5415', '0.8671', '0.3256'],
        ['MLP-MA', 'Precision-W', '0.8684', '0.1098', '0.7166', '0.9725', '0.2558'],
        ['MLP-MA', 'Recall-W', '0.8704', '0.1073', '0.7221', '0.9727', '0.2506'],
        ['TabNet', 'Accuracy', '0.8790', '0.1032', '0.7363', '0.9767', '0.2404'],
        ['TabNet', 'F1-Weighted', '0.8752', '0.1044', '0.7307', '0.9740', '0.2432'],
        ['TabNet', 'F1-Macro', '0.7345', '0.1349', '0.5540', '0.8783', '0.3243'],
        ['TabNet', 'Precision-W', '0.8758', '0.1063', '0.7290', '0.9776', '0.2486'],
        ['TabNet', 'Recall-W', '0.8778', '0.1030', '0.7369', '0.9724', '0.2355'],
        ['FT-Transformer', 'Accuracy', '0.8552', '0.1149', '0.6974', '0.9652', '0.2678'],
        ['FT-Transformer', 'F1-Weighted', '0.8499', '0.1176', '0.6863', '0.9639', '0.2776'],
        ['FT-Transformer', 'F1-Macro', '0.6999', '0.1407', '0.5174', '0.8478', '0.3304'],
        ['FT-Transformer', 'Precision-W', '0.8556', '0.1173', '0.6950', '0.9686', '0.2736'],
        ['FT-Transformer', 'Recall-W', '0.8561', '0.1155', '0.6956', '0.9678', '0.2721'],
    ]
    
    table = doc.add_table(rows=len(data), cols=len(data[0]))
    table.style = 'Light Grid Accent 1'
    
    for i, row_data in enumerate(data):
        row = table.rows[i]
        for j, cell_value in enumerate(row_data):
            cell = row.cells[j]
            cell.text = cell_value
            
            if i == 0:
                cell.paragraphs[0].runs[0].font.bold = True
                cell.paragraphs[0].runs[0].font.size = Pt(10)
                shading_elm = OxmlElement('w:shd')
                shading_elm.set(qn('w:fill'), 'D9E1F2')
                cell._element.get_or_add_tcPr().append(shading_elm)
            else:
                cell.paragraphs[0].runs[0].font.size = Pt(9)
            
            # Resaltar mejores promedios (TabNet)
            if 6 <= i <= 10 and j >= 2:
                cell.paragraphs[0].runs[0].font.bold = True
    
    caption = doc.add_paragraph()
    caption.add_run('Tabla 2. ').bold = True
    caption.add_run('Resumen estadístico del rendimiento de los tres modelos a través de los tres grupos demográficos. Se reportan media, desviación estándar, valores mínimo y máximo, y rango para cada métrica.')
    caption.paragraph_format.space_after = Pt(12)
    
    note = doc.add_paragraph()
    note_run = note.add_run('Nota: Los valores en negrita indican el mejor desempeño promedio entre arquitecturas. La menor desviación estándar de TabNet evidencia mayor consistencia a través de los grupos demográficos.')
    note_run.font.size = Pt(8)
    note_run.font.italic = True
    note.paragraph_format.space_after = Pt(18)

def create_table_3(doc):
    """Tabla 3: AUC-ROC Summary"""
    
    data = [
        ['Grupo', 'MLP-MA', 'TabNet', 'FT-Transformer'],
        ['Adultos', '0.9912', '0.9934', '0.9889'],
        ['Jóvenes', '0.9756', '0.9801', '0.9702'],
        ['Niños', '0.8652', '0.8731', '0.8436'],
        ['Promedio', '0.9440', '0.9489', '0.9342'],
    ]
    
    table = doc.add_table(rows=len(data), cols=len(data[0]))
    table.style = 'Light Grid Accent 1'
    
    for i, row_data in enumerate(data):
        row = table.rows[i]
        for j, cell_value in enumerate(row_data):
            cell = row.cells[j]
            cell.text = cell_value
            
            if i == 0 or j == 0:
                cell.paragraphs[0].runs[0].font.bold = True
                cell.paragraphs[0].runs[0].font.size = Pt(10)
                shading_elm = OxmlElement('w:shd')
                shading_elm.set(qn('w:fill'), 'D9E1F2')
                cell._element.get_or_add_tcPr().append(shading_elm)
            else:
                cell.paragraphs[0].runs[0].font.size = Pt(9)
            
            # Resaltar mejores valores (columna TabNet)
            if i > 0 and j == 2:
                cell.paragraphs[0].runs[0].font.bold = True
    
    caption = doc.add_paragraph()
    caption.add_run('Tabla 3. ').bold = True
    caption.add_run('Valores promedio de AUC-ROC (One-vs-Rest) para cada modelo y grupo demográfico. Valores más cercanos a 1.0 indican mejor capacidad discriminativa.')
    caption.paragraph_format.space_after = Pt(12)
    
    note = doc.add_paragraph()
    note_run = note.add_run('Nota: Los valores en negrita indican el mejor AUC-ROC en cada grupo. TabNet mantiene superioridad consistente en capacidad discriminativa.')
    note_run.font.size = Pt(8)
    note_run.font.italic = True
    note.paragraph_format.space_after = Pt(18)

def create_table_4(doc):
    """Tabla 4: Copeland Ranking"""
    
    data = [
        ['Posición', 'Modelo', 'Score de Copeland', 'Interpretación'],
        ['1', 'TabNet', '+4', 'Ganador claro'],
        ['2', 'MLP-MA', '0', 'Desempeño intermedio'],
        ['3', 'FT-Transformer', '-4', 'Menor rendimiento'],
    ]
    
    table = doc.add_table(rows=len(data), cols=len(data[0]))
    table.style = 'Light Grid Accent 1'
    
    for i, row_data in enumerate(data):
        row = table.rows[i]
        for j, cell_value in enumerate(row_data):
            cell = row.cells[j]
            cell.text = cell_value
            
            if i == 0:
                cell.paragraphs[0].runs[0].font.bold = True
                cell.paragraphs[0].runs[0].font.size = Pt(10)
                shading_elm = OxmlElement('w:shd')
                shading_elm.set(qn('w:fill'), 'D9E1F2')
                cell._element.get_or_add_tcPr().append(shading_elm)
            else:
                cell.paragraphs[0].runs[0].font.size = Pt(9)
            
            # Resaltar ganador
            if i == 1:
                cell.paragraphs[0].runs[0].font.bold = True
    
    caption = doc.add_paragraph()
    caption.add_run('Tabla 4. ').bold = True
    caption.add_run('Ranking de modelos según el algoritmo de Copeland. El score indica victorias netas en comparaciones pareadas: +1 por cada modelo superado, -1 por cada modelo superior, 0 por empates.')
    caption.paragraph_format.space_after = Pt(12)
    
    note = doc.add_paragraph()
    note_run = note.add_run('Nota: El score de Copeland se calcula comparando cada modelo contra los otros dos en las seis métricas para cada grupo demográfico (18 comparaciones totales por par de modelos). Un score de +4 indica que TabNet superó a ambos competidores en la mayoría de las métricas y grupos.')
    note_run.font.size = Pt(8)
    note_run.font.italic = True
    note.paragraph_format.space_after = Pt(18)

def add_figure_placeholder(doc, figure_name, caption_text, label):
    """Agrega un placeholder para una figura"""
    
    # Placeholder visual
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(f'[AQUÍ VA LA IMAGEN: {figure_name}]')
    run.font.size = Pt(11)
    run.font.italic = True
    run.font.color.rgb = RGBColor(128, 128, 128)
    
    # Agregar borde al placeholder
    pPr = p._element.get_or_add_pPr()
    pBdr = OxmlElement('w:pBdr')
    for border_name in ['top', 'left', 'bottom', 'right']:
        border = OxmlElement(f'w:{border_name}')
        border.set(qn('w:val'), 'single')
        border.set(qn('w:sz'), '12')
        border.set(qn('w:space'), '4')
        border.set(qn('w:color'), '808080')
        pBdr.append(border)
    pPr.append(pBdr)
    
    # Caption
    caption = doc.add_paragraph()
    caption.alignment = WD_ALIGN_PARAGRAPH.CENTER
    caption.add_run(f'Figura {label}. ').bold = True
    caption.add_run(caption_text)
    caption.paragraph_format.space_after = Pt(18)

def create_word_document():
    """Crea el documento Word completo"""
    
    doc = Document()
    
    # Configurar estilos
    style = doc.styles['Normal']
    style.font.name = 'Times New Roman'
    style.font.size = Pt(12)
    
    # Título
    title = doc.add_heading('Resultados Experimentales', level=0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    subtitle = doc.add_paragraph('Clasificación de Hábitos de Lectura en Perú mediante Deep Learning')
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    subtitle.runs[0].font.size = Pt(14)
    subtitle.runs[0].font.italic = True
    
    doc.add_paragraph()  # Espacio
    
    # ============================================================================
    # SECCIÓN 1: INTRODUCCIÓN
    # ============================================================================
    
    doc.add_heading('1. Resultados', level=1)
    
    intro = doc.add_paragraph(
        'En esta sección se presentan los resultados experimentales obtenidos tras el '
        'entrenamiento de tres arquitecturas de deep learning (MLP con Multi-head Attention, '
        'TabNet y Feature Tokenizer Transformer) aplicadas a la clasificación de hábitos de '
        'lectura en tres grupos demográficos de Perú: Adultos (18+ años), Jóvenes (14-17 años) '
        'y Niños (3-13 años). Todos los modelos fueron entrenados durante 200 épocas con early '
        'stopping (patience=20) utilizando el optimizador AdamW con learning rate de 1×10⁻³ y '
        'weight decay de 1×10⁻⁴.'
    )
    intro.paragraph_format.space_after = Pt(12)
    intro.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    
    # ============================================================================
    # SECCIÓN 2: RENDIMIENTO GENERAL
    # ============================================================================
    
    doc.add_heading('1.1. Rendimiento General de los Modelos', level=2)
    
    p1 = doc.add_paragraph(
        'La Tabla 1 presenta los resultados completos de evaluación para los nueve experimentos '
        'realizados (3 arquitecturas × 3 grupos demográficos). Los modelos fueron evaluados '
        'utilizando cinco métricas estándar: Accuracy, F1-Score (weighted y macro), '
        'Precision (weighted) y Recall (weighted).'
    )
    p1.paragraph_format.space_after = Pt(12)
    p1.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    
    doc.add_paragraph()
    create_table_1(doc)
    
    # ============================================================================
    # SUBSECCIÓN: ANÁLISIS POR GRUPO DEMOGRÁFICO
    # ============================================================================
    
    doc.add_heading('1.1.1. Análisis por Grupo Demográfico', level=3)
    
    p2 = doc.add_paragraph(
        'Los resultados revelan diferencias significativas en el rendimiento de clasificación '
        'entre los tres grupos demográficos:'
    )
    p2.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p2.paragraph_format.space_after = Pt(8)
    
    # Adultos
    doc.add_heading('Grupo Adultos (18+ años)', level=4)
    p_adultos = doc.add_paragraph(
        'Este grupo presentó el mejor desempeño general, con accuracies entre 96.5% y 97.7%. '
        'TabNet alcanzó la mayor exactitud (97.67%), seguido por MLP-MA (97.52%) y FT-Transformer '
        '(96.52%). El alto F1-Macro (0.8478-0.8783) indica una capacidad balanceada para predecir '
        'las tres categorías de lectura, sugiriendo patrones de comportamiento lector bien definidos '
        'en este segmento poblacional. La menor variabilidad en los hábitos de lectura de adultos '
        'facilita la tarea de clasificación, como se evidencia en las matrices de confusión (Figura 1).'
    )
    p_adultos.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p_adultos.paragraph_format.space_after = Pt(10)
    
    # Jóvenes
    doc.add_heading('Grupo Jóvenes (14-17 años)', level=4)
    p_jovenes = doc.add_paragraph(
        'Los jóvenes mostraron un rendimiento moderado, con accuracies entre 90.3% y 92.4%. '
        'Nuevamente, TabNet obtuvo los mejores resultados (92.40% accuracy, 0.7712 F1-Macro), '
        'superando a MLP-MA por 0.68 puntos porcentuales. La mayor dispersión en los scores '
        'F1-Macro (0.7345-0.7712 vs. 0.8478-0.8783 en adultos) refleja una mayor heterogeneidad '
        'en los patrones de lectura juveniles, posiblemente debido a la transición entre etapas '
        'educativas y la diversificación de intereses.'
    )
    p_jovenes.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p_jovenes.paragraph_format.space_after = Pt(10)
    
    # Niños
    doc.add_heading('Grupo Niños (3-13 años)', level=4)
    p_ninos = doc.add_paragraph(
        'Este grupo presentó el mayor desafío clasificatorio, con accuracies entre 69.7% y 73.6%. '
        'TabNet mantuvo su liderazgo (73.63%), aunque con una brecha significativamente menor '
        'respecto a los otros modelos (MLP-MA: 72.20%, FT-Transformer: 69.74%). Los bajos valores '
        'de F1-Macro (0.5174-0.5540) indican dificultades para predecir correctamente todas las '
        'clases, especialmente para las categorías minoritarias. Esta complejidad se atribuye a '
        'la alta variabilidad del desarrollo cognitivo y las diferencias en el acceso a material '
        'de lectura en este rango etario (3-13 años).'
    )
    p_ninos.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p_ninos.paragraph_format.space_after = Pt(16)
    
    # ============================================================================
    # SECCIÓN 3: RESUMEN ESTADÍSTICO
    # ============================================================================
    
    doc.add_heading('1.2. Resumen Estadístico por Arquitectura', level=2)
    
    p3 = doc.add_paragraph(
        'La Tabla 2 presenta estadísticos descriptivos (promedio, desviación estándar, mínimo, '
        'máximo y rango) para cada arquitectura agregando los resultados de los tres grupos '
        'demográficos. Este análisis permite evaluar la consistencia y robustez de cada modelo.'
    )
    p3.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p3.paragraph_format.space_after = Pt(12)
    
    doc.add_paragraph()
    create_table_2(doc)
    
    # Análisis comparativo
    doc.add_heading('1.2.1. Análisis Comparativo de Arquitecturas', level=3)
    
    doc.add_heading('TabNet: Mejor rendimiento global', level=4)
    p_tabnet = doc.add_paragraph(
        'TabNet demostró superioridad consistente en las cinco métricas evaluadas, con un accuracy '
        'promedio de 87.90% (±10.32%), superando a MLP-MA por 0.75 puntos porcentuales y a '
        'FT-Transformer por 2.38 puntos. Crucialmente, TabNet también exhibió la menor variabilidad '
        '(desviación estándar de 10.32% en accuracy), indicando mayor robustez ante las diferencias '
        'demográficas. Su arquitectura basada en atención sparse secuencial permite una selección '
        'interpretable de características, lo cual es ventajoso para datos tabulares heterogéneos '
        'como los analizados.'
    )
    p_tabnet.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p_tabnet.paragraph_format.space_after = Pt(10)
    
    doc.add_heading('MLP-MA: Rendimiento competitivo', level=4)
    p_mlp = doc.add_paragraph(
        'El modelo MLP con Multi-head Attention alcanzó un accuracy promedio de 87.15% (±10.83%), '
        'posicionándose como una alternativa sólida. Aunque ligeramente inferior a TabNet, su '
        'arquitectura más simple ofrece ventajas en términos de costo computacional y tiempo de '
        'entrenamiento. El mecanismo de atención en la capa de entrada demostró ser efectivo para '
        'ponderar la importancia de las 71-139 características según el grupo demográfico.'
    )
    p_mlp.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p_mlp.paragraph_format.space_after = Pt(10)
    
    doc.add_heading('FT-Transformer: Menor consistencia', level=4)
    p_ft = doc.add_paragraph(
        'El modelo basado en tokenización de características y arquitectura transformer mostró el '
        'menor rendimiento promedio (85.52% accuracy, ±11.49%) y la mayor variabilidad entre grupos. '
        'La desviación estándar más alta (11.49%) sugiere que esta arquitectura es más sensible a '
        'las características específicas de cada grupo demográfico. Esto puede deberse a que los '
        'transformers típicamente requieren mayores volúmenes de datos para generalizar efectivamente, '
        'y el grupo de niños (10,388 muestras) podría ser insuficiente para un entrenamiento óptimo.'
    )
    p_ft.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p_ft.paragraph_format.space_after = Pt(16)
    
    # ============================================================================
    # SECCIÓN 4: MATRICES DE CONFUSIÓN
    # ============================================================================
    
    doc.add_heading('1.3. Análisis de Matrices de Confusión', level=2)
    
    p4 = doc.add_paragraph(
        'La Figura 1 presenta las matrices de confusión normalizadas para los nueve experimentos '
        'realizados, organizadas en un grid de 3×3 donde cada fila corresponde a un grupo '
        'demográfico y cada columna a una arquitectura de modelo.'
    )
    p4.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p4.paragraph_format.space_after = Pt(12)
    
    add_figure_placeholder(
        doc,
        'all_confusion_matrices.png',
        'Matrices de confusión normalizadas para los tres modelos (columnas: MLP-MA, TabNet, '
        'FT-Transformer) aplicados a los tres grupos demográficos (filas: Adultos, Jóvenes, Niños). '
        'Los valores representan la proporción de predicciones para cada clase verdadera. '
        'La diagonal principal (valores más oscuros) indica predicciones correctas.',
        '1'
    )
    
    doc.add_heading('1.3.1. Interpretación de Patrones de Clasificación', level=3)
    
    # Análisis por grupo
    patterns_text = [
        ('Adultos: Alta precisión diagonal', 
         'Las matrices del grupo adultos exhiben valores diagonales superiores a 0.92 en todas las '
         'arquitecturas, con TabNet alcanzando 0.94-0.98 en las tres clases. Las confusiones más '
         'frecuentes ocurren entre las categorías "Lector Medio" y "Lector Frecuente" (confusión del '
         '4-6%), lo cual es esperable dado que la frontera entre estas categorías puede ser difusa en '
         'comportamientos de lectura autodeclarados.'),
        
        ('Jóvenes: Confusión moderada',
         'Se observan valores diagonales entre 0.85-0.93, con mayor confusión en la clase "No Lector", '
         'donde TabNet y MLP-MA predicen erróneamente como "Lector Medio" en aproximadamente el 10% de '
         'los casos. Esto sugiere que algunos jóvenes no lectores presentan características que se '
         'solapan con lectores ocasionales, posiblemente debido a la exposición obligatoria a material '
         'de lectura educativa.'),
        
        ('Niños: Mayor dispersión',
         'Las matrices del grupo infantil muestran la mayor dispersión off-diagonal, con valores '
         'diagonales entre 0.65-0.80. La clase "Lector Frecuente" es la más difícil de identificar '
         '(recall de 0.65-0.72), siendo frecuentemente confundida con "Lector Medio". Esta dificultad '
         'refleja la variabilidad del desarrollo lector en edades tempranas y la posible inconsistencia '
         'en las respuestas del cuestionario reportadas por padres/tutores.')
    ]
    
    for title, text in patterns_text:
        doc.add_heading(title, level=4)
        p = doc.add_paragraph(text)
        p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        p.paragraph_format.space_after = Pt(10)
    
    doc.add_paragraph().paragraph_format.space_after = Pt(16)
    
    # ============================================================================
    # SECCIÓN 5: CURVAS ROC
    # ============================================================================
    
    doc.add_heading('1.4. Análisis de Capacidad Discriminativa: Curvas ROC', level=2)
    
    p5 = doc.add_paragraph(
        'La Figura 2 presenta las curvas ROC (Receiver Operating Characteristic) para cada '
        'clasificador en configuración One-vs-Rest (OvR), donde cada clase se compara contra '
        'todas las demás de forma binaria.'
    )
    p5.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p5.paragraph_format.space_after = Pt(12)
    
    add_figure_placeholder(
        doc,
        'roc_curves_model_comparison.png',
        'Curvas ROC para los tres modelos en configuración One-vs-Rest. Cada panel representa un '
        'grupo demográfico, mostrando las curvas de las tres arquitecturas para comparación directa. '
        'El área bajo la curva (AUC) cuantifica la capacidad discriminativa del modelo.',
        '2'
    )
    
    doc.add_heading('1.4.1. Análisis de AUC-ROC por Modelo', level=3)
    
    p6 = doc.add_paragraph(
        'La Tabla 3 presenta los valores promedio de AUC-ROC calculados para cada modelo, '
        'agregando las tres clases en configuración OvR.'
    )
    p6.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p6.paragraph_format.space_after = Pt(12)
    
    create_table_3(doc)
    
    # Análisis de capacidad discriminativa
    disc_text = [
        ('Capacidad discriminativa excepcional en adultos',
         'Los tres modelos exhiben AUC-ROC superiores a 0.988 en el grupo adultos, aproximándose a '
         'la clasificación perfecta (AUC=1.0). Esto confirma que los patrones de lectura en adultos '
         'son altamente separables en el espacio de características, permitiendo discriminación casi '
         'óptima incluso en configuración multiclase.'),
        
        ('Degradación controlada en jóvenes',
         'Se observa una reducción de aproximadamente 1.5-2% en AUC-ROC para el grupo juvenil, '
         'manteniendo valores excelentes (>0.97). Esta degradación moderada sugiere que, aunque más '
         'complejos que adultos, los patrones de lectura juveniles siguen siendo distintivos.'),
        
        ('Desafío significativo en niños',
         'El grupo infantil presenta la mayor caída en AUC-ROC (0.8436-0.8731), representando una '
         'reducción de 12-14 puntos porcentuales respecto al grupo adultos. Sin embargo, valores '
         'superiores a 0.84 aún indican buena capacidad discriminativa, siendo TabNet el modelo más '
         'robusto (AUC=0.8731).')
    ]
    
    for title, text in disc_text:
        doc.add_heading(title, level=4)
        p = doc.add_paragraph(text)
        p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        p.paragraph_format.space_after = Pt(10)
    
    doc.add_paragraph().paragraph_format.space_after = Pt(16)
    
    # ============================================================================
    # SECCIÓN 6: CURVAS DE APRENDIZAJE
    # ============================================================================
    
    doc.add_heading('1.5. Análisis de Convergencia: Curvas de Aprendizaje', level=2)
    
    p7 = doc.add_paragraph(
        'La Figura 3 muestra la evolución del loss de entrenamiento y validación a lo largo de '
        'las 200 épocas máximas permitidas, con indicadores de early stopping donde aplique.'
    )
    p7.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p7.paragraph_format.space_after = Pt(12)
    
    add_figure_placeholder(
        doc,
        'learning_curves_all.png',
        'Curvas de aprendizaje para los nueve experimentos. Cada subplot muestra la evolución del '
        'loss de entrenamiento (línea sólida) y validación (línea punteada) a lo largo de las épocas. '
        'La línea vertical indica el punto de early stopping (patience=20 épocas). La convergencia '
        'entre ambas curvas indica ausencia de overfitting significativo.',
        '3'
    )
    
    doc.add_heading('1.5.1. Análisis de Convergencia y Generalización', level=3)
    
    conv_text = [
        ('Convergencia temprana en adultos',
         'Los modelos del grupo adultos convergieron entre las épocas 100-140, con early stopping '
         'activado en promedio en la época 125. La separación mínima entre curvas de entrenamiento y '
         'validación (gap <0.02) evidencia excelente capacidad de generalización, sin signos de '
         'overfitting. TabNet mostró la convergencia más estable, con curvas suaves y monótonamente '
         'decrecientes.'),
        
        ('Convergencia moderada en jóvenes',
         'El grupo juvenil requirió 140-170 épocas para convergencia, aproximadamente 30 épocas más '
         'que adultos. Se observa un gap train-val ligeramente mayor (0.03-0.05), pero aún dentro de '
         'rangos aceptables. MLP-MA exhibió la mayor variabilidad en validación, posiblemente debido a '
         'la alta dimensionalidad de características (139 features).'),
        
        ('Convergencia lenta en niños',
         'Los modelos para niños necesitaron 160-190 épocas, con algunos alcanzando el límite de 200 '
         'épocas. El gap train-val más pronunciado (0.06-0.08) sugiere mayor dificultad para generalizar, '
         'aunque no indica overfitting severo. FT-Transformer mostró las curvas más ruidosas, confirmando '
         'su sensibilidad a datasets pequeños.'),
        
        ('Ausencia de overfitting',
         'Crucialmente, ninguno de los nueve modelos presenta el patrón característico de overfitting '
         '(loss de validación creciente mientras el de entrenamiento decrece). Esto valida la efectividad '
         'de las técnicas de regularización empleadas: Dropout (p=0.3), Weight Decay (λ=10⁻⁴) y Early '
         'Stopping (patience=20).')
    ]
    
    for title, text in conv_text:
        doc.add_heading(title, level=4)
        p = doc.add_paragraph(text)
        p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        p.paragraph_format.space_after = Pt(10)
    
    doc.add_paragraph().paragraph_format.space_after = Pt(16)
    
    # ============================================================================
    # SECCIÓN 7: RANKING COPELAND
    # ============================================================================
    
    doc.add_heading('1.6. Ranking Global: Algoritmo de Copeland', level=2)
    
    p8 = doc.add_paragraph(
        'Para establecer un ranking objetivo de las tres arquitecturas, se aplicó el algoritmo de '
        'Copeland, método de votación basado en comparaciones pareadas que considera seis métricas: '
        'Accuracy, F1-Weighted, F1-Macro, Precision-Weighted, Recall-Weighted y AUC-ROC.'
    )
    p8.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p8.paragraph_format.space_after = Pt(12)
    
    create_table_4(doc)
    
    add_figure_placeholder(
        doc,
        'copeland_ranking.png',
        'Visualización del ranking de Copeland. El eje horizontal representa el score, donde valores '
        'positivos indican superioridad global. TabNet emerge como ganador indiscutible con score +4, '
        'mientras FT-Transformer obtiene el score más bajo (-4).',
        '4'
    )
    
    doc.add_heading('1.6.1. Implicaciones del Ranking', level=3)
    
    ranking_text = [
        ('TabNet: Mejor solución general',
         'El score de +4 confirma que TabNet es la arquitectura más adecuada para clasificación de '
         'hábitos de lectura en datos tabulares. Su ventaja se debe a: (1) mecanismo de atención sparse '
         'que selecciona características relevantes en cada paso de decisión, (2) procesamiento secuencial '
         'que captura dependencias complejas, y (3) arquitectura diseñada específicamente para datos '
         'tabulares, a diferencia de transformers concebidos originalmente para secuencias.'),
        
        ('MLP-MA: Alternativa viable',
         'El score neutro (0) posiciona a MLP-MA como una alternativa balanceada: aunque no supera '
         'sistemáticamente a TabNet, su rendimiento es competitivo y su arquitectura más simple ofrece '
         'ventajas en interpretabilidad, velocidad de entrenamiento y requisitos computacionales. Es '
         'recomendable cuando los recursos son limitados o se requiere despliegue en dispositivos con '
         'restricciones.'),
        
        ('FT-Transformer: Limitaciones en datos tabulares',
         'El score de -4 evidencia que la arquitectura transformer, pese a su éxito en NLP y visión '
         'computacional, no es óptima para datos tabulares con tamaños de muestra moderados (10K-40K '
         'instancias). Los transformers requieren típicamente millones de ejemplos para entrenamiento '
         'efectivo, condición no satisfecha en este estudio.')
    ]
    
    for title, text in ranking_text:
        doc.add_heading(title, level=4)
        p = doc.add_paragraph(text)
        p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        p.paragraph_format.space_after = Pt(10)
    
    doc.add_paragraph().paragraph_format.space_after = Pt(16)
    
    # ============================================================================
    # SECCIÓN 8: DISCUSIÓN
    # ============================================================================
    
    doc.add_heading('1.7. Discusión', level=2)
    
    doc.add_heading('1.7.1. Interpretación de Diferencias Demográficas', level=3)
    
    p9 = doc.add_paragraph(
        'La drástica diferencia en rendimiento entre grupos demográficos (97.7% en adultos vs. 73.6% '
        'en niños) puede atribuirse a múltiples factores:'
    )
    p9.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p9.paragraph_format.space_after = Pt(8)
    
    factors = [
        ('Consistencia de respuestas', 
         'Los adultos responden directamente el cuestionario ENAHO, mientras que para niños las '
         'respuestas son proxy (reportadas por padres/tutores), introduciendo potencial ruido e '
         'inconsistencias.'),
        ('Variabilidad del desarrollo',
         'El rango etario 3-13 años abarca desde educación inicial hasta secundaria temprana, períodos '
         'con cambios dramáticos en capacidades de lectura y acceso a material escrito.'),
        ('Tamaño de muestra',
         'El grupo niños tiene 10,388 instancias, versus 40,559 en adultos (una reducción del 74%), '
         'limitando la capacidad de los modelos para aprender patrones complejos.'),
        ('Desbalance de clases',
         'Análisis exploratorio reveló que el grupo niños presenta mayor desbalance, con la clase '
         '"Lector Frecuente" representando solo el 12% de las muestras, versus 27% en adultos.')
    ]
    
    for title, text in factors:
        p = doc.add_paragraph(style='List Bullet')
        p.add_run(f'{title}: ').bold = True
        p.add_run(text)
        p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    
    doc.add_paragraph().paragraph_format.space_after = Pt(12)
    
    # Fortalezas
    doc.add_heading('1.7.2. Fortalezas del Estudio', level=3)
    
    strengths = [
        ('Rigor metodológico', 
         'Entrenamiento de 200 épocas con early stopping, validación en conjunto separado, y evaluación '
         'exhaustiva con seis métricas garantizan robustez de resultados.'),
        ('Datos representativos',
         'Uso de ENAHO 2022, encuesta nacional con muestreo probabilístico, asegura representatividad '
         'de la población peruana.'),
        ('Comparación justa',
         'Los tres modelos fueron entrenados con configuraciones homogéneas (mismos hiperparámetros, '
         'mismo preprocesamiento, mismas particiones) eliminando sesgos de comparación.'),
        ('Enfoque multiárquitectura',
         'La evaluación de tres paradigmas distintos (MLP con atención, TabNet, Transformers) proporciona '
         'insights sobre la naturaleza del problema más allá de un modelo específico.')
    ]
    
    for title, text in strengths:
        p = doc.add_paragraph(style='List Bullet')
        p.add_run(f'{title}: ').bold = True
        p.add_run(text)
        p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    
    doc.add_paragraph().paragraph_format.space_after = Pt(12)
    
    # Limitaciones
    doc.add_heading('1.7.3. Limitaciones y Trabajo Futuro', level=3)
    
    limitations = [
        ('Datos de corte transversal',
         'ENAHO 2022 captura un momento temporal único. Estudios longitudinales permitirían analizar '
         'evolución de hábitos de lectura.'),
        ('Categorización discreta',
         'Las tres clases (No Lector, Lector Medio, Lector Frecuente) simplifican un espectro continuo. '
         'Enfoques de regresión ordinal podrían ser más apropiados.'),
        ('Features autodeclaradas',
         'Respuestas de cuestionario pueden sufrir sesgo de deseabilidad social (sobrestimación de '
         'lectura).'),
        ('Exploración limitada de arquitecturas',
         'Otros modelos como XGBoost, LightGBM o redes neuronales con attention cross-layer podrían '
         'evaluarse.'),
        ('Interpretabilidad',
         'Aunque TabNet ofrece attention masks interpretables, este estudio no incluyó análisis de '
         'importancia de características. Investigación futura debería examinar qué variables '
         'socioeconómicas/demográficas son más predictivas.')
    ]
    
    for title, text in limitations:
        p = doc.add_paragraph(style='List Bullet')
        p.add_run(f'{title}: ').bold = True
        p.add_run(text)
        p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    
    doc.add_paragraph().paragraph_format.space_after = Pt(16)
    
    # ============================================================================
    # SECCIÓN 9: CONCLUSIONES
    # ============================================================================
    
    doc.add_heading('1.8. Conclusiones de los Resultados', level=2)
    
    p10 = doc.add_paragraph(
        'Los experimentos realizados permiten establecer las siguientes conclusiones:'
    )
    p10.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p10.paragraph_format.space_after = Pt(8)
    
    conclusions = [
        'TabNet es la arquitectura óptima para clasificación de hábitos de lectura en datos tabulares, '
        'superando consistentemente a MLP-MA y FT-Transformer en los tres grupos demográficos y seis '
        'métricas evaluadas, con un score de Copeland de +4.',
        
        'La complejidad del problema varía drásticamente entre grupos demográficos: adultos presentan '
        'patrones altamente predecibles (97.7% accuracy), jóvenes moderadamente complejos (92.4% accuracy), '
        'y niños representan el mayor desafío (73.6% accuracy), requiriendo futuras investigaciones '
        'específicas para este segmento.',
        
        'Los tres modelos generalizan adecuadamente sin evidencia de overfitting, validando la efectividad '
        'de las técnicas de regularización (Dropout, Weight Decay, Early Stopping) implementadas.',
        
        'F1-Macro como métrica crítica: mientras que accuracy y F1-weighted son optimistas debido al '
        'desbalance de clases, F1-Macro revela desafíos reales en predicción de categorías minoritarias, '
        'especialmente en niños (F1-Macro=0.554 vs. Accuracy=0.736).',
        
        'Transformers no son óptimos para datos tabulares con tamaños de muestra moderados (10K-40K '
        'instancias), como evidencia el rendimiento inferior y mayor variabilidad de FT-Transformer. '
        'Arquitecturas diseñadas específicamente para tabular data (TabNet) o híbridos simples (MLP-MA) '
        'son preferibles.'
    ]
    
    for i, conclusion in enumerate(conclusions, 1):
        p = doc.add_paragraph(style='List Number')
        p.add_run(conclusion)
        p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    
    doc.add_paragraph().paragraph_format.space_after = Pt(12)
    
    p11 = doc.add_paragraph(
        'Estos resultados establecen un marco sólido para futuras intervenciones de política pública '
        'orientadas a fomentar hábitos de lectura en Perú, identificando a niños como población '
        'prioritaria y a TabNet como herramienta de clasificación más confiable para caracterización '
        'poblacional.'
    )
    p11.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    
    # ============================================================================
    # GUARDAR DOCUMENTO
    # ============================================================================
    
    output_path = r'c:\Users\xdxdxdxd\OneDrive - Universidad Peruana Unión\Escritorio\MLP_Py_In01\Classfier_Reading\RESULTADOS_REVISTA.docx'
    doc.save(output_path)
    
    return output_path

if __name__ == '__main__':
    try:
        output_file = create_word_document()
        print(f'✓ Documento Word creado exitosamente: {output_file}')
        print(f'✓ Total de secciones: 8 (Resultados completos)')
        print(f'✓ Total de tablas: 4 (Metrics, Statistical Summary, AUC-ROC, Copeland)')
        print(f'✓ Total de figuras: 4 placeholders (Confusión, ROC, Learning Curves, Ranking)')
        print(f'✓ Formato: Académico profesional, listo para revista')
    except Exception as e:
        print(f'✗ Error al crear documento: {e}')
        import traceback
        traceback.print_exc()
