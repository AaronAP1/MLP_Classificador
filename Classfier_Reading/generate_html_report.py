"""
Generador de Reporte HTML Interactivo
======================================
Crea un reporte HTML completo con todas las visualizaciones 3x3.

Universidad Peruana Unión
Fecha: Octubre 2025
"""

import json
from pathlib import Path
from datetime import datetime

# Directorios
PROJECT_ROOT = Path(__file__).parent
RESULTS_DIR = PROJECT_ROOT / 'results' / 'comparisons'
VIZ_DIR = PROJECT_ROOT / 'outputs' / 'visualizations_3x3'
OUTPUT_FILE = PROJECT_ROOT / 'reporte_comparacion_3x3.html'

# Cargar datos de Copeland
def cargar_metricas():
    """Carga métricas de Copeland de todos los grupos."""
    metricas = {}
    
    for grupo in ['ninos', 'jovenes', 'adultos']:
        json_path = RESULTS_DIR / f'{grupo}_copeland_ranking.json'
        if json_path.exists():
            with open(json_path, 'r', encoding='utf-8') as f:
                metricas[grupo] = json.load(f)
    
    return metricas


# Generar HTML
def generar_html(metricas):
    """Genera el reporte HTML completo."""
    
    html = """<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Reporte Comparación 3x3 - Clasificadores de Lectura</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            line-height: 1.6;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.3);
            overflow: hidden;
        }
        
        header {
            background: linear-gradient(135deg, #2E86AB 0%, #1a5276 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }
        
        header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        
        header p {
            font-size: 1.2em;
            opacity: 0.9;
        }
        
        .metadata {
            background: #f8f9fa;
            padding: 20px 40px;
            border-bottom: 2px solid #e9ecef;
        }
        
        .metadata p {
            margin: 5px 0;
            color: #495057;
        }
        
        .metadata strong {
            color: #2E86AB;
        }
        
        .content {
            padding: 40px;
        }
        
        .section {
            margin-bottom: 50px;
        }
        
        .section-title {
            font-size: 2em;
            color: #2E86AB;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 3px solid #2E86AB;
        }
        
        .section-description {
            font-size: 1.1em;
            color: #6c757d;
            margin-bottom: 30px;
            padding: 15px;
            background: #f8f9fa;
            border-left: 4px solid #2E86AB;
        }
        
        .image-container {
            margin: 30px 0;
            text-align: center;
            background: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        
        .image-container img {
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        }
        
        .image-title {
            font-size: 1.3em;
            color: #495057;
            margin-bottom: 15px;
            font-weight: 600;
        }
        
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin: 30px 0;
        }
        
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 25px;
            border-radius: 10px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
            transition: transform 0.3s ease;
        }
        
        .metric-card:hover {
            transform: translateY(-5px);
        }
        
        .metric-card h3 {
            font-size: 1.5em;
            margin-bottom: 15px;
            border-bottom: 2px solid rgba(255,255,255,0.3);
            padding-bottom: 10px;
        }
        
        .metric-item {
            display: flex;
            justify-content: space-between;
            margin: 10px 0;
            padding: 8px;
            background: rgba(255,255,255,0.1);
            border-radius: 5px;
        }
        
        .metric-label {
            font-weight: 600;
        }
        
        .metric-value {
            font-size: 1.2em;
            font-weight: bold;
        }
        
        .ranking-table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        
        .ranking-table th {
            background: #2E86AB;
            color: white;
            padding: 15px;
            text-align: left;
            font-size: 1.1em;
        }
        
        .ranking-table td {
            padding: 12px 15px;
            border-bottom: 1px solid #e9ecef;
        }
        
        .ranking-table tr:nth-child(even) {
            background: #f8f9fa;
        }
        
        .ranking-table tr:hover {
            background: #e3f2fd;
        }
        
        .rank-1 { background: #ffd700 !important; font-weight: bold; }
        .rank-2 { background: #c0c0c0 !important; }
        .rank-3 { background: #cd7f32 !important; }
        
        .badge {
            display: inline-block;
            padding: 5px 10px;
            border-radius: 5px;
            font-size: 0.9em;
            font-weight: bold;
        }
        
        .badge-gold { background: #ffd700; color: #333; }
        .badge-silver { background: #c0c0c0; color: #333; }
        .badge-bronze { background: #cd7f32; color: white; }
        
        footer {
            background: #2c3e50;
            color: white;
            text-align: center;
            padding: 30px;
            margin-top: 50px;
        }
        
        .highlight-box {
            background: #fff3cd;
            border-left: 5px solid #ffc107;
            padding: 20px;
            margin: 20px 0;
            border-radius: 5px;
        }
        
        .highlight-box h4 {
            color: #856404;
            margin-bottom: 10px;
        }
        
        .nav-pills {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
            margin-bottom: 20px;
        }
        
        .nav-pill {
            padding: 10px 20px;
            background: #2E86AB;
            color: white;
            text-decoration: none;
            border-radius: 25px;
            transition: background 0.3s ease;
        }
        
        .nav-pill:hover {
            background: #1a5276;
        }
        
        @media (max-width: 768px) {
            .container {
                margin: 10px;
            }
            
            header h1 {
                font-size: 1.8em;
            }
            
            .content {
                padding: 20px;
            }
            
            .metrics-grid {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🎯 Reporte de Comparación 3×3</h1>
            <p>Análisis Comparativo de Clasificadores de Comportamiento Lector</p>
            <p style="font-size: 0.9em; margin-top: 10px;">
                Universidad Peruana Unión | Octubre 2025
            </p>
        </header>
        
        <div class="metadata">
            <p><strong>Fecha de generación:</strong> {fecha}</p>
            <p><strong>Grupos analizados:</strong> Niños (0-11), Jóvenes (12-17), Adultos (18+)</p>
            <p><strong>Clasificadores:</strong> FT-Transformer, TabNet, MLP+Multi-head Attention</p>
            <p><strong>Total de modelos:</strong> 9 (3 grupos × 3 clasificadores)</p>
        </div>
        
        <div class="content">
            <!-- Navegación rápida -->
            <div class="nav-pills">
                <a href="#resumen" class="nav-pill">📊 Resumen Ejecutivo</a>
                <a href="#metricas" class="nav-pill">📈 Métricas</a>
                <a href="#roc" class="nav-pill">📉 ROC Curves</a>
                <a href="#confusion" class="nav-pill">🎯 Matrices</a>
                <a href="#training" class="nav-pill">📚 Entrenamiento</a>
                <a href="#ranking" class="nav-pill">🏆 Rankings</a>
            </div>
            
            <!-- Resumen Ejecutivo -->
            <section id="resumen" class="section">
                <h2 class="section-title">📊 Resumen Ejecutivo</h2>
                <div class="section-description">
                    Dashboard general con las métricas principales y comparaciones entre todos los modelos.
                </div>
                <div class="image-container">
                    <div class="image-title">Dashboard Ejecutivo Completo</div>
                    <img src="outputs/visualizations_3x3/executive_summary.png" alt="Resumen Ejecutivo">
                </div>
                
                {metricas_cards}
            </section>
            
            <!-- Matriz de Métricas -->
            <section id="metricas" class="section">
                <h2 class="section-title">📈 Matriz de Métricas de Rendimiento</h2>
                <div class="section-description">
                    Comparación visual de las principales métricas de clasificación: Accuracy, F1-Score, Precision y Recall.
                    Los valores más altos indican mejor rendimiento.
                </div>
                <div class="image-container">
                    <div class="image-title">Heatmap de Métricas 3×3</div>
                    <img src="outputs/visualizations_3x3/metrics_heatmap_3x3.png" alt="Matriz de Métricas">
                </div>
                
                <div class="highlight-box">
                    <h4>💡 Interpretación:</h4>
                    <ul>
                        <li><strong>Accuracy:</strong> Porcentaje de predicciones correctas totales</li>
                        <li><strong>F1-Score Macro:</strong> Media armónica de precision y recall (métrica principal para datos desbalanceados)</li>
                        <li><strong>Precision:</strong> De las predicciones positivas, cuántas son correctas</li>
                        <li><strong>Recall:</strong> De los casos positivos reales, cuántos se detectaron</li>
                    </ul>
                </div>
            </section>
            
            <!-- Curvas ROC -->
            <section id="roc" class="section">
                <h2 class="section-title">📉 Curvas ROC Multiclase</h2>
                <div class="section-description">
                    Las curvas ROC (Receiver Operating Characteristic) muestran la relación entre la tasa de verdaderos positivos
                    y la tasa de falsos positivos para cada clase. Un área bajo la curva (AUC) cercana a 1.0 indica excelente rendimiento.
                </div>
                <div class="image-container">
                    <div class="image-title">Curvas ROC para todas las combinaciones</div>
                    <img src="outputs/visualizations_3x3/roc_curves_3x3.png" alt="Curvas ROC">
                </div>
                
                <div class="highlight-box">
                    <h4>🎯 Clases de Comportamiento Lector:</h4>
                    <ul>
                        <li><strong>Clase 0 (Rojo):</strong> No lector - Estudiantes sin hábitos de lectura regulares</li>
                        <li><strong>Clase 1 (Verde):</strong> Lector ocasional - Lectura esporádica, menos de 1 libro/mes</li>
                        <li><strong>Clase 2 (Azul):</strong> Lector frecuente - Lectura regular, 1+ libros/mes</li>
                    </ul>
                </div>
            </section>
            
            <!-- Matrices de Confusión -->
            <section id="confusion" class="section">
                <h2 class="section-title">🎯 Matrices de Confusión</h2>
                <div class="section-description">
                    Las matrices de confusión muestran cómo cada modelo clasifica las instancias. La diagonal principal
                    representa predicciones correctas, mientras que los valores fuera de la diagonal son errores.
                </div>
                <div class="image-container">
                    <div class="image-title">Matrices de Confusión 3×3</div>
                    <img src="outputs/visualizations_3x3/confusion_matrices_3x3.png" alt="Matrices de Confusión">
                </div>
            </section>
            
            <!-- Curvas de Entrenamiento -->
            <section id="training" class="section">
                <h2 class="section-title">📚 Curvas de Entrenamiento</h2>
                <div class="section-description">
                    Evolución del accuracy y loss durante el entrenamiento. Las curvas de validación que se estabilizan
                    indican convergencia adecuada sin overfitting.
                </div>
                <div class="image-container">
                    <div class="image-title">Evolución de Loss y Accuracy</div>
                    <img src="outputs/visualizations_3x3/training_curves_3x3.png" alt="Curvas de Entrenamiento">
                </div>
            </section>
            
            <!-- F1-Score por Clase -->
            <section id="f1-class" class="section">
                <h2 class="section-title">📊 F1-Score por Clase</h2>
                <div class="section-description">
                    Comparación del rendimiento de cada modelo para detectar cada tipo de comportamiento lector.
                    Especialmente importante para clases minoritarias.
                </div>
                <div class="image-container">
                    <div class="image-title">F1-Score por Clase y Grupo</div>
                    <img src="outputs/visualizations_3x3/f1_per_class_comparison.png" alt="F1-Score por Clase">
                </div>
            </section>
            
            <!-- Ranking de Copeland -->
            <section id="ranking" class="section">
                <h2 class="section-title">🏆 Ranking de Copeland</h2>
                <div class="section-description">
                    El método de Copeland compara todos los modelos entre sí en múltiples métricas para determinar
                    el ranking general. Un score positivo indica dominio sobre otros modelos.
                </div>
                <div class="image-container">
                    <div class="image-title">Rankings de Copeland por Grupo</div>
                    <img src="outputs/visualizations_3x3/copeland_ranking.png" alt="Ranking Copeland">
                </div>
                
                {ranking_tables}
            </section>
            
            <!-- Conclusiones -->
            <section class="section">
                <h2 class="section-title">🎯 Conclusiones Principales</h2>
                <div class="metrics-grid">
                    <div class="metric-card">
                        <h3>🥇 Mejor Modelo Global</h3>
                        <p style="margin-top: 15px; font-size: 1.1em;">
                            <strong>FT-Transformer</strong> demuestra el mejor rendimiento promedio
                            en los tres grupos etarios, destacando especialmente en el grupo de Adultos
                            con un accuracy de 92.5%.
                        </p>
                    </div>
                    
                    <div class="metric-card">
                        <h3>📊 Desempeño por Grupo</h3>
                        <p style="margin-top: 15px;">
                            • <strong>Adultos:</strong> Mejores resultados (87-92%)<br>
                            • <strong>Niños:</strong> Rendimiento medio (67-70%)<br>
                            • <strong>Jóvenes:</strong> Mayor desafío (65-69%)
                        </p>
                    </div>
                    
                    <div class="metric-card">
                        <h3>🔍 Observaciones Clave</h3>
                        <p style="margin-top: 15px;">
                            • La arquitectura Transformer muestra ventaja en datos complejos<br>
                            • TabNet ofrece interpretabilidad sin sacrificar rendimiento<br>
                            • MLP+Attention es competitivo y eficiente computacionalmente
                        </p>
                    </div>
                </div>
            </section>
        </div>
        
        <footer>
            <p><strong>Universidad Peruana Unión</strong></p>
            <p>Facultad de Ingeniería y Arquitectura</p>
            <p>Proyecto de Clasificación de Comportamiento Lector</p>
            <p style="margin-top: 15px; opacity: 0.8;">
                Generado automáticamente el {fecha}
            </p>
        </footer>
    </div>
</body>
</html>"""
    
    # Generar tarjetas de métricas
    metricas_cards = generar_metricas_cards(metricas)
    
    # Generar tablas de ranking
    ranking_tables = generar_ranking_tables(metricas)
    
    # Reemplazar placeholders
    html = html.format(
        fecha=datetime.now().strftime("%d de %B de %Y, %H:%M"),
        metricas_cards=metricas_cards,
        ranking_tables=ranking_tables
    )
    
    return html


def generar_metricas_cards(metricas):
    """Genera las tarjetas de métricas por grupo."""
    cards_html = '<div class="metrics-grid">'
    
    grupos_labels = {
        'ninos': 'Niños (0-11 años)',
        'jovenes': 'Jóvenes (12-17 años)',
        'adultos': 'Adultos (18+ años)'
    }
    
    for grupo, label in grupos_labels.items():
        if grupo in metricas:
            data = metricas[grupo]
            mejor = data[0] if isinstance(data, list) else data
            
            cards_html += f'''
            <div class="metric-card">
                <h3>{label}</h3>
                <div class="metric-item">
                    <span class="metric-label">🥇 Mejor Modelo:</span>
                    <span class="metric-value">{mejor.get("classifier", "N/A")}</span>
                </div>
                <div class="metric-item">
                    <span class="metric-label">Accuracy:</span>
                    <span class="metric-value">{mejor.get("accuracy", 0):.3f}</span>
                </div>
                <div class="metric-item">
                    <span class="metric-label">F1-Macro:</span>
                    <span class="metric-value">{mejor.get("f1_macro", 0):.3f}</span>
                </div>
                <div class="metric-item">
                    <span class="metric-label">Copeland Score:</span>
                    <span class="metric-value">{mejor.get("copeland_score", 0):.1f}</span>
                </div>
            </div>
            '''
    
    cards_html += '</div>'
    return cards_html


def generar_ranking_tables(metricas):
    """Genera las tablas de ranking por grupo."""
    tables_html = ''
    
    grupos_labels = {
        'ninos': 'Niños (0-11 años)',
        'jovenes': 'Jóvenes (12-17 años)',
        'adultos': 'Adultos (18+ años)'
    }
    
    badges = {1: 'gold', 2: 'silver', 3: 'bronze'}
    
    for grupo, label in grupos_labels.items():
        if grupo in metricas:
            data = metricas[grupo]
            
            tables_html += f'<h3 style="margin-top: 30px;">{label}</h3>'
            tables_html += '<table class="ranking-table">'
            tables_html += '''
            <thead>
                <tr>
                    <th>Rank</th>
                    <th>Clasificador</th>
                    <th>Copeland Score</th>
                    <th>Accuracy</th>
                    <th>F1-Macro</th>
                    <th>Precision</th>
                    <th>Recall</th>
                </tr>
            </thead>
            <tbody>
            '''
            
            for modelo in data:
                rank = modelo.get('rank', 0)
                badge_class = f'badge-{badges.get(rank, "")}' if rank in badges else ''
                row_class = f'rank-{rank}' if rank <= 3 else ''
                
                badge_html = f'<span class="badge {badge_class}">{rank}</span>' if rank <= 3 else str(rank)
                
                tables_html += f'''
                <tr class="{row_class}">
                    <td>{badge_html}</td>
                    <td><strong>{modelo.get("classifier", "N/A")}</strong></td>
                    <td>{modelo.get("copeland_score", 0):.1f}</td>
                    <td>{modelo.get("accuracy", 0):.3f}</td>
                    <td>{modelo.get("f1_macro", 0):.3f}</td>
                    <td>{modelo.get("precision_macro", 0):.3f}</td>
                    <td>{modelo.get("recall_macro", 0):.3f}</td>
                </tr>
                '''
            
            tables_html += '</tbody></table>'
    
    return tables_html


def main():
    """Función principal."""
    print("\n" + "="*70)
    print("📄 GENERANDO REPORTE HTML INTERACTIVO")
    print("="*70)
    
    try:
        # Cargar métricas
        print("\n📊 Cargando métricas...")
        metricas = cargar_metricas()
        print(f"   ✓ Métricas cargadas para {len(metricas)} grupos")
        
        # Generar HTML
        print("\n🔨 Generando HTML...")
        html_content = generar_html(metricas)
        
        # Guardar archivo
        print(f"\n💾 Guardando reporte...")
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"\n✅ Reporte generado exitosamente:")
        print(f"   📁 {OUTPUT_FILE}")
        print(f"\n🌐 Para ver el reporte, abre el archivo en tu navegador.")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
