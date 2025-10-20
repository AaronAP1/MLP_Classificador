# -*- coding: utf-8 -*-
"""
Análisis Comparativo Completo - Los 3 Grupos Etarios

Este script analiza en profundidad:
1. Características de cada dataset (niños, jóvenes, adultos)
2. Complejidad del problema de clasificación
3. Comportamiento de modelos simples
4. Importancia de features
5. Comparación entre grupos
6. Recomendaciones específicas por grupo

Universidad Peruana Unión
Fecha: Octubre 2025
"""

import sys
import os

# Configurar UTF-8 para Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, 
    f1_score, 
    classification_report,
    confusion_matrix
)
import json
from datetime import datetime

# Agregar path del proyecto
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configuración
OUTPUT_DIR = project_root / 'experiments' / 'comparative_analysis'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


# ===================================================================
# CARGA DE DATOS
# ===================================================================

def load_group_data(group_name):
    """Cargar datos de un grupo específico."""
    data_dir = project_root / 'data' / 'processed' / group_name
    
    train_df = pd.read_csv(data_dir / f'{group_name}_train.csv')
    val_df = pd.read_csv(data_dir / f'{group_name}_val.csv')
    test_df = pd.read_csv(data_dir / f'{group_name}_test.csv')
    
    return train_df, val_df, test_df


def load_all_groups():
    """Cargar datos de los 3 grupos."""
    print("="*70)
    print("📂 CARGANDO DATOS DE LOS 3 GRUPOS")
    print("="*70)
    
    groups = {}
    
    for group_name in ['ninos', 'jovenes', 'adultos']:
        print(f"\n📁 Cargando {group_name.upper()}...")
        train_df, val_df, test_df = load_group_data(group_name)
        
        groups[group_name] = {
            'train': train_df,
            'val': val_df,
            'test': test_df,
            'total': len(train_df) + len(val_df) + len(test_df),
            'n_features': len(train_df.columns) - 1
        }
        
        print(f"   Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")
        print(f"   Total: {groups[group_name]['total']:,} | Features: {groups[group_name]['n_features']}")
    
    return groups


# ===================================================================
# ANÁLISIS 1: CARACTERÍSTICAS BÁSICAS
# ===================================================================

def analyze_basic_characteristics(groups):
    """Analizar características básicas de cada grupo."""
    print("\n" + "="*70)
    print("📊 ANÁLISIS 1: CARACTERÍSTICAS BÁSICAS DE CADA GRUPO")
    print("="*70)
    
    results = []
    
    for group_name, data in groups.items():
        train_df = data['train']
        
        # Distribución de clases
        class_dist = train_df['target'].value_counts().sort_index()
        class_pcts = 100 * class_dist / len(train_df)
        
        # Calcular balance
        balance_metric = class_pcts.max() - class_pcts.min()
        
        # Gini impurity (0 = perfectamente balanceado, 1 = totalmente desbalanceado)
        probs = class_pcts / 100
        gini = 1 - np.sum(probs ** 2)
        
        result = {
            'grupo': group_name,
            'n_muestras': len(train_df),
            'n_features': len(train_df.columns) - 1,
            'n_clases': len(class_dist),
            'clase_0_pct': class_pcts.get(0, 0),
            'clase_1_pct': class_pcts.get(1, 0),
            'clase_2_pct': class_pcts.get(2, 0),
            'balance_metric': balance_metric,
            'gini_impurity': gini
        }
        
        results.append(result)
        
        print(f"\n{'─'*70}")
        print(f"🎯 {group_name.upper()}")
        print(f"{'─'*70}")
        print(f"   Muestras:  {result['n_muestras']:,}")
        print(f"   Features:  {result['n_features']}")
        print(f"   Clases:    {result['n_clases']}")
        print(f"\n   Distribución:")
        for cls, pct in class_pcts.items():
            if pct < 1:
                emoji = "🔴"
            elif pct < 30:
                emoji = "🟡"
            else:
                emoji = "🟢"
            print(f"      {emoji} Clase {cls}: {pct:6.2f}%")
        
        print(f"\n   Balance Metric: {balance_metric:.2f}%")
        
        if balance_metric < 20:
            print(f"      ✅ BALANCEADO")
        elif balance_metric < 50:
            print(f"      ⚠️  MODERADAMENTE DESBALANCEADO")
        else:
            print(f"      🔴 EXTREMADAMENTE DESBALANCEADO")
        
        print(f"   Gini Impurity:  {gini:.4f}")
    
    # Crear DataFrame comparativo
    df_comparison = pd.DataFrame(results)
    df_comparison.to_csv(OUTPUT_DIR / 'basic_characteristics.csv', index=False)
    
    # Visualización
    plot_class_distributions(groups)
    
    return df_comparison


def plot_class_distributions(groups):
    """Graficar distribuciones de clases."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Distribución de Clases por Grupo Etario', fontsize=16, fontweight='bold')
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    for ax, (group_name, data) in zip(axes, groups.items()):
        train_df = data['train']
        class_counts = train_df['target'].value_counts().sort_index()
        class_pcts = 100 * class_counts / len(train_df)
        
        bars = ax.bar(class_counts.index, class_pcts, color=colors, alpha=0.7, edgecolor='black')
        
        # Añadir valores
        for bar, pct, count in zip(bars, class_pcts, class_counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{pct:.1f}%\n({count:,})',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.set_title(f'{group_name.upper()}\n({len(train_df):,} muestras)', 
                    fontsize=12, fontweight='bold')
        ax.set_xlabel('Clase', fontsize=11)
        ax.set_ylabel('Porcentaje (%)', fontsize=11)
        ax.set_ylim(0, max(class_pcts) * 1.2)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'class_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n📊 Gráfico guardado: class_distributions.png")


# ===================================================================
# ANÁLISIS 2: COMPLEJIDAD DEL PROBLEMA
# ===================================================================

def analyze_problem_complexity(groups):
    """Analizar complejidad del problema de clasificación."""
    print("\n" + "="*70)
    print("🔬 ANÁLISIS 2: COMPLEJIDAD DEL PROBLEMA DE CLASIFICACIÓN")
    print("="*70)
    
    results = []
    
    for group_name, data in groups.items():
        print(f"\n{'─'*70}")
        print(f"🎯 {group_name.upper()}")
        print(f"{'─'*70}")
        
        train_df = data['train']
        test_df = data['test']
        
        X_train = train_df.drop(columns=['target']).fillna(0).values
        y_train = train_df['target'].values
        X_test = test_df.drop(columns=['target']).fillna(0).values
        y_test = test_df['target'].values
        
        # Probar 4 modelos de complejidad creciente
        models = {
            'DT(depth=1)': DecisionTreeClassifier(max_depth=1, random_state=42),
            'DT(depth=3)': DecisionTreeClassifier(max_depth=3, random_state=42),
            'LogReg': LogisticRegression(max_iter=1000, random_state=42),
            'DT(depth=10)': DecisionTreeClassifier(max_depth=10, random_state=42),
        }
        
        group_results = {'grupo': group_name}
        
        print("\n   Entrenando modelos baseline...")
        for model_name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='macro')
            
            group_results[f'{model_name}_acc'] = acc
            group_results[f'{model_name}_f1'] = f1
            
            # Emoji según accuracy
            if acc >= 0.95:
                emoji = "🔴"
                status = "TRIVIAL"
            elif acc >= 0.85:
                emoji = "🟡"
                status = "FÁCIL"
            elif acc >= 0.70:
                emoji = "🟢"
                status = "MODERADO"
            else:
                emoji = "🔵"
                status = "DIFÍCIL"
            
            print(f"      {emoji} {model_name:15s} | Acc: {acc:.4f} | F1: {f1:.4f} | {status}")
        
        results.append(group_results)
        
        # Análisis de complejidad
        best_acc = max([group_results[k] for k in group_results.keys() if k.endswith('_acc')])
        
        print(f"\n   📊 Complejidad del problema:")
        if best_acc >= 0.95:
            print(f"      🔴 PROBLEMA TRIVIAL (mejor acc: {best_acc:.1%})")
            print(f"         → Incluso modelos simples resuelven el problema")
            print(f"         → Deep learning puede estar sobreajustando")
        elif best_acc >= 0.85:
            print(f"      🟡 PROBLEMA FÁCIL (mejor acc: {best_acc:.1%})")
            print(f"         → Modelos clásicos funcionan bien")
            print(f"         → Deep learning tiene ventaja moderada")
        elif best_acc >= 0.70:
            print(f"      🟢 PROBLEMA MODERADO (mejor acc: {best_acc:.1%})")
            print(f"         → Se beneficia de modelos más sofisticados")
            print(f"         → Deep learning justificado")
        else:
            print(f"      🔵 PROBLEMA DIFÍCIL (mejor acc: {best_acc:.1%})")
            print(f"         → Requiere modelos complejos")
            print(f"         → Deep learning altamente recomendado")
    
    # Guardar resultados
    df_complexity = pd.DataFrame(results)
    df_complexity.to_csv(OUTPUT_DIR / 'problem_complexity.csv', index=False)
    
    # Visualización
    plot_complexity_comparison(df_complexity)
    
    return df_complexity


def plot_complexity_comparison(df):
    """Graficar comparación de complejidad."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Complejidad del Problema por Grupo Etario', fontsize=16, fontweight='bold')
    
    # Preparar datos
    models = ['DT(depth=1)', 'DT(depth=3)', 'LogReg', 'DT(depth=10)']
    
    # Plot 1: Accuracy
    ax = axes[0]
    x = np.arange(len(df))
    width = 0.2
    
    for i, model in enumerate(models):
        col = f'{model}_acc'
        if col in df.columns:
            values = df[col].values
            ax.bar(x + i*width, values, width, label=model, alpha=0.8)
    
    ax.set_xlabel('Grupo Etario', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Accuracy por Modelo', fontsize=13, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([g.upper() for g in df['grupo']])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.1)
    
    # Plot 2: F1-Score
    ax = axes[1]
    for i, model in enumerate(models):
        col = f'{model}_f1'
        if col in df.columns:
            values = df[col].values
            ax.bar(x + i*width, values, width, label=model, alpha=0.8)
    
    ax.set_xlabel('Grupo Etario', fontsize=12)
    ax.set_ylabel('F1-Score Macro', fontsize=12)
    ax.set_title('F1-Score por Modelo', fontsize=13, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([g.upper() for g in df['grupo']])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.1)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'complexity_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n📊 Gráfico guardado: complexity_comparison.png")


# ===================================================================
# ANÁLISIS 3: IMPORTANCIA DE FEATURES
# ===================================================================

def analyze_feature_importance(groups):
    """Analizar importancia de features en cada grupo."""
    print("\n" + "="*70)
    print("🌳 ANÁLISIS 3: IMPORTANCIA DE FEATURES")
    print("="*70)
    
    results = {}
    
    for group_name, data in groups.items():
        print(f"\n{'─'*70}")
        print(f"🎯 {group_name.upper()}")
        print(f"{'─'*70}")
        
        train_df = data['train']
        X_train = train_df.drop(columns=['target']).fillna(0).values
        y_train = train_df['target'].values
        
        # Entrenar Random Forest para importancias
        print("   Entrenando Random Forest...")
        rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        rf.fit(X_train, y_train)
        
        importances = rf.feature_importances_
        
        # Top features
        indices = np.argsort(importances)[::-1]
        
        print(f"\n   📊 Top 10 features más importantes:")
        cumsum = 0
        top_10_features = []
        
        for i in range(min(10, len(indices))):
            idx = indices[i]
            imp = importances[idx]
            cumsum += imp
            
            if imp > 0.2:
                emoji = "🔴"
            elif imp > 0.1:
                emoji = "🟡"
            elif imp > 0.05:
                emoji = "🟢"
            else:
                emoji = "⚪"
            
            print(f"      {i+1:2d}. {emoji} Feature {idx:3d} | {imp:.4f} ({cumsum:.1%} acum.)")
            top_10_features.append({'rank': i+1, 'feature_idx': int(idx), 'importance': float(imp)})
        
        # Concentración de importancia
        top_3_sum = sum(importances[indices[:3]])
        top_5_sum = sum(importances[indices[:5]])
        top_10_sum = sum(importances[indices[:10]])
        
        print(f"\n   📈 Concentración de importancia:")
        print(f"      Top 3 features:  {top_3_sum:.1%}")
        print(f"      Top 5 features:  {top_5_sum:.1%}")
        print(f"      Top 10 features: {top_10_sum:.1%}")
        
        if top_3_sum > 0.8:
            print(f"      🔴 ALTO: Problema dominado por pocas features")
        elif top_3_sum > 0.5:
            print(f"      🟡 MODERADO: Varias features importantes")
        else:
            print(f"      🟢 DISTRIBUIDO: Muchas features contribuyen")
        
        results[group_name] = {
            'top_10': top_10_features,
            'top_3_sum': float(top_3_sum),
            'top_5_sum': float(top_5_sum),
            'top_10_sum': float(top_10_sum)
        }
    
    # Guardar resultados
    with open(OUTPUT_DIR / 'feature_importance.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    # Visualización
    plot_feature_importance(results)
    
    return results


def plot_feature_importance(results):
    """Graficar importancia de features."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Concentración de Importancia de Features', fontsize=16, fontweight='bold')
    
    for ax, (group_name, data) in zip(axes, results.items()):
        categories = ['Top 3', 'Top 5', 'Top 10', 'Resto']
        values = [
            data['top_3_sum'],
            data['top_5_sum'] - data['top_3_sum'],
            data['top_10_sum'] - data['top_5_sum'],
            1.0 - data['top_10_sum']
        ]
        
        colors = ['#FF6B6B', '#FFA07A', '#FFD700', '#90EE90']
        explode = (0.1, 0.05, 0, 0)
        
        ax.pie(values, labels=categories, autopct='%1.1f%%', startangle=90,
               colors=colors, explode=explode, textprops={'fontsize': 10, 'fontweight': 'bold'})
        ax.set_title(f'{group_name.upper()}', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'feature_concentration.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n📊 Gráfico guardado: feature_concentration.png")


# ===================================================================
# ANÁLISIS 4: COMPARACIÓN GENERAL
# ===================================================================

def generate_comprehensive_report(groups, df_basic, df_complexity, feature_importance):
    """Generar reporte comprensivo."""
    print("\n" + "="*70)
    print("📋 REPORTE COMPREHENSIVO - COMPARACIÓN DE LOS 3 GRUPOS")
    print("="*70)
    
    report = {
        'fecha': datetime.now().isoformat(),
        'grupos': {}
    }
    
    for group_name in ['ninos', 'jovenes', 'adultos']:
        print(f"\n{'═'*70}")
        print(f"🎯 {group_name.upper()}")
        print(f"{'═'*70}")
        
        # Datos básicos
        basic = df_basic[df_basic['grupo'] == group_name].iloc[0]
        complexity = df_complexity[df_complexity['grupo'] == group_name].iloc[0]
        features = feature_importance[group_name]
        
        print(f"\n📊 CARACTERÍSTICAS:")
        print(f"   • Muestras: {basic['n_muestras']:,}")
        print(f"   • Features: {basic['n_features']}")
        print(f"   • Balance:  {basic['balance_metric']:.1f}% diferencia")
        
        print(f"\n🔬 COMPLEJIDAD:")
        dt1_acc = complexity['DT(depth=1)_acc']
        dt10_acc = complexity['DT(depth=10)_acc']
        logreg_acc = complexity['LogReg_acc']
        
        print(f"   • Decision Tree (depth=1):  {dt1_acc:.1%}")
        print(f"   • Logistic Regression:      {logreg_acc:.1%}")
        print(f"   • Decision Tree (depth=10): {dt10_acc:.1%}")
        
        print(f"\n🌳 FEATURES:")
        print(f"   • Top 3 features explican: {features['top_3_sum']:.1%}")
        print(f"   • Top 10 features explican: {features['top_10_sum']:.1%}")
        
        # Clasificación de dificultad
        if dt10_acc >= 0.95:
            dificultad = "TRIVIAL"
            emoji = "🔴"
            recomendacion = "Problema muy simple. Deep learning innecesario."
        elif dt10_acc >= 0.85:
            dificultad = "FÁCIL"
            emoji = "🟡"
            recomendacion = "Modelos clásicos suficientes. DL ofrece mejora marginal."
        elif dt10_acc >= 0.70:
            dificultad = "MODERADO"
            emoji = "🟢"
            recomendacion = "Deep learning justificado y recomendado."
        else:
            dificultad = "DIFÍCIL"
            emoji = "🔵"
            recomendacion = "Deep learning altamente recomendado."
        
        print(f"\n{emoji} DIFICULTAD: {dificultad}")
        print(f"   → {recomendacion}")
        
        # Recomendaciones específicas
        print(f"\n💡 RECOMENDACIONES:")
        
        if basic['balance_metric'] > 50:
            print(f"   ⚠️  Class weights OBLIGATORIOS (desbalance {basic['balance_metric']:.0f}%)")
            weights_needed = True
        elif basic['balance_metric'] > 20:
            print(f"   ⚠️  Class weights recomendados (desbalance {basic['balance_metric']:.0f}%)")
            weights_needed = True
        else:
            print(f"   ✅ Sin class weights (balanceado)")
            weights_needed = False
        
        if features['top_3_sum'] > 0.8:
            print(f"   ⚠️  Considerar feature selection (3 features dominan {features['top_3_sum']:.0%})")
        
        if dt10_acc >= 0.95:
            print(f"   ⚠️  Resultados perfectos son sospechosos - validar dataset")
        
        # Guardar en reporte
        report['grupos'][group_name] = {
            'n_muestras': int(basic['n_muestras']),
            'n_features': int(basic['n_features']),
            'balance_metric': float(basic['balance_metric']),
            'clase_0_pct': float(basic['clase_0_pct']),
            'clase_1_pct': float(basic['clase_1_pct']),
            'clase_2_pct': float(basic['clase_2_pct']),
            'dt1_accuracy': float(dt1_acc),
            'logreg_accuracy': float(logreg_acc),
            'dt10_accuracy': float(dt10_acc),
            'top3_feature_importance': float(features['top_3_sum']),
            'top10_feature_importance': float(features['top_10_sum']),
            'dificultad': dificultad,
            'class_weights_needed': weights_needed,
            'recomendacion': recomendacion
        }
    
    # Guardar reporte
    with open(OUTPUT_DIR / 'comprehensive_report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=4, ensure_ascii=False)
    
    print(f"\n\n📄 Reporte guardado: comprehensive_report.json")
    
    return report


def create_comparison_table(report):
    """Crear tabla comparativa."""
    print("\n" + "="*70)
    print("📊 TABLA COMPARATIVA FINAL")
    print("="*70)
    
    # Crear tabla
    data = []
    for group_name in ['ninos', 'jovenes', 'adultos']:
        g = report['grupos'][group_name]
        data.append({
            'Grupo': group_name.upper(),
            'Muestras': f"{g['n_muestras']:,}",
            'Features': g['n_features'],
            'Balance': f"{g['balance_metric']:.0f}%",
            'DT10 Acc': f"{g['dt10_accuracy']:.1%}",
            'Top3 Feat': f"{g['top3_feature_importance']:.0%}",
            'Dificultad': g['dificultad'],
            'Weights': '✅' if g['class_weights_needed'] else '❌'
        })
    
    df_table = pd.DataFrame(data)
    
    print("\n" + df_table.to_string(index=False))
    
    df_table.to_csv(OUTPUT_DIR / 'comparison_table.csv', index=False)
    
    return df_table


# ===================================================================
# MAIN
# ===================================================================

def main():
    """Ejecutar análisis completo."""
    print("="*70)
    print("🔬 ANÁLISIS COMPARATIVO COMPLETO - 3 GRUPOS ETARIOS")
    print("="*70)
    print(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output: {OUTPUT_DIR}")
    
    try:
        # 1. Cargar datos
        groups = load_all_groups()
        
        # 2. Análisis básico
        df_basic = analyze_basic_characteristics(groups)
        
        # 3. Análisis de complejidad
        df_complexity = analyze_problem_complexity(groups)
        
        # 4. Análisis de features
        feature_importance = analyze_feature_importance(groups)
        
        # 5. Reporte comprehensivo
        report = generate_comprehensive_report(groups, df_basic, df_complexity, feature_importance)
        
        # 6. Tabla comparativa
        comparison_table = create_comparison_table(report)
        
        print("\n" + "="*70)
        print("✅ ANÁLISIS COMPLETADO EXITOSAMENTE")
        print("="*70)
        print(f"\n📁 Archivos generados en: {OUTPUT_DIR}")
        print("   • basic_characteristics.csv")
        print("   • class_distributions.png")
        print("   • problem_complexity.csv")
        print("   • complexity_comparison.png")
        print("   • feature_importance.json")
        print("   • feature_concentration.png")
        print("   • comprehensive_report.json")
        print("   • comparison_table.csv")
        
        print("\n🎯 CONCLUSIONES PRINCIPALES:")
        print("\n   NIÑOS:")
        print(f"      • Dificultad: {report['grupos']['ninos']['dificultad']}")
        print(f"      • {report['grupos']['ninos']['recomendacion']}")
        
        print("\n   JÓVENES:")
        print(f"      • Dificultad: {report['grupos']['jovenes']['dificultad']}")
        print(f"      • {report['grupos']['jovenes']['recomendacion']}")
        
        print("\n   ADULTOS:")
        print(f"      • Dificultad: {report['grupos']['adultos']['dificultad']}")
        print(f"      • {report['grupos']['adultos']['recomendacion']}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
