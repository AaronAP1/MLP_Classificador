# -*- coding: utf-8 -*-
"""
Script de Diagnóstico - Detección de Data Leakage

Este script verifica si el 100% accuracy es legítimo o hay problemas:
1. Duplicados entre train/val/test
2. Features que son el target disfrazado
3. Problema trivialmente simple
4. Verificación con modelos simples

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
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report

# Agregar path del proyecto
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def check_duplicates(train_df, val_df, test_df):
    """Verificar duplicados entre conjuntos."""
    print("\n" + "="*70)
    print("🔍 VERIFICACIÓN 1: DUPLICADOS ENTRE CONJUNTOS")
    print("="*70)
    
    # Hash de cada fila (sin target)
    train_hashes = set(pd.util.hash_pandas_object(train_df.drop(columns=['target'])))
    val_hashes = set(pd.util.hash_pandas_object(val_df.drop(columns=['target'])))
    test_hashes = set(pd.util.hash_pandas_object(test_df.drop(columns=['target'])))
    
    # Intersecciones
    dup_train_val = len(train_hashes.intersection(val_hashes))
    dup_train_test = len(train_hashes.intersection(test_hashes))
    dup_val_test = len(val_hashes.intersection(test_hashes))
    
    print(f"\n📊 Resultados:")
    print(f"   Train-Val duplicados:  {dup_train_val:,}")
    print(f"   Train-Test duplicados: {dup_train_test:,}")
    print(f"   Val-Test duplicados:   {dup_val_test:,}")
    
    total_dups = dup_train_val + dup_train_test + dup_val_test
    
    if total_dups > 0:
        print(f"\n🔴 PROBLEMA CRÍTICO: {total_dups} duplicados detectados!")
        print("   → Hay DATA LEAKAGE confirmado")
        print("   → Acción: Re-hacer split de datos")
        return False
    else:
        print("\n✅ No hay duplicados entre conjuntos")
        return True


def check_target_correlation(train_df):
    """Verificar correlación con target."""
    print("\n" + "="*70)
    print("🔍 VERIFICACIÓN 2: CORRELACIÓN CON TARGET")
    print("="*70)
    
    X = train_df.drop(columns=['target'])
    y = train_df['target']
    
    # Calcular correlaciones
    correlations = []
    for col in X.columns:
        try:
            corr = np.abs(np.corrcoef(X[col], y)[0, 1])
            if not np.isnan(corr):
                correlations.append((col, corr))
        except:
            pass
    
    correlations.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n📊 Top 10 features más correlacionadas con target:")
    for i, (col, corr) in enumerate(correlations[:10], 1):
        if corr > 0.95:
            marker = "🔴 CRÍTICO"
        elif corr > 0.8:
            marker = "🟡 ALTO"
        elif corr > 0.5:
            marker = "🟢 MODERADO"
        else:
            marker = "⚪ BAJO"
        
        print(f"   {i:2d}. {marker} | {col:30s} | {corr:.4f}")
    
    # Análisis
    if correlations[0][1] > 0.98:
        print(f"\n🔴 SOSPECHOSO: '{correlations[0][0]}' tiene correlación {correlations[0][1]:.4f}")
        print("   → Esta feature podría ser el target disfrazado")
        print("   → O hay leakage de información futura")
        return False
    elif correlations[0][1] > 0.9:
        print(f"\n🟡 ADVERTENCIA: '{correlations[0][0]}' tiene correlación muy alta ({correlations[0][1]:.4f})")
        print("   → Verificar si es legítima")
        return None  # Inconcluso
    else:
        print(f"\n✅ Correlaciones normales (máx: {correlations[0][1]:.4f})")
        return True


def check_class_distribution(train_df, val_df, test_df):
    """Verificar distribución de clases."""
    print("\n" + "="*70)
    print("🔍 VERIFICACIÓN 3: DISTRIBUCIÓN DE CLASES")
    print("="*70)
    
    print("\n📊 Distribución por conjunto:")
    
    for name, df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
        counts = df['target'].value_counts().sort_index()
        total = len(df)
        
        print(f"\n   {name} ({total:,} muestras):")
        for cls, count in counts.items():
            pct = 100 * count / total
            print(f"      Clase {cls}: {count:5,} ({pct:5.2f}%)")
    
    # Verificar balance
    train_counts = train_df['target'].value_counts()
    train_pcts = 100 * train_counts / len(train_df)
    
    is_balanced = train_pcts.max() - train_pcts.min() < 20
    
    if is_balanced:
        print("\n✅ Dataset balanceado (diferencia < 20%)")
    else:
        print("\n⚠️  Dataset desbalanceado")
    
    return is_balanced


def test_simple_models(X_train, y_train, X_test, y_test):
    """Probar con modelos simples."""
    print("\n" + "="*70)
    print("🔍 VERIFICACIÓN 4: MODELOS SIMPLES")
    print("="*70)
    
    print("\n🧪 Entrenando modelos baseline...")
    
    results = []
    
    # 1. Decision Tree (depth=3)
    print("\n1. Decision Tree (max_depth=3)")
    dt = DecisionTreeClassifier(max_depth=3, random_state=42)
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    
    print(f"   Accuracy: {acc:.4f}")
    print(f"   F1-Macro: {f1:.4f}")
    results.append(('DecisionTree(3)', acc, f1))
    
    # 2. Logistic Regression
    print("\n2. Logistic Regression")
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    
    print(f"   Accuracy: {acc:.4f}")
    print(f"   F1-Macro: {f1:.4f}")
    results.append(('LogisticRegression', acc, f1))
    
    # 3. Decision Tree (depth=10)
    print("\n3. Decision Tree (max_depth=10)")
    dt10 = DecisionTreeClassifier(max_depth=10, random_state=42)
    dt10.fit(X_train, y_train)
    y_pred = dt10.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    
    print(f"   Accuracy: {acc:.4f}")
    print(f"   F1-Macro: {f1:.4f}")
    results.append(('DecisionTree(10)', acc, f1))
    
    # Análisis
    print("\n" + "-"*70)
    print("📊 ANÁLISIS:")
    
    best_acc = max(r[1] for r in results)
    
    if best_acc > 0.98:
        print(f"\n🔴 PROBLEMA: Modelo simple alcanza {best_acc:.2%} accuracy")
        print("   → Posible data leakage o problema trivial")
        print("   → Incluso modelos básicos resuelven el problema perfectamente")
        return False
    elif best_acc > 0.90:
        print(f"\n🟡 ADVERTENCIA: Modelo simple alcanza {best_acc:.2%} accuracy")
        print("   → Problema puede ser más simple de lo esperado")
        print("   → TabNet puede estar sobreajustando")
        return None
    else:
        print(f"\n✅ Modelos simples: {best_acc:.2%} accuracy (razonable)")
        print("   → TabNet tiene ventaja clara (100% vs {best_acc:.0%})")
        print("   → Justifica uso de deep learning")
        return True


def feature_importance_analysis(X_train, y_train, X_test, y_test):
    """Analizar importancia de features."""
    print("\n" + "="*70)
    print("🔍 VERIFICACIÓN 5: IMPORTANCIA DE FEATURES")
    print("="*70)
    
    print("\n🌳 Entrenando Decision Tree para importancia...")
    
    dt = DecisionTreeClassifier(max_depth=10, random_state=42)
    dt.fit(X_train, y_train)
    
    # Obtener importancias
    importances = dt.feature_importances_
    feature_names = [f"feat_{i}" for i in range(len(importances))]
    
    # Ordenar
    indices = np.argsort(importances)[::-1]
    
    print(f"\n📊 Top 10 features más importantes:")
    cumsum = 0
    for i in range(min(10, len(indices))):
        idx = indices[i]
        imp = importances[idx]
        cumsum += imp
        
        if imp > 0.5:
            marker = "🔴 DOMINA"
        elif imp > 0.2:
            marker = "🟡 ALTO"
        elif imp > 0.05:
            marker = "🟢 MODERADO"
        else:
            marker = "⚪ BAJO"
        
        print(f"   {i+1:2d}. {marker} | Feature {idx:3d} | {imp:.4f} ({cumsum:.1%} acumulado)")
    
    # Análisis
    if importances[indices[0]] > 0.8:
        print(f"\n🔴 PROBLEMA: Una sola feature domina ({importances[indices[0]]:.1%})")
        print("   → Problema puede ser trivialmente simple")
        print("   → O hay una feature que es el target")
        return False
    elif importances[indices[0]] > 0.5:
        print(f"\n🟡 ADVERTENCIA: Feature principal muy dominante ({importances[indices[0]]:.1%})")
        return None
    else:
        print(f"\n✅ Importancia distribuida (máx: {importances[indices[0]]:.1%})")
        return True


def generate_diagnosis_report(results):
    """Generar reporte final."""
    print("\n" + "="*70)
    print("📋 DIAGNÓSTICO FINAL")
    print("="*70)
    
    # Contar problemas
    critical = sum(1 for r in results.values() if r is False)
    warnings = sum(1 for r in results.values() if r is None)
    passed = sum(1 for r in results.values() if r is True)
    
    print(f"\n📊 Resumen:")
    print(f"   ✅ Verificaciones pasadas: {passed}/5")
    print(f"   🟡 Advertencias:           {warnings}/5")
    print(f"   🔴 Problemas críticos:     {critical}/5")
    
    print(f"\n📝 Detalle:")
    for name, result in results.items():
        if result is True:
            emoji = "✅"
        elif result is None:
            emoji = "🟡"
        else:
            emoji = "🔴"
        print(f"   {emoji} {name}")
    
    # Conclusión
    print("\n" + "="*70)
    print("🎯 CONCLUSIÓN")
    print("="*70)
    
    if critical > 0:
        print("\n🔴 DATA LEAKAGE DETECTADO")
        print("\n   Problemas encontrados:")
        for name, result in results.items():
            if result is False:
                print(f"      • {name}")
        
        print("\n   ⚠️  ACCIÓN REQUERIDA:")
        print("      1. NO usar estos resultados en la tesis")
        print("      2. Re-hacer split de datos (train/val/test)")
        print("      3. Verificar pipeline de preprocesamiento")
        print("      4. Asegurar independencia entre conjuntos")
        
    elif warnings > 0:
        print("\n🟡 RESULTADOS SOSPECHOSOS")
        print("\n   Requiere investigación adicional:")
        for name, result in results.items():
            if result is None:
                print(f"      • {name}")
        
        print("\n   💡 RECOMENDACIÓN:")
        print("      1. Revisar features con alta correlación")
        print("      2. Verificar si el problema es realista")
        print("      3. Considerar validación cruzada adicional")
        print("      4. Contrastar con resultados de jóvenes/adultos")
        
    else:
        print("\n✅ DATASET PARECE LEGÍTIMO")
        print("\n   El 100% accuracy es probablemente legítimo porque:")
        print("      • No hay duplicados entre conjuntos")
        print("      • No hay features sospechosas")
        print("      • Dataset bien balanceado")
        print("      • Modelos simples < 98% (TabNet tiene ventaja)")
        print("      • Importancia de features distribuida")
        
        print("\n   📝 PARA LA TESIS:")
        print("      ✅ Puedes usar estos resultados")
        print("      ✅ Explicar por qué niños es más fácil que otros grupos")
        print("      ✅ Comparar con jóvenes y adultos (probablemente < 100%)")
        print("      ✅ Discutir implicaciones pedagógicas")


def main():
    """Ejecutar diagnóstico completo."""
    print("="*70)
    print("🔬 DIAGNÓSTICO DE DATA LEAKAGE - Dataset de Niños")
    print("="*70)
    
    # Cargar datos
    data_dir = project_root / 'data' / 'processed' / 'ninos'
    
    print(f"\n📂 Cargando datos desde: {data_dir}")
    
    try:
        train_df = pd.read_csv(data_dir / 'ninos_train.csv')
        val_df = pd.read_csv(data_dir / 'ninos_val.csv')
        test_df = pd.read_csv(data_dir / 'ninos_test.csv')
    except Exception as e:
        print(f"\n❌ Error al cargar datos: {e}")
        return 1
    
    print(f"   Train: {len(train_df):,} muestras")
    print(f"   Val:   {len(val_df):,} muestras")
    print(f"   Test:  {len(test_df):,} muestras")
    print(f"   Features: {len(train_df.columns) - 1}")
    
    # Ejecutar verificaciones
    results = {}
    
    # 1. Duplicados
    results['Duplicados'] = check_duplicates(train_df, val_df, test_df)
    
    # 2. Correlación con target
    results['Correlación Target'] = check_target_correlation(train_df)
    
    # 3. Distribución
    results['Distribución'] = check_class_distribution(train_df, val_df, test_df)
    
    # 4. Modelos simples
    X_train = train_df.drop(columns=['target']).values
    y_train = train_df['target'].values
    X_test = test_df.drop(columns=['target']).values
    y_test = test_df['target'].values
    
    results['Modelos Simples'] = test_simple_models(X_train, y_train, X_test, y_test)
    
    # 5. Importancia de features
    results['Importancia Features'] = feature_importance_analysis(X_train, y_train, X_test, y_test)
    
    # Reporte final
    generate_diagnosis_report(results)
    
    print("\n" + "="*70)
    print("✅ DIAGNÓSTICO COMPLETADO")
    print("="*70)
    
    return 0


if __name__ == "__main__":
    exit(main())
