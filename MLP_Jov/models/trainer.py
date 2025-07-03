import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import precision_score, f1_score, confusion_matrix
from utils.visualizations import plot_confusion_matrix
from utils.metrics import compute_metrics, plot_roc_curve, print_classification_report

def train_and_evaluate(model,
                       train_loader,
                       val_loader,
                       test_loader,
                       config,
                       class_names):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # 1) Calcula pesos inversos a la frecuencia de las clases en train
    # Extrae todos los labels del dataset de entrenamiento
    y_train_all = np.array([y for _, y in train_loader.dataset])
    counts = np.bincount(y_train_all)
    class_weights = 1.0 / counts
    class_weights = class_weights / class_weights.sum()       # opcional: normalizar
    class_weights = torch.tensor(class_weights,
                                 dtype=torch.float32,
                                 device=device)

    # 2) Define la pérdida con esos pesos
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # 3) Optimizador (Adam) con scheduler
    optimizer = optim.Adam(model.parameters(), lr=config["model"]["lr"], weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)
    
    # Gradient clipping para evitar explosión de gradientes
    max_grad_norm = 1.0

    best_f1 = 0.0
    patience_counter = 0
    max_patience = 20
    os.makedirs(config["outputs"]["model_dir"], exist_ok=True)
    best_path = os.path.join(config["outputs"]["model_dir"], "best_model.pt")

     # Prepara CSV de métricas por época
    metrics_csv = os.path.join(config["outputs"]["model_dir"], "metrics_by_epoch.csv")
    with open(metrics_csv, 'w') as f:
        f.write("epoch,train_loss,val_accuracy,val_precision,val_recall,val_f1\n")

    # 4) Loop de entrenamiento
    for epoch in range(1, config["model"]["epochs"] + 1):
        model.train()
        total_loss = 0.0
        nan_batches = 0

        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            
            # Verificar datos de entrada
            if torch.isnan(Xb).any() or torch.isinf(Xb).any():
                nan_batches += 1
                continue
                
            optimizer.zero_grad()
            logits = model(Xb)
            
            # Verificar salida del modelo
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                nan_batches += 1
                continue
                
            loss = criterion(logits, yb)
            
            # Verificar si la pérdida es válida
            if torch.isnan(loss) or torch.isinf(loss):
                nan_batches += 1
                continue
                
            loss.backward()
            
            # Verificar gradientes
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                nan_batches += 1
                continue
            
            optimizer.step()
            total_loss += loss.item()
            
        # Reportar problemas de NaN/Inf al final de la época
        if nan_batches > 0:
            print(f"⚠️  Época {epoch}: {nan_batches} batches saltados por NaN/Inf")

        # 5) Validación
        model.eval()
        all_preds, all_targets, all_probs = [], [], []
        with torch.no_grad():
            for Xv, yv in val_loader:
                Xv = Xv.to(device)
                logits = model(Xv)
                probs = torch.softmax(logits, dim=1)
                preds = logits.argmax(dim=1).cpu().numpy()
                
                all_preds.extend(preds)
                all_targets.extend(yv.numpy())
                all_probs.extend(probs.cpu().numpy())

        # Calcular métricas de validación
        all_probs = np.array(all_probs)
        val_metrics = compute_metrics(all_targets, all_preds, all_probs)

        # Guardar métricas en CSV
        with open(metrics_csv, 'a') as f:
            f.write(f"{epoch},{total_loss/len(train_loader):.4f},"
                   f"{val_metrics['accuracy']:.4f},{val_metrics['precision']:.4f},"
                   f"{val_metrics['recall']:.4f},{val_metrics['f1']:.4f}\n")

        # 2.2) (Opcional) checkpoint de cada época
        epoch_chk = os.path.join(config["outputs"]["model_dir"], f"epoch_{epoch:02d}.pt")
        torch.save(model.state_dict(), epoch_chk)

        print(f"Epoch {epoch:02d}  loss={total_loss/len(train_loader):.4f}"
              f"  val_acc={val_metrics['accuracy']:.4f}  val_prec={val_metrics['precision']:.4f}"
              f"  val_rec={val_metrics['recall']:.4f}  val_f1={val_metrics['f1']:.4f}")

        # Ajustar el scheduler
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_metrics['f1'])
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr != old_lr:
            print(f"📉 Learning rate reducido: {old_lr:.6f} → {new_lr:.6f}")

        # Early Stopping
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            patience_counter = 0
            torch.save(model.state_dict(), best_path)
            print(f"🎯 Nuevo mejor F1: {best_f1:.4f}")
        else:
            patience_counter += 1

        if patience_counter >= max_patience:
            print(f"⏹️  Early stopping en época {epoch:02d} (sin mejora por {max_patience} épocas)")
            break

    print(f"\nMejor F1 validación: {best_f1:.4f} → modelo guardado en {best_path}")

    # 6) Evaluación final en Test
    model.load_state_dict(torch.load(best_path))
    model.eval()
    all_preds, all_targets, all_probs = [], [], []
    with torch.no_grad():
        for Xt, yt in test_loader:
            Xt = Xt.to(device)
            logits = model(Xt)
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1).cpu().numpy()
            
            all_preds.extend(preds)
            all_targets.extend(yt.numpy())
            all_probs.extend(probs.cpu().numpy())

    # Calcular métricas completas de test
    all_probs = np.array(all_probs)
    test_metrics = compute_metrics(all_targets, all_preds, all_probs)

    # Imprimir reporte completo
    print_classification_report(test_metrics, class_names)

    # 7) Guardar y visualizar resultados
    os.makedirs(config["outputs"]["plots_dir"], exist_ok=True)
    
    # Matriz de confusión
    cm_path = os.path.join(config["outputs"]["plots_dir"], "confusion_matrix.png")
    plot_confusion_matrix(test_metrics['confusion_matrix'],
                          classes=class_names,
                          normalize=False,
                          title="Matriz de confusión - Test",
                          save_path=cm_path)
    print(f"Matriz de confusión guardada en {cm_path}")
    
    # Matriz de confusión normalizada
    cm_norm_path = os.path.join(config["outputs"]["plots_dir"], "confusion_matrix_normalized.png")
    plot_confusion_matrix(test_metrics['confusion_matrix'],
                          classes=class_names,
                          normalize=True,
                          title="Matriz de confusión normalizada - Test",
                          save_path=cm_norm_path)
    print(f"Matriz de confusión normalizada guardada en {cm_norm_path}")
    
    # Curva ROC
    roc_path = os.path.join(config["outputs"]["plots_dir"], "roc_curve.png")
    plot_roc_curve(all_targets, all_probs, class_names, save_path=roc_path)
    print(f"Curva ROC guardada en {roc_path}")
    
    # Guardar métricas finales en archivo
    final_metrics_path = os.path.join(config["outputs"]["model_dir"], "final_test_metrics.txt")
    with open(final_metrics_path, 'w') as f:
        f.write("MÉTRICAS FINALES DE TEST\n")
        f.write("="*40 + "\n\n")
        f.write(f"Accuracy Global: {test_metrics['accuracy']:.4f}\n")
        f.write(f"Precision (weighted): {test_metrics['precision']:.4f}\n")
        f.write(f"Recall (weighted): {test_metrics['recall']:.4f}\n")
        f.write(f"F1-Score (weighted): {test_metrics['f1']:.4f}\n")
        if 'roc_auc' in test_metrics and test_metrics['roc_auc'] is not None:
            f.write(f"ROC AUC (weighted): {test_metrics['roc_auc']:.4f}\n")
        
        f.write(f"\nMÉTRICAS POR CLASE:\n")
        f.write(f"{'Clase':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}\n")
        f.write("-" * 56 + "\n")
        
        for i, class_name in enumerate(class_names):
            prec = test_metrics['precision_per_class'][i]
            rec = test_metrics['recall_per_class'][i]
            f1 = test_metrics['f1_per_class'][i]
            f.write(f"{class_name:<20} {prec:<12.4f} {rec:<12.4f} {f1:<12.4f}\n")
        
        f.write(f"\nMATRIZ DE CONFUSIÓN:\n")
        f.write(str(test_metrics['confusion_matrix']))
    
    print(f"Métricas finales guardadas en {final_metrics_path}")
    
    return test_metrics
