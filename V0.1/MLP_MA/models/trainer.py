import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from utils.metrics import compute_metrics, plot_roc_curve, print_classification_report, plot_confusion_matrix


def train_and_evaluate(model, train_loader, val_loader, test_loader, config, class_names, device=None):
    device = device or (torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    model.to(device)

    # Pesos por clase desde train
    y_train_all = np.array([y for _, y in train_loader.dataset])
    counts = np.bincount(y_train_all)
    class_weights = (1.0 / np.maximum(counts, 1)).astype(np.float32)
    class_weights = class_weights / class_weights.sum()
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, device=device))

    optimizer = optim.AdamW(model.parameters(), lr=config["model"]["lr"], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)

    # AMP solo si hay CUDA disponible y está habilitado en config
    amp_requested = config["model"].get("amp", False)
    amp_enabled = amp_requested and torch.cuda.is_available()
    if amp_requested and not amp_enabled:
        print("AMP solicitado pero CUDA no disponible; se entrenará en precisión estándar.")
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    max_grad_norm = 1.0

    best_f1 = 0.0
    patience_counter = 0
    max_patience = 20

    os.makedirs(config["outputs"]["model_dir"], exist_ok=True)
    metrics_csv = os.path.join(config["outputs"]["model_dir"], "metrics_by_epoch.csv")
    with open(metrics_csv, 'w', encoding='utf-8') as f:
        f.write("epoch,train_loss,val_accuracy,val_precision,val_recall,val_f1\n")

    best_path = os.path.join(config["outputs"]["model_dir"], "best_model.pt")

    for epoch in range(1, config["model"]["epochs"] + 1):
        model.train()
        total_loss = 0.0
        bad_batches = 0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                logits, _ = model(Xb)
                loss = criterion(logits, yb)
            if torch.isnan(loss) or torch.isinf(loss):
                bad_batches += 1
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
        if bad_batches:
            print(f"⚠️  Época {epoch}: {bad_batches} batches con NaN/Inf omitidos")

        # Validación
        model.eval()
        all_preds, all_targets, all_probs = [], [], []
        with torch.no_grad():
            for Xv, yv in val_loader:
                Xv = Xv.to(device)
                logits, _ = model(Xv)
                probs = torch.softmax(logits, dim=1)
                preds = logits.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_targets.extend(yv.numpy())
                all_probs.extend(probs.cpu().numpy())
        all_probs = np.array(all_probs)
        val_metrics = compute_metrics(all_targets, all_preds, all_probs)

        with open(metrics_csv, 'a', encoding='utf-8') as f:
            f.write(f"{epoch},{total_loss/max(1,len(train_loader)):.4f},{val_metrics['accuracy']:.4f},{val_metrics['precision']:.4f},{val_metrics['recall']:.4f},{val_metrics['f1']:.4f}\n")

        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_metrics['f1'])
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr != old_lr:
            print(f"📉 LR: {old_lr:.6f} → {new_lr:.6f}")

        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            patience_counter = 0
            torch.save(model.state_dict(), best_path)
            print(f"🎯 Nuevo mejor F1: {best_f1:.4f}")
        else:
            patience_counter += 1
        if patience_counter >= max_patience:
            print(f"⏹️  Early stopping en época {epoch}")
            break

    print(f"Mejor F1 val: {best_f1:.4f} → {best_path}")

    # Evaluación en test
    model.load_state_dict(torch.load(best_path, map_location=device))
    model.eval()
    all_preds, all_targets, all_probs = [], [], []
    with torch.no_grad():
        for Xt, yt in test_loader:
            Xt = Xt.to(device)
            logits, _ = model(Xt)
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(yt.numpy())
            all_probs.extend(probs.cpu().numpy())
    all_probs = np.array(all_probs)
    test_metrics = compute_metrics(all_targets, all_preds, all_probs)
    print_classification_report(test_metrics, class_names)

    os.makedirs(config["outputs"]["plots_dir"], exist_ok=True)
    cm_path = os.path.join(config["outputs"]["plots_dir"], "confusion_matrix.png")
    plot_confusion_matrix(test_metrics['confusion_matrix'], class_names, normalize=False, title="Matriz de confusión - Test", save_path=cm_path)
    cmn_path = os.path.join(config["outputs"]["plots_dir"], "confusion_matrix_normalized.png")
    plot_confusion_matrix(test_metrics['confusion_matrix'], class_names, normalize=True, title="Matriz de confusión normalizada - Test", save_path=cmn_path)
    roc_path = os.path.join(config["outputs"]["plots_dir"], "roc_curve.png")
    plot_roc_curve(all_targets, all_probs, class_names, save_path=roc_path)

    # Guardar metrics finales
    final_metrics_path = os.path.join(config["outputs"]["model_dir"], "final_test_metrics.txt")
    with open(final_metrics_path, 'w', encoding='utf-8') as f:
        f.write("METRICAS TEST\n")
        f.write(f"Accuracy: {test_metrics['accuracy']:.4f}\n")
        f.write(f"Precision (weighted): {test_metrics['precision']:.4f}\n")
        f.write(f"Recall (weighted): {test_metrics['recall']:.4f}\n")
        f.write(f"F1 (weighted): {test_metrics['f1']:.4f}\n")
        if 'roc_auc' in test_metrics and test_metrics['roc_auc'] is not None:
            f.write(f"ROC AUC (weighted): {test_metrics['roc_auc']:.4f}\n")

    return test_metrics
