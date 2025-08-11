import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from utils.metrics import compute_metrics
from utils.plot_training_metrics import plot_training_metrics

def train_and_evaluate(model, train_loader, val_loader, test_loader, config, class_names, device=None):
    device = device or (torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    model.to(device)
    y_train_all = np.array([y for _, _, y in train_loader.dataset])
    counts = np.bincount(y_train_all)
    class_weights = (1.0 / np.maximum(counts, 1)).astype(np.float32)
    class_weights = class_weights / class_weights.sum()
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, device=device))
    optimizer = optim.AdamW(model.parameters(), lr=config["model"]["lr"], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)
    scaler = torch.cuda.amp.GradScaler(enabled=config["model"].get("amp", False) and torch.cuda.is_available())
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
        for Xn, Xc, yb in train_loader:
            Xn, Xc, yb = Xn.to(device), Xc.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=config["model"].get("amp", False) and torch.cuda.is_available()):
                logits, _ = model(Xn, Xc)
                loss = criterion(logits, yb)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
        model.eval()
        all_preds, all_targets, all_probs = [], [], []
        with torch.no_grad():
            for Xn, Xc, yv in val_loader:
                Xn, Xc = Xn.to(device), Xc.to(device)
                logits, _ = model(Xn, Xc)
                probs = torch.softmax(logits, dim=1)
                preds = logits.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_targets.extend(yv.numpy())
                all_probs.extend(probs.cpu().numpy())
        all_probs = np.array(all_probs)
        val_metrics = compute_metrics(all_targets, all_preds, all_probs)
        with open(metrics_csv, 'a', encoding='utf-8') as f:
            f.write(f"{epoch},{total_loss/max(1,len(train_loader)):.4f},{val_metrics['accuracy']:.4f},{val_metrics['precision']:.4f},{val_metrics['recall']:.4f},{val_metrics['f1']:.4f}\n")
        scheduler.step(val_metrics['f1'])
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            patience_counter = 0
            torch.save(model.state_dict(), best_path)
        else:
            patience_counter += 1
        if patience_counter >= max_patience:
            break
    model.load_state_dict(torch.load(best_path, map_location=device))
    model.eval()
    all_preds, all_targets, all_probs = [], [], []
    with torch.no_grad():
        for Xn, Xc, yt in test_loader:
            Xn, Xc = Xn.to(device), Xc.to(device)
            logits, _ = model(Xn, Xc)
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(yt.numpy())
            all_probs.extend(probs.cpu().numpy())
    all_probs = np.array(all_probs)
    test_metrics = compute_metrics(all_targets, all_preds, all_probs)
    plot_training_metrics(metrics_csv, config["outputs"]["plots_dir"])
    return test_metrics
