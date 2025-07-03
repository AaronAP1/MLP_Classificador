import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import precision_score, f1_score, confusion_matrix
from utils.visualizations import plot_confusion_matrix

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

    # 3) Optimizador (Adam)
    optimizer = optim.Adam(model.parameters(), lr=config["model"]["lr"])

    best_f1 = 0.0
    os.makedirs(config["outputs"]["model_dir"], exist_ok=True)
    best_path = os.path.join(config["outputs"]["model_dir"], "best_model.pt")

     # Prepara CSV de métricas por época
    metrics_csv = os.path.join(config["outputs"]["model_dir"], "metrics_by_epoch.csv")
    with open(metrics_csv, 'w') as f:
        f.write("epoch,loss,val_prec,val_f1\n")

    # 4) Loop de entrenamiento
    for epoch in range(1, config["model"]["epochs"] + 1):
        model.train()
        total_loss = 0.0

        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(Xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # 5) Validación
        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for Xv, yv in val_loader:
                Xv = Xv.to(device)
                preds = model(Xv).argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_targets.extend(yv.numpy())

        val_prec = precision_score(all_targets, all_preds, average='weighted', zero_division=0)
        val_f1   = f1_score(all_targets, all_preds, average='weighted', zero_division=0)

                # 2.1) Guardar métricas en CSV
        with open(metrics_csv, 'a') as f:
            f.write(f"{epoch},{total_loss/len(train_loader):.4f},{val_prec:.4f},{val_f1:.4f}\n")

        # 2.2) (Opcional) checkpoint de cada época
        epoch_chk = os.path.join(config["outputs"]["model_dir"], f"epoch_{epoch:02d}.pt")
        torch.save(model.state_dict(), epoch_chk)

        print(f"Epoch {epoch:02d}  loss={total_loss/len(train_loader):.4f}"
              f"  val_prec={val_prec:.4f}  val_f1={val_f1:.4f}")

        # Guarda el mejor según F1 de validación
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), best_path)

    print(f"\nMejor F1 validación: {best_f1:.4f} → modelo guardado en {best_path}")

    # 6) Evaluación final en Test
    model.load_state_dict(torch.load(best_path))
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for Xt, yt in test_loader:
            Xt = Xt.to(device)
            preds = model(Xt).argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(yt.numpy())

    test_prec = precision_score(all_targets, all_preds, average='weighted', zero_division=0)
    test_f1   = f1_score(all_targets, all_preds, average='weighted', zero_division=0)
    cm = confusion_matrix(all_targets, all_preds)

    print(f"\nTest precision: {test_prec:.4f}")
    print(f"Test F1-score: {test_f1:.4f}")
    print("Matriz de confusión (sin normalizar):")
    print(cm)

    # 7) Guarda y dibuja la matriz de confusión
    os.makedirs(config["outputs"]["plots_dir"], exist_ok=True)
    cm_path = os.path.join(config["outputs"]["plots_dir"], "confusion_matrix.png")
    plot_confusion_matrix(cm,
                          classes=class_names,
                          normalize=False,
                          title="Matriz de confusión",
                          save_path=cm_path)
    print(f"Confusion matrix guardada en {cm_path}")
