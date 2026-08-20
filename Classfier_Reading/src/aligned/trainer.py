"""
Entrenador comun para las arquitecturas PyTorch (MLP+MA y FT-Transformer).

Fija lo que declara el articulo:
  * optimizador AdamW              (III-E-2)
  * Cross-Entropy Loss             (III-E-2)
  * early stopping paciencia 20    (III-E-3)
  * 200 epocas en el ajuste final  (IV)
  * gradient clipping y ReduceLROnPlateau

Registra `train_acc` y `val_acc` por epoca, que es lo que necesita la
figura de curvas de aprendizaje. La implementacion anterior solo
guardaba la perdida, por lo que las curvas del articulo no podian
provenir de ella.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def make_loader(X, y, batch_size: int, shuffle: bool, seed: int = 42,
                num_workers: int = 0) -> DataLoader:
    ds = TensorDataset(
        torch.as_tensor(np.asarray(X), dtype=torch.float32),
        torch.as_tensor(np.asarray(y), dtype=torch.long),
    )
    gen = torch.Generator().manual_seed(seed)
    # pin_memory acelera la copia host->GPU; en CPU no aporta nada.
    pin = torch.cuda.is_available()
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      generator=gen if shuffle else None, drop_last=False,
                      pin_memory=pin, num_workers=num_workers,
                      persistent_workers=bool(num_workers))


def compute_class_weights(y, n_classes: int) -> np.ndarray:
    """Pesos inversamente proporcionales a la frecuencia de cada clase."""
    y = np.asarray(y).astype(int)
    counts = np.bincount(y, minlength=n_classes).astype(float)
    counts[counts == 0] = 1.0
    w = len(y) / (n_classes * counts)
    return w


class TorchTrainer:
    def __init__(
        self,
        model: nn.Module,
        lr: float,
        weight_decay: float,
        class_weights: Optional[np.ndarray] = None,
        gradient_clip: float = 1.0,
        scheduler_patience: int = 10,
        scheduler_factor: float = 0.5,
        device: Optional[str] = None,
        amp: Optional[bool] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.gradient_clip = gradient_clip
        # Precision mixta: solo en GPU, donde reduce memoria y acelera el
        # FT-Transformer, que es la arquitectura mas cara del estudio.
        self.amp = (self.device.startswith("cuda") if amp is None else amp)
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp)

        # III-E-2: AdamW, no Adam.
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min",
            patience=scheduler_patience, factor=scheduler_factor,
        )

        w = (torch.as_tensor(class_weights, dtype=torch.float32).to(self.device)
             if class_weights is not None else None)
        self.criterion = nn.CrossEntropyLoss(weight=w)

        self.history: Dict[str, list] = {
            "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []
        }
        self.best_state: Optional[dict] = None
        self.best_val_loss = float("inf")
        self.best_epoch = 0

    def _run_epoch(self, loader: DataLoader, train: bool) -> Tuple[float, float]:
        self.model.train() if train else self.model.eval()
        total_loss, correct, total = 0.0, 0, 0

        with torch.set_grad_enabled(train):
            for xb, yb in loader:
                xb = xb.to(self.device, non_blocking=True)
                yb = yb.to(self.device, non_blocking=True)
                if train:
                    self.optimizer.zero_grad(set_to_none=True)

                with torch.autocast("cuda", enabled=self.amp):
                    logits = self.model(xb)
                    loss = self.criterion(logits, yb)

                if train:
                    self.scaler.scale(loss).backward()
                    if self.gradient_clip:
                        self.scaler.unscale_(self.optimizer)
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                total_loss += loss.item() * yb.size(0)
                correct += (logits.argmax(1) == yb).sum().item()
                total += yb.size(0)

        return total_loss / max(total, 1), correct / max(total, 1)

    def fit(self, train_loader, val_loader, epochs: int, patience: int,
            verbose: bool = True) -> Dict[str, list]:
        stale = 0
        for epoch in range(epochs):
            tr_loss, tr_acc = self._run_epoch(train_loader, train=True)
            va_loss, va_acc = self._run_epoch(val_loader, train=False)

            self.history["train_loss"].append(tr_loss)
            self.history["train_acc"].append(tr_acc)
            self.history["val_loss"].append(va_loss)
            self.history["val_acc"].append(va_acc)
            self.scheduler.step(va_loss)

            if va_loss < self.best_val_loss - 1e-6:
                self.best_val_loss, self.best_epoch, stale = va_loss, epoch, 0
                self.best_state = {k: v.detach().cpu().clone()
                                   for k, v in self.model.state_dict().items()}
            else:
                stale += 1

            if verbose and (epoch + 1) % 20 == 0:
                print(f"    epoca {epoch + 1:3d}/{epochs} | "
                      f"train {tr_loss:.4f}/{tr_acc:.4f} | "
                      f"val {va_loss:.4f}/{va_acc:.4f}")

            if stale >= patience:          # III-E-3
                if verbose:
                    print(f"    early stopping en la epoca {epoch + 1} "
                          f"(mejor: {self.best_epoch + 1})")
                break

        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)
        return self.history

    @torch.no_grad()
    def predict_proba(self, X) -> np.ndarray:
        self.model.eval()
        xb = torch.as_tensor(np.asarray(X), dtype=torch.float32).to(self.device)
        out = []
        for i in range(0, len(xb), 4096):                 # por lotes, por memoria
            with torch.autocast("cuda", enabled=self.amp):
                logits = self.model(xb[i:i + 4096])
            out.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
        return np.vstack(out)

    def predict(self, X) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)

    def save(self, path):
        torch.save({"model_state_dict": self.model.state_dict(),
                    "history": self.history,
                    "best_epoch": self.best_epoch,
                    "best_val_loss": self.best_val_loss}, str(path))
