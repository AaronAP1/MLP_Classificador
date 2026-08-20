"""
Las tres arquitecturas del articulo (seccion III-C).

Correcciones respecto a la implementacion auditada:

  * MLP+MA ahora implementa atencion multi-cabeza real:
    Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) V, con las cabezas
    concatenadas y proyectadas por W_O, tal como declara la ecuacion de
    la seccion III-C-1. La version anterior reducia Q y K a un escalar
    por muestra, le aplicaba sigmoid y la usaba como compuerta global:
    no habia cabezas, ni softmax, ni pesos por variable, lo que ademas
    hacia imposible la lectura de importancia por variable.

  * TabNet usa 3 decision steps y mascaras sparsemax (el codigo anterior
    usaba 5 steps y entmax).

  * FT-Transformer conserva su diseño (embeddings de 64, 4 encoders,
    token [CLS]) y ademas expone los pesos de atencion por cabeza y capa,
    necesarios para la seccion III-G-2, que antes no se podian extraer.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn


# =====================================================================
# 1. MLP + Multi-head Attention
# =====================================================================
class MultiHeadFeatureAttention(nn.Module):
    """
    Atencion multi-cabeza sobre las variables de entrada.

    Cada una de las `num_features` variables se trata como un token de
    dimension 1 y se proyecta a `attn_dim`. Sobre esa secuencia se aplica
    self-attention estandar, de modo que el peso a_ij expresa cuanta
    importancia asigna la variable i a la variable j -- exactamente la
    lectura que la seccion III-G del articulo necesita.
    """

    def __init__(self, num_features: int, attn_dim: int = 64, n_heads: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        if attn_dim % n_heads != 0:
            raise ValueError(f"attn_dim ({attn_dim}) debe ser divisible por n_heads ({n_heads})")

        self.num_features = num_features
        self.attn_dim = attn_dim
        self.n_heads = n_heads
        self.d_k = attn_dim // n_heads

        # Token por variable: proyeccion escalar -> attn_dim
        self.embed = nn.Parameter(torch.empty(num_features, attn_dim))
        self.embed_bias = nn.Parameter(torch.zeros(num_features, attn_dim))
        nn.init.xavier_uniform_(self.embed)

        self.W_q = nn.Linear(attn_dim, attn_dim)
        self.W_k = nn.Linear(attn_dim, attn_dim)
        self.W_v = nn.Linear(attn_dim, attn_dim)
        self.W_o = nn.Linear(attn_dim, attn_dim)          # proyeccion de salida
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(attn_dim)

        # Colapsa el token atendido de vuelta a un escalar por variable,
        # para que la salida conserve la dimension original de entrada.
        self.collapse = nn.Linear(attn_dim, 1)

        for layer in (self.W_q, self.W_k, self.W_v, self.W_o):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: (B, num_features)
        Returns:
            (B, num_features) y, opcionalmente, los pesos (B, n_heads, F, F)
        """
        B, F_ = x.shape
        # Tokenizacion: (B, F, attn_dim)
        tokens = x.unsqueeze(-1) * self.embed.unsqueeze(0) + self.embed_bias.unsqueeze(0)

        q = self.W_q(tokens).view(B, F_, self.n_heads, self.d_k).transpose(1, 2)
        k = self.W_k(tokens).view(B, F_, self.n_heads, self.d_k).transpose(1, 2)
        v = self.W_v(tokens).view(B, F_, self.n_heads, self.d_k).transpose(1, 2)

        # softmax(QK^T / sqrt(d_k)) V
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn = torch.softmax(scores, dim=-1)
        # El dropout se aplica solo a la copia que alimenta el producto;
        # los pesos que se devuelven para interpretabilidad conservan la
        # propiedad de sumar 1 por consulta.
        heads = torch.matmul(self.dropout(attn), v)       # (B, H, F, d_k)

        # Concat de cabezas + W_O
        concat = heads.transpose(1, 2).contiguous().view(B, F_, self.attn_dim)
        out = self.norm(self.W_o(concat) + tokens)        # residual
        attended = self.collapse(out).squeeze(-1)         # (B, F)

        return attended, (attn if return_attention else None)


class MLPMultiHeadAttention(nn.Module):
    """
    Atencion multi-cabeza sobre las variables, seguida de un MLP denso
    256 -> 128 -> 64 -> 32 con ReLU y Dropout, y una capa softmax de 3
    salidas (seccion III-C-1).
    """

    def __init__(
        self,
        num_features: int,
        num_classes: int = 3,
        hidden_dims: Optional[List[int]] = None,
        n_heads: int = 4,
        attn_dim: int = 64,
        dropout: float = 0.3,
        depth: Optional[int] = None,
    ):
        super().__init__()
        hidden_dims = list(hidden_dims or [256, 128, 64, 32])
        if depth is not None:
            hidden_dims = hidden_dims[:depth] if depth <= len(hidden_dims) else (
                hidden_dims + [hidden_dims[-1]] * (depth - len(hidden_dims))
            )

        self.num_features = num_features
        self.attention = MultiHeadFeatureAttention(num_features, attn_dim, n_heads, dropout)

        layers: List[nn.Module] = []
        in_dim = num_features
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        self.mlp = nn.Sequential(*layers)
        self.classifier = nn.Linear(in_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attended, _ = self.attention(x)
        return self.classifier(self.mlp(attended))

    @torch.no_grad()
    def feature_importance(self, x: torch.Tensor) -> np.ndarray:
        """
        Importancia por variable: atencion recibida, promediada sobre
        cabezas y consultas. Devuelve un vector de longitud num_features.
        """
        self.eval()
        _, attn = self.attention(x, return_attention=True)
        return attn.mean(dim=(0, 1, 2)).cpu().numpy()


# =====================================================================
# 2. TabNet
# =====================================================================
class TabNetModel:
    """
    Envoltura de pytorch-tabnet fijada a los valores del articulo:
    3 decision steps y mascaras sparsemax (seccion III-C-2).
    """

    def __init__(
        self,
        num_features: int,
        num_classes: int = 3,
        n_steps: int = 3,
        mask_type: str = "sparsemax",
        n_d: int = 32,
        n_a: int = 32,
        gamma: float = 1.3,
        n_independent: int = 2,
        n_shared: int = 2,
        lambda_sparse: float = 1e-3,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        seed: int = 42,
        device: str = "auto",
    ):
        from pytorch_tabnet.tab_model import TabNetClassifier

        self.num_features = num_features
        self.num_classes = num_classes
        self.n_steps = n_steps
        self.mask_type = mask_type

        self.model = TabNetClassifier(
            n_d=n_d, n_a=n_a, n_steps=n_steps, gamma=gamma,
            n_independent=n_independent, n_shared=n_shared,
            lambda_sparse=lambda_sparse, mask_type=mask_type,
            optimizer_fn=torch.optim.AdamW,                     # III-E-2
            optimizer_params={"lr": lr, "weight_decay": weight_decay},
            # pytorch-tabnet invoca scheduler.step() sin argumentos, por lo
            # que ReduceLROnPlateau no es utilizable aqui. StepLR produce el
            # mismo decaimiento gradual que en los modelos PyTorch.
            scheduler_fn=torch.optim.lr_scheduler.StepLR,
            scheduler_params={"step_size": 50, "gamma": 0.9},
            seed=seed, verbose=0,
            device_name=("cuda" if torch.cuda.is_available() else "cpu")
            if device == "auto" else device,
        )

    def fit(self, X_tr, y_tr, X_val, y_val, epochs: int, patience: int,
            batch_size: int, virtual_batch_size: int = 128,
            class_weights: Optional[np.ndarray] = None):
        weights = None
        if class_weights is not None:
            per_class = {i: float(w) for i, w in enumerate(class_weights)}
            weights = np.array([per_class[int(c)] for c in y_tr])

        self.model.fit(
            X_train=np.asarray(X_tr, dtype=np.float32),
            y_train=np.asarray(y_tr).astype(int),
            eval_set=[(np.asarray(X_val, dtype=np.float32), np.asarray(y_val).astype(int))],
            eval_name=["val"], eval_metric=["logloss", "accuracy"],
            max_epochs=epochs, patience=patience,
            batch_size=batch_size,
            virtual_batch_size=min(virtual_batch_size, batch_size),
            weights=weights if weights is not None else 0,
            drop_last=False,
        )
        return self.history

    @property
    def history(self) -> Dict[str, list]:
        h = getattr(self.model, "history", None)
        if h is None:
            return {}
        raw = h.history if hasattr(h, "history") else dict(h)
        return {
            "train_loss": list(raw.get("loss", [])),
            "val_loss": list(raw.get("val_logloss", [])),
            "val_acc": list(raw.get("val_accuracy", [])),
        }

    def predict(self, X) -> np.ndarray:
        return self.model.predict(np.asarray(X, dtype=np.float32))

    def predict_proba(self, X) -> np.ndarray:
        return self.model.predict_proba(np.asarray(X, dtype=np.float32))

    def feature_importance(self, X=None) -> np.ndarray:
        """
        Importancia global I(v_i) = suma de las mascaras sobre los pasos
        (seccion III-G-1). pytorch-tabnet ya entrega esta agregacion.
        """
        return np.asarray(self.model.feature_importances_)

    def step_masks(self, X) -> Dict[str, np.ndarray]:
        """Mascara de cada decision step, para la lectura step-wise."""
        _, masks = self.model.explain(np.asarray(X, dtype=np.float32))
        return {f"step_{i + 1}": np.asarray(m).mean(axis=0) for i, m in masks.items()}

    def save(self, path): self.model.save_model(str(path))
    def load(self, path): self.model.load_model(str(path))


# =====================================================================
# 3. FT-Transformer
# =====================================================================
class FeatureTokenizer(nn.Module):
    """Cada variable escalar -> embedding de `emb_dim` dimensiones."""

    def __init__(self, num_features: int, emb_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_features, emb_dim))
        self.bias = nn.Parameter(torch.zeros(num_features, emb_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)


class FTTransformer(nn.Module):
    """
    FT-Transformer con token [CLS] (seccion III-C-3).

    A diferencia de la version auditada, el encoder se construye capa a
    capa para poder devolver los pesos de atencion de cada cabeza y cada
    capa, que la seccion III-G-2 necesita para calcular

        I(v_i) = (1 / HL) * sum_h sum_l sum_j alpha_ij^(h,l)
    """

    def __init__(
        self,
        num_features: int,
        num_classes: int = 3,
        emb_dim: int = 64,
        n_layers: int = 4,
        n_heads: int = 8,
        dim_feedforward: int = 256,
        dropout: float = 0.3,
        depth: Optional[int] = None,   # ignorado: el articulo fija 4 encoders
    ):
        super().__init__()
        if emb_dim % n_heads != 0:
            raise ValueError(f"emb_dim ({emb_dim}) debe ser divisible por n_heads ({n_heads})")

        self.num_features = num_features
        self.emb_dim = emb_dim
        self.n_layers = n_layers
        self.n_heads = n_heads

        self.tokenizer = FeatureTokenizer(num_features, emb_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_dim))
        nn.init.normal_(self.cls_token, std=0.02)

        self.attn = nn.ModuleList([
            nn.MultiheadAttention(emb_dim, n_heads, dropout=dropout, batch_first=True)
            for _ in range(n_layers)
        ])
        self.ffn = nn.ModuleList([
            nn.Sequential(
                nn.Linear(emb_dim, dim_feedforward), nn.GELU(),
                nn.Dropout(dropout), nn.Linear(dim_feedforward, emb_dim),
            ) for _ in range(n_layers)
        ])
        self.norm1 = nn.ModuleList([nn.LayerNorm(emb_dim) for _ in range(n_layers)])
        self.norm2 = nn.ModuleList([nn.LayerNorm(emb_dim) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)

        self.head = nn.Sequential(
            nn.LayerNorm(emb_dim), nn.Linear(emb_dim, emb_dim // 2),
            nn.ReLU(), nn.Dropout(dropout), nn.Linear(emb_dim // 2, num_classes),
        )

    def _encode(self, x: torch.Tensor, collect: bool):
        B = x.size(0)
        tokens = self.tokenizer(x)
        cls = self.cls_token.expand(B, -1, -1)
        h = torch.cat([cls, tokens], dim=1)               # (B, F+1, emb)

        maps = []
        for i in range(self.n_layers):
            a, w = self.attn[i](h, h, h, need_weights=collect,
                                average_attn_weights=False)
            h = self.norm1[i](h + self.dropout(a))
            h = self.norm2[i](h + self.dropout(self.ffn[i](h)))
            if collect and w is not None:
                maps.append(w)                            # (B, heads, F+1, F+1)
        return h, maps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self._encode(x, collect=False)
        return self.head(h[:, 0, :])                      # clasificacion por [CLS]

    @torch.no_grad()
    def feature_importance(self, x: torch.Tensor) -> np.ndarray:
        """
        I(v_i) promediando la atencion recibida por cada variable sobre
        todas las cabezas y capas. Se descarta la posicion 0 ([CLS]).
        """
        self.eval()
        _, maps = self._encode(x, collect=True)
        if not maps:
            return np.zeros(self.num_features)
        stacked = torch.stack(maps)                       # (L, B, H, F+1, F+1)
        received = stacked.mean(dim=(0, 1, 2, 3))         # promedio sobre L,B,H,i
        return received[1:].cpu().numpy()

    @torch.no_grad()
    def attention_maps(self, x: torch.Tensor) -> np.ndarray:
        """Mapa (L, H, F+1, F+1) promediado sobre el batch, para heatmaps."""
        self.eval()
        _, maps = self._encode(x, collect=True)
        return torch.stack(maps).mean(dim=1).cpu().numpy()


# =====================================================================
# Fabrica
# =====================================================================
def build_model(name: str, num_features: int, cfg: dict, hp: dict, seed: int = 42):
    """
    Crea una arquitectura a partir de la configuracion estructural
    (`cfg['models'][name]`) y de los hiperparametros buscados (`hp`).
    """
    mcfg = cfg["models"][name]
    n_classes = cfg["n_classes"]

    if name == "mlp_ma":
        return MLPMultiHeadAttention(
            num_features=num_features, num_classes=n_classes,
            hidden_dims=mcfg["hidden_dims"], n_heads=mcfg["n_heads"],
            attn_dim=mcfg["attn_dim"], dropout=hp["dropout"],
            depth=hp.get("depth"),
        )

    if name == "ft_transformer":
        return FTTransformer(
            num_features=num_features, num_classes=n_classes,
            emb_dim=mcfg["emb_dim"], n_layers=mcfg["n_layers"],
            n_heads=mcfg["n_heads"], dim_feedforward=mcfg["dim_feedforward"],
            dropout=hp["dropout"],
        )

    if name == "tabnet":
        # El articulo fija 3 decision steps (III-C-2). Solo el preset
        # "paper" los explora, y lo hace con la clave explicita `n_steps`
        # que emite `gridsearch.build_grid`; `depth` (numero de capas del
        # MLP) nunca debe reinterpretarse como decision steps.
        n_steps = int(hp.get("n_steps", mcfg["n_steps"]))
        n_steps = n_steps if 2 <= n_steps <= 4 else mcfg["n_steps"]
        return TabNetModel(
            num_features=num_features, num_classes=n_classes,
            n_steps=n_steps, mask_type=mcfg["mask_type"],
            n_d=mcfg["n_d"], n_a=mcfg["n_a"], gamma=mcfg["gamma"],
            n_independent=mcfg["n_independent"], n_shared=mcfg["n_shared"],
            lambda_sparse=mcfg["lambda_sparse"],
            lr=hp["lr"], weight_decay=hp["weight_decay"], seed=seed,
        )

    raise ValueError(f"Modelo desconocido: {name!r}")


MODEL_NAMES = ["mlp_ma", "tabnet", "ft_transformer"]
IS_TORCH = {"mlp_ma": True, "ft_transformer": True, "tabnet": False}
