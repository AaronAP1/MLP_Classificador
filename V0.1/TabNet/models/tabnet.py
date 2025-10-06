from pytorch_tabnet.tab_model import TabNetClassifier
import torch

class TabNetWrapper:
    def __init__(self, input_dim, output_dim, cat_idxs, cat_dims, cat_emb_dim, config):
        self.model = TabNetClassifier(
            n_d=config.get('n_d', 32),
            n_a=config.get('n_a', 32),
            n_steps=config.get('n_steps', 5),
            gamma=config.get('gamma', 1.5),
            cat_idxs=cat_idxs,
            cat_dims=cat_dims,
            cat_emb_dim=cat_emb_dim,
            n_independent=config.get('n_independent', 2),
            n_shared=config.get('n_shared', 2),
            lambda_sparse=config.get('lambda_sparse', 1e-3),
            optimizer_fn=torch.optim.Adam,
            optimizer_params=dict(lr=config.get('lr', 2e-2)),
            mask_type=config.get('mask_type', 'entmax'),
            verbose=1
        )
    def fit(self, X_train, y_train, X_val, y_val, max_epochs=200, batch_size=128, patience=20):
        self.model.fit(
            X_train=X_train, y_train=y_train,
            eval_set=[(X_val, y_val)],
            max_epochs=max_epochs,
            patience=patience,
            batch_size=batch_size,
            virtual_batch_size=batch_size//4,
            num_workers=0,
            drop_last=False
        )
    def predict(self, X):
        return self.model.predict(X)
    def predict_proba(self, X):
        return self.model.predict_proba(X)
    def explain(self, X):
        return self.model.explain(X)
