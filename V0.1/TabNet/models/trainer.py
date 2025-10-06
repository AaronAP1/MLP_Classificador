import os
import numpy as np
from utils.metrics import compute_metrics
from utils.plot_training_metrics import plot_training_metrics

class TabNetTrainer:
    def __init__(self, wrapper, config):
        self.wrapper = wrapper
        self.config = config
    def fit_eval(self, X_train, y_train, X_val, y_val, X_test, y_test, class_names):
        out_dir = self.config["outputs"]["model_dir"]
        os.makedirs(out_dir, exist_ok=True)
        max_epochs = self.config["model"]["epochs"]
        batch_size = self.config["model"]["batch_size"]
        patience = 20
        self.wrapper.fit(X_train, y_train, X_val, y_val, max_epochs=max_epochs, batch_size=batch_size, patience=patience)
        preds = self.wrapper.predict(X_test)
        probas = self.wrapper.predict_proba(X_test)
        metrics = compute_metrics(y_test, preds, probas)
        # Guardado básico (TabNet maneja su propio guardado si se usa save_model)
        return metrics
