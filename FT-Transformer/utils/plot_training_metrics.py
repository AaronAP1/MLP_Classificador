import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_training_metrics(metrics_csv, out_dir):
    df = pd.read_csv(metrics_csv)
    plt.figure(figsize=(8,5))
    plt.plot(df['epoch'], df['train_loss'], label='Train Loss')
    plt.plot(df['epoch'], df['val_f1'], label='Val F1')
    plt.xlabel('Epoch')
    plt.ylabel('Loss / F1')
    plt.legend()
    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, 'training_metrics.png'), dpi=300)
    plt.close()
