import csv, os
import matplotlib.pyplot as plt

def plot_training_metrics(csv_path: str, out_dir: str):
    epochs, loss, acc, prec, rec, f1 = [], [], [], [], [], []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row['epoch']))
            loss.append(float(row['train_loss']))
            acc.append(float(row['val_accuracy']))
            prec.append(float(row['val_precision']))
            rec.append(float(row['val_recall']))
            f1.append(float(row['val_f1']))
    os.makedirs(out_dir, exist_ok=True)
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(epochs, loss, label='Train Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.grid(True); plt.legend()
    plt.subplot(1,2,2)
    plt.plot(epochs, acc, label='Val Acc')
    plt.plot(epochs, f1, label='Val F1')
    plt.xlabel('Epoch'); plt.ylabel('Score'); plt.grid(True); plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'training_metrics.png'), dpi=300, bbox_inches='tight')
    plt.close()
