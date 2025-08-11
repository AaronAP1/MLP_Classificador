import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns  # Comentado para evitar dependencia
import os

def plot_training_metrics(metrics_csv_path, save_dir=None):
    """
    Visualiza las métricas de entrenamiento desde el CSV
    """
    # Leer datos
    df = pd.read_csv(metrics_csv_path)
    
    # Configurar estilo
    plt.style.use('default')  # Cambio de seaborn a default
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Métricas de Entrenamiento por Época', fontsize=16, fontweight='bold')
    
    # Loss de entrenamiento
    axes[0, 0].plot(df['epoch'], df['train_loss'], 'b-', linewidth=2, marker='o', markersize=4)
    axes[0, 0].set_title('Pérdida de Entrenamiento')
    axes[0, 0].set_xlabel('Época')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy de validación
    axes[0, 1].plot(df['epoch'], df['val_accuracy'], 'g-', linewidth=2, marker='s', markersize=4)
    axes[0, 1].set_title('Accuracy de Validación')
    axes[0, 1].set_xlabel('Época')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Precision y Recall de validación
    axes[1, 0].plot(df['epoch'], df['val_precision'], 'r-', linewidth=2, marker='^', markersize=4, label='Precision')
    axes[1, 0].plot(df['epoch'], df['val_recall'], 'orange', linewidth=2, marker='v', markersize=4, label='Recall')
    axes[1, 0].set_title('Precision y Recall de Validación')
    axes[1, 0].set_xlabel('Época')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # F1-Score de validación
    axes[1, 1].plot(df['epoch'], df['val_f1'], 'purple', linewidth=2, marker='d', markersize=4)
    axes[1, 1].set_title('F1-Score de Validación')
    axes[1, 1].set_xlabel('Época')
    axes[1, 1].set_ylabel('F1-Score')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Ajustar layout
    plt.tight_layout()
    
    # Guardar si se especifica directorio
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'training_metrics.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Gráfico de métricas guardado en: {save_path}")
    
    plt.show()
    
    # Resumen estadístico
    print("\nRESUMEN DE MÉTRICAS DE ENTRENAMIENTO:")
    print("="*50)
    print(f"Mejor Accuracy: {df['val_accuracy'].max():.4f} (Época {df.loc[df['val_accuracy'].idxmax(), 'epoch']})")
    print(f"Mejor Precision: {df['val_precision'].max():.4f} (Época {df.loc[df['val_precision'].idxmax(), 'epoch']})")
    print(f"Mejor Recall: {df['val_recall'].max():.4f} (Época {df.loc[df['val_recall'].idxmax(), 'epoch']})")
    print(f"Mejor F1-Score: {df['val_f1'].max():.4f} (Época {df.loc[df['val_f1'].idxmax(), 'epoch']})")
    print(f"Loss final: {df['train_loss'].iloc[-1]:.4f}")

if __name__ == "__main__":
    # Ejemplo de uso
    import sys
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
        save_dir = sys.argv[2] if len(sys.argv) > 2 else None
        plot_training_metrics(csv_path, save_dir)
    else:
        print("Uso: python plot_training_metrics.py <path_to_metrics.csv> [save_directory]")
