import json
import sys
sys.path.append('.')
from utils.etl import preprocess_and_load

# Cargar configuraciones
with open('configs/config_jovenes.json') as f:
    config = json.load(f)
with open('configs/paths.json') as f:
    paths = json.load(f)

print('Iniciando preprocesamiento del dataset completo...')
try:
    train_loader, val_loader, test_loader, input_dim, n_classes = preprocess_and_load(config, paths)
    print(f'✓ Preprocesamiento exitoso!')
    print(f'  Input dimension: {input_dim}')
    print(f'  Number of classes: {n_classes}')
    print(f'  Train batches: {len(train_loader)}')
    print(f'  Validation batches: {len(val_loader)}')
    print(f'  Test batches: {len(test_loader)}')
except Exception as e:
    print(f'✗ Error: {e}')
    import traceback
    traceback.print_exc()
