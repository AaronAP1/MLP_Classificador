import pandas as pd

for group in ['adultos', 'jovenes', 'ninos']:
    df = pd.read_csv(f'data/unprocessed/{group}.csv')
    print(f'\n{"="*50}')
    print(f'{group.upper()}')
    print(f'{"="*50}')
    
    if 'habito_lectura_completo' in df.columns:
        print('Distribución ORIGINAL (raw):')
        print(df['habito_lectura_completo'].value_counts())
    else:
        print('⚠️  No hay columna habito_lectura_completo')
    
    # Ver distribución en datos procesados
    df_proc = pd.read_csv(f'data/processed/{group}/{group}_train.csv')
    print('\nDistribución PROCESADA (train):')
    print(df_proc['target'].value_counts())
    print(f'\nPorcentajes:')
    print(df_proc['target'].value_counts(normalize=True) * 100)
