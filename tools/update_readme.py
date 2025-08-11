import os, re, glob, json
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
README = os.path.join(ROOT, 'README.md')

# Folders to scan: model roots
MODELS = [
    {
        'name': 'MLP_MA',
        'root': os.path.join(ROOT, 'MLP_MA'),
    },
    {
        'name': 'FT-Transformer',
        'root': os.path.join(ROOT, 'FT-Transformer'),
    },
    {
        'name': 'TabNet',
        'root': os.path.join(ROOT, 'TabNet'),
    },
]

# Find latest run by timestamp folder name under outputs/models and outputs/plots

def latest_run_dir(base_dir):
    if not os.path.isdir(base_dir):
        return None
    subs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    # prefer YYYYMMDD_HHMMSS folders
    cand = []
    for d in subs:
        try:
            datetime.strptime(d, '%Y%m%d_%H%M%S')
            cand.append(d)
        except Exception:
            continue
    if not cand:
        return None
    cand.sort()
    return cand[-1]


def read_metrics_txt(path):
    if not path or not os.path.exists(path):
        return None
    lines = open(path, 'r', encoding='utf-8').read().strip().splitlines()
    # Expect lines like: "Precision (weighted): 0.9876"
    data = {}
    for ln in lines:
        if ':' in ln:
            k, v = ln.split(':', 1)
            data[k.strip()] = v.strip()
    return data


def build_section():
    blocks = []
    for m in MODELS:
        plots_dir = os.path.join(m['root'], 'outputs', 'plots')
        models_dir = os.path.join(m['root'], 'outputs', 'models')
        # fall back to root outputs if per-model not populated
        alt_plots = os.path.join(ROOT, 'outputs', 'plots')
        alt_models = os.path.join(ROOT, 'outputs', 'models')
        run_plot = latest_run_dir(plots_dir) or latest_run_dir(alt_plots)
        run_model = latest_run_dir(models_dir) or latest_run_dir(alt_models)
        # use a single run_id if possible
        run_id = run_plot or run_model
        if not run_id:
            blocks.append(f"### {m['name']}\nSin corridas recientes.\n")
            continue
        # relative paths from README
        rel_plots = None
        if os.path.isdir(os.path.join(m['root'], 'outputs', 'plots', run_id)):
            rel_plots = os.path.join(os.path.relpath(m['root'], ROOT), 'outputs', 'plots', run_id)
        elif os.path.isdir(os.path.join(ROOT, 'outputs', 'plots', run_id)):
            rel_plots = os.path.join('outputs', 'plots', run_id)
        rel_models = None
        if os.path.isdir(os.path.join(m['root'], 'outputs', 'models', run_id)):
            rel_models = os.path.join(os.path.relpath(m['root'], ROOT), 'outputs', 'models', run_id)
        elif os.path.isdir(os.path.join(ROOT, 'outputs', 'models', run_id)):
            rel_models = os.path.join('outputs', 'models', run_id)
        metrics = None
        if rel_models:
            metrics = read_metrics_txt(os.path.join(ROOT, rel_models, 'final_test_metrics.txt'))
        block = [f"### {m['name']} (run {run_id})"]
        if metrics:
            prec = metrics.get('Precision (weighted)', 'N/A')
            f1 = metrics.get('F1 (weighted)', 'N/A')
            acc = metrics.get('Accuracy', 'N/A')
            block.append(f"- Accuracy: {acc} | Precision (weighted): {prec} | F1 (weighted): {f1}")
        if rel_plots:
            cm = os.path.join(rel_plots, 'confusion_matrix.png')
            roc = os.path.join(rel_plots, 'roc_curve.png')
            train = os.path.join(rel_plots, 'training_metrics.png')
            block.append("")
            block.append(f"![CM]({cm})")
            block.append(f"![ROC]({roc})")
            block.append(f"![Train]({train})")
        blocks.append('\n'.join(block))
    return '\n\n'.join(blocks)


def update_readme():
    content = open(README, 'r', encoding='utf-8').read()
    start = '<!-- AUTO-PLOTS-START -->'
    end = '<!-- AUTO-PLOTS-END -->'
    if start not in content or end not in content:
        print('Markers not found in README')
        return
    section = build_section()
    pattern = re.compile(rf"{re.escape(start)}[\s\S]*?{re.escape(end)}")
    def repl(_):
        return start + "\n" + section + "\n" + end
    new_content = pattern.sub(repl, content)
    with open(README, 'w', encoding='utf-8') as f:
        f.write(new_content)
    print('README actualizado')

if __name__ == '__main__':
    update_readme()
