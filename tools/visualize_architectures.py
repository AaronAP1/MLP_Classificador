"""
Visualización mejorada de arquitecturas FT-Transformer y TabNet
con estilo más cercano a redes neuronales
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch, Rectangle
import numpy as np

# Configuración global
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 10

def draw_neuron(ax, x, y, radius=0.15, color='#4A90E2', label=''):
    """Dibuja una neurona circular"""
    circle = Circle((x, y), radius, color=color, ec='black', linewidth=1.5, zorder=3)
    ax.add_patch(circle)
    if label:
        ax.text(x, y, label, ha='center', va='center', fontsize=7, 
                color='white', fontweight='bold', zorder=4)
    return circle

def draw_layer_neurons(ax, x, y_start, y_end, n_neurons, color='#4A90E2', label_prefix=''):
    """Dibuja múltiples neuronas en una columna"""
    neurons = []
    if n_neurons <= 8:
        y_positions = np.linspace(y_start, y_end, n_neurons)
        for i, y in enumerate(y_positions):
            neuron = draw_neuron(ax, x, y, radius=0.15, color=color, 
                               label=f'{label_prefix}{i+1}' if label_prefix else '')
            neurons.append((x, y))
    else:
        # Si hay muchas neuronas, mostrar solo algunas representativas
        y_positions = np.linspace(y_start, y_end, 8)
        for i in range(3):
            draw_neuron(ax, x, y_positions[i], radius=0.15, color=color)
            neurons.append((x, y_positions[i]))
        # Puntos suspensivos
        ax.text(x, y_positions[4], '...', ha='center', va='center', 
               fontsize=16, fontweight='bold')
        for i in range(5, 8):
            draw_neuron(ax, x, y_positions[i], radius=0.15, color=color)
            neurons.append((x, y_positions[i]))
    return neurons

def draw_connection(ax, x1, y1, x2, y2, alpha=0.3, color='gray', linewidth=0.5):
    """Dibuja conexión entre neuronas"""
    ax.plot([x1, x2], [y1, y2], color=color, alpha=alpha, linewidth=linewidth, zorder=1)

def draw_box(ax, x, y, width, height, color='#FFE4B5', text='', text_size=10):
    """Dibuja un recuadro con texto"""
    box = FancyBboxPatch((x - width/2, y - height/2), width, height,
                          boxstyle="round,pad=0.05", 
                          facecolor=color, edgecolor='black',
                          linewidth=2, zorder=2)
    ax.add_patch(box)
    if text:
        ax.text(x, y, text, ha='center', va='center', 
               fontsize=text_size, fontweight='bold', zorder=3)

def draw_arrow(ax, x1, y1, x2, y2, color='black', linewidth=2):
    """Dibuja flecha entre componentes"""
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                           arrowstyle='->', mutation_scale=20,
                           color=color, linewidth=linewidth, zorder=2)
    ax.add_patch(arrow)

def visualize_ft_transformer():
    """Genera visualización de FT-Transformer con estilo de red neuronal"""
    fig, ax = plt.subplots(figsize=(18, 10))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Fondo limpio
    ax.add_patch(Rectangle((0, 0), 18, 10, facecolor='white', zorder=0))
    
    # Título
    ax.text(9, 9.5, 'FT-Transformer: Feature Tokenizer Transformer', 
           ha='center', fontsize=16, fontweight='bold', color='#2C3E50')
    ax.text(9, 9.1, '71 features → 64-dim embeddings → 4 Transformer Layers → 3 classes',
           ha='center', fontsize=10, color='#7F8C8D')
    
    # === CAPA 1: INPUT FEATURES ===
    input_x = 1.5
    input_neurons = draw_layer_neurons(ax, input_x, 2, 8, 12, color='#3498DB')
    ax.text(input_x, 8.6, 'INPUT', ha='center', fontsize=11, fontweight='bold', color='#2980B9')
    ax.text(input_x, 0.8, '(71 features)', ha='center', fontsize=8, color='#7F8C8D')
    
    # === CAPA 2: TOKENIZATION ===
    token_x = 3.2
    token_neurons = draw_layer_neurons(ax, token_x, 2, 8, 12, color='#9B59B6')
    ax.text(token_x, 8.6, 'TOKENIZE', ha='center', fontsize=11, fontweight='bold', color='#8E44AD')
    ax.text(token_x, 0.8, '(71 × 64d)', ha='center', fontsize=8, color='#7F8C8D')
    
    # Conexiones densas input -> tokens
    for (x1, y1) in input_neurons:
        for (x2, y2) in token_neurons[:8]:
            draw_connection(ax, x1, y1, x2, y2, alpha=0.05, linewidth=0.3, color='#9B59B6')
    
    # [CLS] Token especial
    cls_token_y = 1
    draw_neuron(ax, token_x, cls_token_y, radius=0.2, color='#E74C3C', label='CLS')
    
    # === TRANSFORMER LAYERS (4 capas) ===
    layer_x_positions = [5.2, 7.2, 9.2, 11.2]
    layer_colors = ['#F39C12', '#E67E22', '#D68910', '#CA6F1E']
    
    prev_layer_neurons = token_neurons
    
    for idx, (layer_x, color) in enumerate(zip(layer_x_positions, layer_colors)):
        # Neuronas del transformer layer (representando embeddings)
        layer_neurons = draw_layer_neurons(ax, layer_x, 2.5, 7.5, 10, color=color)
        
        # Label
        ax.text(layer_x, 8.6, f'LAYER {idx+1}', ha='center', fontsize=10, fontweight='bold', color='#E67E22')
        ax.text(layer_x, 8.2, 'Attention+FF', ha='center', fontsize=7, style='italic', color='#7F8C8D')
        
        # Conexiones desde capa anterior (más densas)
        for (x1, y1) in prev_layer_neurons[:8]:
            for (x2, y2) in layer_neurons[:6]:
                draw_connection(ax, x1, y1, x2, y2, alpha=0.08, linewidth=0.3, color=color)
        
        # Indicador visual de atención (líneas que conectan dentro de la misma capa)
        if len(layer_neurons) >= 4:
            # Self-attention visual (algunas neuronas conectadas entre sí)
            for i in range(0, min(4, len(layer_neurons)), 2):
                if i+1 < len(layer_neurons):
                    x_n, y_n = layer_neurons[i]
                    x_n2, y_n2 = layer_neurons[i+1]
                    ax.plot([x_n + 0.15, x_n2 - 0.15], [y_n, y_n2], 
                           'r-', alpha=0.2, linewidth=1.5, zorder=1)
        
        prev_layer_neurons = layer_neurons
    
    # === [CLS] OUTPUT EXTRACTION ===
    cls_out_x = 13
    cls_out_y = 5
    draw_neuron(ax, cls_out_x, cls_out_y, radius=0.35, color='#E74C3C', label='CLS')
    ax.text(cls_out_x, 8.6, '[CLS]', ha='center', fontsize=11, fontweight='bold', color='#C0392B')
    ax.text(cls_out_x, 8.2, 'Embedding', ha='center', fontsize=8, color='#7F8C8D')
    
    # Conexiones desde última capa hacia [CLS] (enfocadas)
    last_layer_x = layer_x_positions[-1]
    last_layer_neurons_subset = prev_layer_neurons[:8]
    for (x1, y1) in last_layer_neurons_subset:
        draw_connection(ax, x1, y1, cls_out_x - 0.4, cls_out_y, 
                       alpha=0.25, linewidth=0.8, color='#E74C3C')
    
    # === OUTPUT LAYER (3 clases) ===
    output_x = 15.5
    output_classes = ['Lector\nFrecuente', 'Lector\nMedio', 'No\nLector']
    output_colors = ['#27AE60', '#F39C12', '#E74C3C']
    output_y_positions = [6.5, 5, 3.5]
    
    ax.text(output_x, 8.6, 'OUTPUT', ha='center', fontsize=11, fontweight='bold', color='#16A085')
    ax.text(output_x, 8.2, '(Softmax)', ha='center', fontsize=8, style='italic', color='#7F8C8D')
    
    for y_pos, class_name, color in zip(output_y_positions, output_classes, output_colors):
        # Neurona de salida
        draw_neuron(ax, output_x, y_pos, radius=0.3, color=color)
        
        # Label a la derecha
        ax.text(output_x + 1, y_pos, class_name, ha='left', va='center',
               fontsize=9, fontweight='bold', color=color)
        
        # Conexiones desde [CLS]
        draw_connection(ax, cls_out_x + 0.4, cls_out_y, output_x - 0.35, y_pos,
                       alpha=0.6, linewidth=2, color=color)
    
    # === ANOTACIONES VISUALES ===
    # Flechas de flujo principal (arriba)
    arrow_y = 9.3
    flow_arrows = [
        (input_x + 0.3, token_x - 0.3),
        (token_x + 0.3, layer_x_positions[0] - 0.3),
        (layer_x_positions[-1] + 0.3, cls_out_x - 0.5),
        (cls_out_x + 0.5, output_x - 0.5)
    ]
    
    for x1, x2 in flow_arrows:
        ax.annotate('', xy=(x2, arrow_y), xytext=(x1, arrow_y),
                   arrowprops=dict(arrowstyle='->', lw=2.5, color='#34495E'))
    
    # Etiquetas de proceso
    process_labels = [
        ((input_x + token_x)/2, 'Embed'),
        ((token_x + layer_x_positions[0])/2, 'Encode'),
        ((layer_x_positions[-1] + cls_out_x)/2, 'Extract'),
        ((cls_out_x + output_x)/2, 'Classify')
    ]
    
    for x, label in process_labels:
        ax.text(x, arrow_y + 0.3, label, ha='center', fontsize=8, 
               fontweight='bold', color='#34495E')
    
    # === INDICADORES DE ATENCIÓN ===
    # Líneas curvas mostrando self-attention entre layers
    for i in range(len(layer_x_positions) - 1):
        x1 = layer_x_positions[i]
        x2 = layer_x_positions[i + 1]
        # Arco superior
        ax.annotate('', xy=(x2 - 0.2, 7.8), xytext=(x1 + 0.2, 7.8),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3',
                                 lw=1.5, color='#E67E22', alpha=0.5))
    
    # === LEYENDA ===
    legend_items = [
        (1, '●', '#3498DB', 'Input'),
        (3, '●', '#9B59B6', 'Tokens'),
        (5, '●', '#E67E22', 'Transform'),
        (7.5, '●', '#E74C3C', '[CLS]'),
        (9.5, '●', '#27AE60', 'Output')
    ]
    
    legend_y = 0.3
    for x, symbol, color, label in legend_items:
        ax.text(x, legend_y, symbol, ha='center', fontsize=16, color=color, fontweight='bold')
        ax.text(x + 0.35, legend_y, label, ha='left', fontsize=9, va='center', color='#2C3E50')
    
    # Anotaciones adicionales
    ax.text(12, 0.3, '├─ Multi-Head Attention', ha='left', fontsize=7, color='#7F8C8D')
    ax.text(12, 0.05, '└─ Feed Forward Network', ha='left', fontsize=7, color='#7F8C8D')
    
    plt.tight_layout()
    return fig

def visualize_tabnet():
    """Genera visualización mejorada de TabNet"""
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Título
    ax.text(8, 9.5, 'TabNet: Attentive Interpretable Tabular Learning',
           ha='center', fontsize=18, fontweight='bold')
    ax.text(8, 9, 'Input: 71 features → Decision Steps: 3 → Output: 3 classes',
           ha='center', fontsize=11, style='italic', color='gray')
    
    # 1. Input Features
    input_neurons = draw_layer_neurons(ax, 1, 1.5, 7.5, 10, color='#4A90E2')
    ax.text(1, 8.2, 'Input Layer\n(71 features)', ha='center',
           fontsize=10, fontweight='bold')
    
    # 2. Feature Transformer inicial
    ft_x = 3
    draw_box(ax, ft_x, 4.5, 1.8, 2.5, color='#BB8FCE',
            text='Feature\nTransformer\n\n(BN + FC)', text_size=9)
    
    # Conexiones input -> feature transformer
    for (x1, y1) in input_neurons[:5]:
        draw_arrow(ax, x1 + 0.2, y1, ft_x - 1, 4.5, color='gray', linewidth=0.8)
    
    # 3. Decision Steps (3 pasos)
    step_positions = [5.5, 8.5, 11.5]
    step_colors = ['#FFE4B5', '#FFD4A5', '#FFC495']
    
    decision_outputs = []
    
    for idx, (step_x, color) in enumerate(zip(step_positions, step_colors)):
        step_y = 5
        
        # Attention Mask (arriba)
        attention_y = 7
        draw_box(ax, step_x, attention_y, 1.6, 0.8, color='#FADBD8',
                text=f'Attention\nMask {idx+1}', text_size=8)
        
        # Decision Step (centro)
        draw_box(ax, step_x, step_y, 1.6, 1.5, color=color,
                text=f'Decision\nStep {idx+1}\n\nFC → ReLU', text_size=9)
        
        # Label
        ax.text(step_x, 7.9, f'Step {idx+1}', ha='center',
               fontsize=10, fontweight='bold', color='#34495E')
        
        # Neuronas de salida del step
        step_out_neurons = draw_layer_neurons(ax, step_x, 2.5, 3.5, 4, color='#F39C12')
        decision_outputs.append((step_x, step_out_neurons))
        
        # Flecha desde attention a decision
        draw_arrow(ax, step_x, attention_y - 0.5, step_x, step_y + 0.8,
                  color='#E74C3C', linewidth=2)
        
        # Conexión desde Feature Transformer o paso anterior
        if idx == 0:
            draw_arrow(ax, ft_x + 1, 4.5, step_x - 0.9, step_y, linewidth=2)
        else:
            prev_x = step_positions[idx - 1]
            draw_arrow(ax, prev_x + 0.9, step_y, step_x - 0.9, step_y,
                      linewidth=2, color='#3498DB')
        
        # Visualizar sparse selection (líneas punteadas hacia attention)
        if idx == 0:
            for i in range(3):
                ax.plot([ft_x + 0.5, step_x - 0.3], [5 + i*0.5, attention_y - 0.3],
                       'r--', alpha=0.4, linewidth=1)
    
    # 4. Agregación (suma)
    agg_x = 13.5
    agg_y = 4
    draw_box(ax, agg_x, agg_y, 1.2, 1.2, color='#AED6F1',
            text='∑\n\nAggregate', text_size=11)
    
    # Conexiones desde decision steps a agregación
    for step_x, neurons in decision_outputs:
        draw_arrow(ax, step_x + 0.8, 3, agg_x - 0.7, agg_y,
                  color='#F39C12', linewidth=1.5)
    
    # 5. Output Layer (3 clases)
    output_x = 15.5
    output_classes = ['Lector\nFrecuente', 'Lector\nMedio', 'No\nLector']
    output_colors = ['#27AE60', '#F39C12', '#E74C3C']
    output_y_positions = [6, 4, 2]
    
    for y_pos, class_name, color in zip(output_y_positions, output_classes, output_colors):
        draw_neuron(ax, output_x, y_pos, radius=0.25, color=color)
        ax.text(output_x + 0.7, y_pos, class_name, ha='left', va='center',
               fontsize=9, fontweight='bold')
        # Conexión desde agregación
        draw_arrow(ax, agg_x + 0.7, agg_y, output_x - 0.3, y_pos,
                  color='gray', linewidth=1.5)
    
    ax.text(output_x, 7, 'Output\n(Softmax)', ha='center',
           fontsize=10, fontweight='bold')
    
    # Leyenda y explicación
    legend_y = 1.2
    ax.text(0.5, legend_y, '● Input Features', color='#4A90E2', fontsize=9, fontweight='bold')
    ax.text(3, legend_y, '□ Feature Transform', color='#BB8FCE', fontsize=9, fontweight='bold')
    ax.text(6, legend_y, '□ Attention Mask (Sparse)', color='#FADBD8', fontsize=9, fontweight='bold')
    ax.text(9.5, legend_y, '● Decision Output', color='#F39C12', fontsize=9, fontweight='bold')
    
    ax.text(0.5, 0.7, '• Sequential attention selects relevant features at each step',
           fontsize=8, style='italic')
    ax.text(0.5, 0.4, '• Sparse masks enable interpretability (see which features are used)',
           fontsize=8, style='italic')
    
    # Metadata
    ax.text(8, 0.1, 'Optimizer: Adam | Loss: CrossEntropyLoss | Key Feature: Interpretable Feature Selection (entmax)',
           ha='center', fontsize=8, style='italic', color='gray')
    
    plt.tight_layout()
    return fig

def visualize_mlp_attention():
    """Genera visualización mejorada de MLP + Multi-head Attention"""
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Título
    ax.text(8, 9.5, 'MLP + Multi-head Attention Architecture',
           ha='center', fontsize=18, fontweight='bold')
    ax.text(8, 9, 'Input: 71 features → Hidden Layers: 4 → Output: 3 classes',
           ha='center', fontsize=11, style='italic', color='gray')
    
    # 1. Input Features
    input_neurons = draw_layer_neurons(ax, 1, 1.5, 7.5, 10, color='#4A90E2')
    ax.text(1, 8.2, 'Input Layer\n(71 features)', ha='center',
           fontsize=10, fontweight='bold')
    
    # 2. Multi-head Attention (primero)
    att_x = 3.5
    att_y = 4.5
    
    # Dibujar 3 "heads"
    head_colors = ['#E8DAEF', '#D7BDE2', '#C39BD3']
    for i, color in enumerate(head_colors):
        y_offset = (i - 1) * 0.4
        draw_box(ax, att_x, att_y + y_offset, 1.4, 0.6, color=color,
                text=f'Head {i+1}', text_size=8)
    
    ax.text(att_x, 6.2, 'Multi-head\nAttention\n(3 heads)', ha='center',
           fontsize=10, fontweight='bold')
    
    # Conexiones input -> attention
    for (x1, y1) in input_neurons[:5]:
        for i in range(3):
            draw_connection(ax, x1 + 0.2, y1, att_x - 0.8, att_y + (i-1)*0.4,
                          alpha=0.15, color='purple')
    
    # 3. Hidden Layers (4 capas densas)
    hidden_configs = [
        (6, 256, '#FFE4B5'),
        (8, 128, '#FFD4A5'),
        (10, 64, '#FFC495'),
        (12, 32, '#FFB485')
    ]
    
    prev_neurons = None
    for layer_idx, (layer_x, n_units, color) in enumerate(hidden_configs):
        # Dibujar neuronas
        neurons = draw_layer_neurons(ax, layer_x, 2, 7, n_units // 32, color=color)
        
        ax.text(layer_x, 7.8, f'Hidden {layer_idx + 1}\n({n_units} units)',
               ha='center', fontsize=9, fontweight='bold')
        ax.text(layer_x, 1.3, 'ReLU\n+Dropout', ha='center',
               fontsize=8, style='italic', color='gray')
        
        # Conexiones desde capa anterior
        if layer_idx == 0:
            # Primera capa recibe desde attention
            for (x2, y2) in neurons[:3]:
                draw_arrow(ax, att_x + 0.8, att_y, layer_x - 0.3, y2,
                          color='purple', linewidth=0.8)
        else:
            # Conexiones densas entre capas
            if prev_neurons:
                for (x1, y1) in prev_neurons[:3]:
                    for (x2, y2) in neurons[:3]:
                        draw_connection(ax, x1 + 0.2, y1, x2 - 0.2, y2, alpha=0.15)
        
        prev_neurons = neurons
    
    # 4. Output Layer (3 clases)
    output_x = 14.5
    output_classes = ['Lector\nFrecuente', 'Lector\nMedio', 'No\nLector']
    output_colors = ['#27AE60', '#F39C12', '#E74C3C']
    output_y_positions = [6, 4, 2]
    
    for y_pos, class_name, color in zip(output_y_positions, output_classes, output_colors):
        draw_neuron(ax, output_x, y_pos, radius=0.25, color=color)
        ax.text(output_x + 0.7, y_pos, class_name, ha='left', va='center',
               fontsize=9, fontweight='bold')
        # Conexiones desde última hidden
        if prev_neurons:
            for (x1, y1) in prev_neurons[:3]:
                draw_connection(ax, x1 + 0.2, y1, output_x - 0.3, y_pos,
                              alpha=0.2, color='gray')
    
    ax.text(output_x, 7.3, 'Output\n(Softmax)', ha='center',
           fontsize=10, fontweight='bold')
    
    # Leyenda
    legend_y = 0.8
    ax.text(0.5, legend_y, '● Input Features', color='#4A90E2', fontsize=9, fontweight='bold')
    ax.text(3, legend_y, '□ Multi-head Attention', color='#C39BD3', fontsize=9, fontweight='bold')
    ax.text(6.5, legend_y, '● Hidden Layers', color='#FFD4A5', fontsize=9, fontweight='bold')
    ax.text(9.5, legend_y, '● Output Classes', fontsize=9, fontweight='bold')
    
    # Metadata
    ax.text(8, 0.3, 'Optimizer: AdamW | Loss: CrossEntropyLoss | Regularization: Dropout(0.3) + Weight Decay',
           ha='center', fontsize=8, style='italic', color='gray')
    
    plt.tight_layout()
    return fig

def main():
    """Genera las 3 visualizaciones"""
    import os
    
    # Crear directorio de salida
    output_dir = r'c:\Users\xdxdxdxd\OneDrive - Universidad Peruana Unión\Escritorio\MLP_Py_In01\outputs\architecture_diagrams'
    os.makedirs(output_dir, exist_ok=True)
    
    print("🎨 Generando visualizaciones de arquitecturas...")
    
    # 1. FT-Transformer
    print("\n1️⃣  Generando FT-Transformer...")
    fig1 = visualize_ft_transformer()
    output_path1 = os.path.join(output_dir, 'ft_transformer_architecture.png')
    fig1.savefig(output_path1, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"   ✅ Guardado: {output_path1}")
    plt.close(fig1)
    
    # 2. TabNet
    print("\n2️⃣  Generando TabNet...")
    fig2 = visualize_tabnet()
    output_path2 = os.path.join(output_dir, 'tabnet_architecture.png')
    fig2.savefig(output_path2, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"   ✅ Guardado: {output_path2}")
    plt.close(fig2)
    
    # 3. MLP + Attention
    print("\n3️⃣  Generando MLP + Multi-head Attention...")
    fig3 = visualize_mlp_attention()
    output_path3 = os.path.join(output_dir, 'mlp_attention_architecture.png')
    fig3.savefig(output_path3, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"   ✅ Guardado: {output_path3}")
    plt.close(fig3)
    
    print(f"\n🎉 ¡Todas las visualizaciones generadas exitosamente!")
    print(f"📁 Ubicación: {output_dir}")

if __name__ == "__main__":
    main()
