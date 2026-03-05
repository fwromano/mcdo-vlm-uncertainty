"""Architecture diagram: Transformer block anatomy showing where dropout is applied.
Shows the processing pipeline from image to uncertainty score.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')
ax.set_xlim(0, 10)
ax.set_ylim(0, 6)
ax.axis('off')

FB = {'fontsize': 10, 'fontfamily': 'sans-serif', 'fontweight': 'bold', 'color': '#000'}
FN = {'fontsize': 9, 'fontfamily': 'sans-serif', 'color': '#333'}
FS = {'fontsize': 8, 'fontfamily': 'sans-serif', 'color': '#777'}
FM = {'fontsize': 8.5, 'fontfamily': 'monospace', 'color': '#333'}

# ─── LEFT: Transformer Block ───
ax.add_patch(FancyBboxPatch((0.3, 0.5), 5.4, 5.0, boxstyle="round,pad=0.15",
             facecolor='#FAFAFA', edgecolor='#333', linewidth=1.5))
ax.text(3.0, 5.25, 'Transformer Block (x12 for B/32)', ha='center', **FB)

# Self-Attention section
ax.add_patch(FancyBboxPatch((0.6, 3.7), 4.8, 1.3, boxstyle="round,pad=0.1",
             facecolor='#F0F0F0', edgecolor='#BBB', linewidth=1))
ax.text(0.9, 4.7, 'Self-Attention', **FB)
ax.text(0.9, 4.35, 'out_proj: 768 -> 768', **FM)
ax.text(3.5, 4.35, 'ZERO VARIANCE', fontsize=9, fontfamily='sans-serif',
        fontweight='bold', color='#999')
ax.text(0.9, 4.0, 'Softmax saturation + residual cancellation', **FS)

# MLP section
ax.add_patch(FancyBboxPatch((0.6, 0.8), 4.8, 2.5, boxstyle="round,pad=0.1",
             facecolor='white', edgecolor='#333', linewidth=1.2))
ax.text(0.9, 3.0, 'Feed-Forward Network (MLP)', **FB)

# c_fc
ax.add_patch(FancyBboxPatch((0.9, 2.3), 2.2, 0.45, boxstyle="round,pad=0.05",
             facecolor='#F5F5F5', edgecolor='#AAA', linewidth=0.8))
ax.text(1.0, 2.45, 'c_fc: 768 -> 3072', **FM)
ax.text(3.3, 2.45, 'expand', **FS)

# GELU
ax.text(2.0, 2.05, 'GELU activation', **FS)

# c_proj — highlighted
ax.add_patch(FancyBboxPatch((0.9, 1.2), 2.2, 0.55, boxstyle="round,pad=0.05",
             facecolor='#E8E8E8', edgecolor='black', linewidth=1.5))
ax.text(1.0, 1.4, 'c_proj: 3072 -> 768', fontsize=8.5, fontfamily='monospace',
        fontweight='bold', color='#000')
ax.text(3.3, 1.4, 'compress', **FS)

# Dropout callout
ax.annotate('DROPOUT HERE\np = 0.01', xy=(3.1, 1.45), xytext=(4.0, 1.1),
            fontsize=9, fontfamily='sans-serif', fontweight='bold', color='black',
            ha='left',
            arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

# ─── RIGHT: Pipeline ───
ax.text(7.5, 5.6, 'Pipeline', ha='center', **FB)

steps = [
    ('Image (224x224)', 5.15),
    ('Vision Encoder\n(frozen weights)', 4.4),
    ('T forward passes\n(different dropout masks)', 3.5),
    ('T x 512 embeddings', 2.7),
    ('Per-dim variance,\nweighted by\ndiscriminative power', 1.8),
    ('Scalar uncertainty\nscore per image', 0.7),
]

for i, (label, y) in enumerate(steps):
    bx = 6.2
    bw = 2.8
    bh = 0.5 if '\n' not in label else 0.7
    ax.add_patch(FancyBboxPatch((bx, y - bh/2), bw, bh, boxstyle="round,pad=0.08",
                 facecolor='white' if i < 4 else '#E8E8E8',
                 edgecolor='#333', linewidth=0.8))
    ax.text(bx + bw/2, y, label, ha='center', va='center',
            fontsize=8 if '\n' in label else 9,
            fontfamily='sans-serif', color='#222')
    if i < len(steps) - 1:
        next_y = steps[i+1][1]
        next_bh = 0.5 if '\n' not in steps[i+1][0] else 0.7
        ax.annotate('', xy=(7.6, next_y + next_bh/2 + 0.03),
                    xytext=(7.6, y - bh/2 - 0.03),
                    arrowprops=dict(arrowstyle='->', color='#666', lw=1.2))

plt.tight_layout()
plt.savefig('/sessions/confident-hopeful-noether/charts/chart_architecture.png', dpi=200,
            bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close()
print("Saved chart_architecture.png")
