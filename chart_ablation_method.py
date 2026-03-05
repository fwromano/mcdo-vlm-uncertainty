"""Ablation test methodology diagram.
Shows: original image -> MC dropout -> uncertainty score
       degraded image -> MC dropout -> uncertainty score (should be higher)
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

fig, ax = plt.subplots(figsize=(10, 4.5))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')
ax.set_xlim(0, 10)
ax.set_ylim(0, 4.5)
ax.axis('off')

# Style constants
BOX_COLOR = '#F5F5F5'
EDGE_COLOR = '#333333'
ARROW_COLOR = '#666666'
FONT = {'fontsize': 10, 'fontfamily': 'sans-serif', 'color': '#222222'}
FONT_SMALL = {'fontsize': 8.5, 'fontfamily': 'sans-serif', 'color': '#666666'}
FONT_BOLD = {'fontsize': 10, 'fontfamily': 'sans-serif', 'fontweight': 'bold', 'color': '#222222'}
FONT_LABEL = {'fontsize': 11, 'fontfamily': 'sans-serif', 'fontweight': 'bold', 'color': '#000000'}

# ─── TOP ROW: Original path ───
# Original image box
ax.add_patch(FancyBboxPatch((0.3, 3.0), 1.8, 1.0, boxstyle="round,pad=0.1",
             facecolor=BOX_COLOR, edgecolor=EDGE_COLOR, linewidth=1.2))
ax.text(1.2, 3.5, 'Original\nImage', ha='center', va='center', **FONT_BOLD)

# Arrow
ax.annotate('', xy=(2.7, 3.5), xytext=(2.2, 3.5),
            arrowprops=dict(arrowstyle='->', color=ARROW_COLOR, lw=1.5))

# MC Dropout box (shared visually)
ax.add_patch(FancyBboxPatch((2.8, 2.85), 2.4, 1.3, boxstyle="round,pad=0.1",
             facecolor='white', edgecolor=EDGE_COLOR, linewidth=1.5))
ax.text(4.0, 3.75, 'Frozen VLM Encoder', ha='center', va='center', **FONT_BOLD)
ax.text(4.0, 3.3, 'T forward passes', ha='center', va='center', **FONT_SMALL)
ax.text(4.0, 2.95, 'with random dropout masks', ha='center', va='center', **FONT_SMALL)

# Arrow to variance
ax.annotate('', xy=(5.8, 3.5), xytext=(5.3, 3.5),
            arrowprops=dict(arrowstyle='->', color=ARROW_COLOR, lw=1.5))

# Variance computation box
ax.add_patch(FancyBboxPatch((5.9, 3.0), 1.8, 1.0, boxstyle="round,pad=0.1",
             facecolor=BOX_COLOR, edgecolor=EDGE_COLOR, linewidth=1.2))
ax.text(6.8, 3.55, 'Compute', ha='center', va='center', **FONT_BOLD)
ax.text(6.8, 3.15, 'Uncertainty', ha='center', va='center', **FONT_BOLD)

# Arrow to score
ax.annotate('', xy=(8.3, 3.5), xytext=(7.8, 3.5),
            arrowprops=dict(arrowstyle='->', color=ARROW_COLOR, lw=1.5))

# Score box - original
ax.add_patch(FancyBboxPatch((8.4, 3.0), 1.3, 1.0, boxstyle="round,pad=0.1",
             facecolor='white', edgecolor=EDGE_COLOR, linewidth=1.2))
ax.text(9.05, 3.55, 'U = 0.42', ha='center', va='center', **FONT_BOLD)
ax.text(9.05, 3.15, '(lower)', ha='center', va='center', **FONT_SMALL)

# ─── BOTTOM ROW: Degraded path ───
# Degraded image box
ax.add_patch(FancyBboxPatch((0.3, 0.7), 1.8, 1.0, boxstyle="round,pad=0.1",
             facecolor=BOX_COLOR, edgecolor=EDGE_COLOR, linewidth=1.2))
ax.text(1.2, 1.2, 'Degraded\nImage', ha='center', va='center', **FONT_BOLD)
ax.text(1.2, 0.82, '(blur / downsample)', ha='center', va='center',
        fontsize=7.5, fontfamily='sans-serif', color='#888888')

# Arrow
ax.annotate('', xy=(2.7, 1.2), xytext=(2.2, 1.2),
            arrowprops=dict(arrowstyle='->', color=ARROW_COLOR, lw=1.5))

# MC Dropout box (bottom)
ax.add_patch(FancyBboxPatch((2.8, 0.55), 2.4, 1.3, boxstyle="round,pad=0.1",
             facecolor='white', edgecolor=EDGE_COLOR, linewidth=1.5))
ax.text(4.0, 1.45, 'Same Encoder', ha='center', va='center', **FONT_BOLD)
ax.text(4.0, 1.0, 'Same T passes', ha='center', va='center', **FONT_SMALL)
ax.text(4.0, 0.65, 'Same dropout masks', ha='center', va='center', **FONT_SMALL)

# Arrow to variance
ax.annotate('', xy=(5.8, 1.2), xytext=(5.3, 1.2),
            arrowprops=dict(arrowstyle='->', color=ARROW_COLOR, lw=1.5))

# Variance computation box
ax.add_patch(FancyBboxPatch((5.9, 0.7), 1.8, 1.0, boxstyle="round,pad=0.1",
             facecolor=BOX_COLOR, edgecolor=EDGE_COLOR, linewidth=1.2))
ax.text(6.8, 1.25, 'Compute', ha='center', va='center', **FONT_BOLD)
ax.text(6.8, 0.85, 'Uncertainty', ha='center', va='center', **FONT_BOLD)

# Arrow to score
ax.annotate('', xy=(8.3, 1.2), xytext=(7.8, 1.2),
            arrowprops=dict(arrowstyle='->', color=ARROW_COLOR, lw=1.5))

# Score box - degraded
ax.add_patch(FancyBboxPatch((8.4, 0.7), 1.3, 1.0, boxstyle="round,pad=0.1",
             facecolor='#E8E8E8', edgecolor=EDGE_COLOR, linewidth=1.5))
ax.text(9.05, 1.25, 'U = 0.61', ha='center', va='center', **FONT_BOLD)
ax.text(9.05, 0.85, '(higher)', ha='center', va='center',
        fontsize=8.5, fontfamily='sans-serif', fontweight='bold', color='#000000')

# ─── COMPARISON BRACKET ───
# Vertical comparison arrow between scores
ax.annotate('', xy=(9.05, 3.0), xytext=(9.05, 1.7),
            arrowprops=dict(arrowstyle='<->', color='black', lw=1.8))
ax.text(9.55, 2.35, 'VALID if\nhigher', ha='left', va='center',
        fontsize=9, fontfamily='sans-serif', fontweight='bold', color='#000000')

# Title
ax.text(5.0, 4.35, 'The Ablation Test: Does Uncertainty Track Image Quality?',
        ha='center', va='center', fontsize=13, fontfamily='sans-serif',
        fontweight='bold', color='#000000')

plt.tight_layout()
plt.savefig('/sessions/confident-hopeful-noether/charts/chart_ablation_method.png', dpi=200,
            bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close()
print("Saved chart_ablation_method.png")
