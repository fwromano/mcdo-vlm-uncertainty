"""Module Scan: which module types carry signal.
Horizontal bar chart of Spearman reliability by dropout strategy.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Data from STATE_OF_EXPLORATION Exp 3
strategies = [
    ('Attention out_proj', 0.000),
    ('Stochastic depth', 0.194),
    ('Uniform all 36', 0.518),
    ('MLP (all)', 0.525),
    ('All 12 c_proj', 0.430),
    ('Single best c_proj', 0.771),
]

names = [s[0] for s in reversed(strategies)]
vals = [s[1] for s in reversed(strategies)]

fig, ax = plt.subplots(figsize=(8, 4.5))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

colors = []
for v, n in zip(vals, names):
    if v == 0:
        colors.append('#DDDDDD')
    elif 'Single' in n:
        colors.append('black')
    else:
        colors.append('#888888')

bars = ax.barh(range(len(names)), vals, color=colors, edgecolor='#333333', linewidth=0.5, height=0.6)

for i, (bar, v) in enumerate(zip(bars, vals)):
    label = f'{v:.3f}' if v > 0 else 'ZERO'
    ax.text(max(bar.get_width(), 0.01) + 0.015, bar.get_y() + bar.get_height()/2,
            label, va='center', fontsize=9, fontweight='bold' if 'ZERO' in label else 'normal')

ax.set_yticks(range(len(names)))
ax.set_yticklabels(names, fontsize=10)
ax.set_xlabel('Spearman Rank Correlation (Reliability)', fontsize=11)
ax.set_title('Module Scan: Where Does Signal Live?', fontsize=14, fontweight='bold')
ax.set_xlim(0, 0.95)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#333333')
ax.spines['bottom'].set_color('#333333')
ax.tick_params(colors='#333333', labelsize=9)

plt.tight_layout()
plt.savefig('/sessions/confident-hopeful-noether/charts/chart_module_scan.png', dpi=200, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print("Saved chart_module_scan.png")
