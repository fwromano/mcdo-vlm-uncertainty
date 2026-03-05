"""Metric Ranking: horizontal bar chart of 10 metrics by blur_r5 validity.
Shows weighted_trace_pre as the clear winner.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Data from STATE_OF_EXPLORATION Sec 2
metrics = [
    ('weighted_trace_pre', 96.4),
    ('topk64_trace_pre', 95.4),
    ('trace_pre', 93.6),
    ('trace_post', 93.2),
    ('mean_cosine_dev', 93.2),
    ('top_eigenvalue', 87.0),
    ('max_dim_var', 85.0),
    ('norm_var', 80.4),
    ('pcs_shift', 72.8),
    ('top1_ratio', 51.6),
]

names = [m[0] for m in reversed(metrics)]
vals = [m[1] for m in reversed(metrics)]

fig, ax = plt.subplots(figsize=(9, 5.5))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

colors = ['black' if v >= 75 else '#BBBBBB' for v in vals]
bars = ax.barh(range(len(names)), vals, color=colors, edgecolor='#333333', linewidth=0.5, height=0.65)

# Validity threshold
ax.axvline(x=75, color='#888888', linestyle='--', linewidth=1)
ax.text(75.5, len(names) - 0.5, '75%', fontsize=8, color='#888888')

# Value labels
for i, (bar, v) in enumerate(zip(bars, vals)):
    ax.text(bar.get_width() + 0.8, bar.get_y() + bar.get_height()/2,
            f'{v}%', va='center', fontsize=9)

ax.set_yticks(range(len(names)))
ax.set_yticklabels(names, fontsize=9, fontfamily='monospace')
ax.set_xlabel('Validity (blur_r5 ablation pass %)', fontsize=11)
ax.set_title('Uncertainty Metric Ranking', fontsize=14, fontweight='bold')
ax.set_xlim(0, 105)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#333333')
ax.spines['bottom'].set_color('#333333')
ax.tick_params(colors='#333333', labelsize=9)

plt.tight_layout()
plt.savefig('/sessions/confident-hopeful-noether/charts/chart_metric_ranking.png', dpi=200, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print("Saved chart_metric_ranking.png")
