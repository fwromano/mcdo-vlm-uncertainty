"""Metric taxonomy: shows the two axes (reliability vs validity)
and the key uncertainty metrics with brief definitions.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(figsize=(10, 5.5))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')
ax.set_xlim(0, 10)
ax.set_ylim(0, 5.5)
ax.axis('off')

FONT_TITLE = {'fontsize': 12, 'fontfamily': 'sans-serif', 'fontweight': 'bold', 'color': '#000000'}
FONT_HEAD = {'fontsize': 10.5, 'fontfamily': 'sans-serif', 'fontweight': 'bold', 'color': '#000000'}
FONT_BODY = {'fontsize': 9, 'fontfamily': 'sans-serif', 'color': '#333333'}
FONT_SMALL = {'fontsize': 8, 'fontfamily': 'sans-serif', 'color': '#777777'}

# ─── LEFT COLUMN: Quality Axes ───
ax.text(2.3, 5.2, 'How We Measure Quality', ha='center', va='center', **FONT_TITLE)

# Reliability box
ax.add_patch(FancyBboxPatch((0.2, 3.4), 4.2, 1.5, boxstyle="round,pad=0.15",
             facecolor='#F8F8F8', edgecolor='#333333', linewidth=1.2))
ax.text(0.5, 4.6, 'RELIABILITY', **FONT_HEAD)
ax.text(0.5, 4.25, '"Do we get the same ranking if we re-measure?"', **FONT_BODY)
ax.text(0.5, 3.85, 'Spearman rho  —  rank correlation between', **FONT_BODY)
ax.text(0.5, 3.55, 'repeated MC dropout runs on same images', **FONT_BODY)

# Validity box
ax.add_patch(FancyBboxPatch((0.2, 1.5), 4.2, 1.5, boxstyle="round,pad=0.15",
             facecolor='#F8F8F8', edgecolor='#333333', linewidth=1.2))
ax.text(0.5, 2.7, 'VALIDITY', **FONT_HEAD)
ax.text(0.5, 2.35, '"Does uncertainty mean what we think it means?"', **FONT_BODY)
ax.text(0.5, 1.95, 'Ablation pass %  —  fraction of images where', **FONT_BODY)
ax.text(0.5, 1.65, 'degraded version has higher uncertainty', **FONT_BODY)

# Key insight
ax.add_patch(FancyBboxPatch((0.2, 0.15), 4.2, 0.95, boxstyle="round,pad=0.1",
             facecolor='white', edgecolor='#999999', linewidth=0.8, linestyle='--'))
ax.text(2.3, 0.75, 'Both axes required. High reliability with', ha='center', **FONT_BODY)
ax.text(2.3, 0.4, 'low validity = precise but wrong.', ha='center', **FONT_BODY)

# ─── RIGHT COLUMN: Uncertainty Metrics ───
ax.text(7.5, 5.2, 'What We Compute per Image', ha='center', va='center', **FONT_TITLE)

metrics = [
    ('weighted_trace_pre', '96.4%', 'Trace of covariance, weighted by\ndiscriminative dimension importance'),
    ('trace_pre', '93.6%', 'Sum of per-dimension variances\nacross T forward passes'),
    ('mean_cosine_dev', '93.2%', 'Average angular deviation of each\npass from the mean embedding'),
    ('top_eigenvalue', '87.0%', 'Largest eigenvalue of the\nT x T covariance matrix'),
    ('norm_var', '80.4%', 'Variance of the L2 norm\nacross T passes'),
]

y = 4.65
for name, val, desc in metrics:
    # Metric name
    ax.text(5.2, y, name, fontsize=9, fontfamily='monospace', fontweight='bold', color='#000000')
    # Validity badge
    ax.text(8.8, y, val, fontsize=9, fontfamily='sans-serif', fontweight='bold',
            color='black' if float(val[:-1]) >= 90 else '#888888', ha='center')
    # Description
    lines = desc.split('\n')
    for i, line in enumerate(lines):
        ax.text(5.2, y - 0.22 - i * 0.2, line, **FONT_SMALL)
    y -= 0.85

# Column header
ax.text(8.8, 5.0, 'Validity', fontsize=8, fontfamily='sans-serif', color='#999999', ha='center')

# Separator
ax.plot([4.7, 4.7], [0.3, 4.9], color='#DDDDDD', linewidth=1)

plt.tight_layout()
plt.savefig('/sessions/confident-hopeful-noether/charts/chart_metric_taxonomy.png', dpi=200,
            bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close()
print("Saved chart_metric_taxonomy.png")
