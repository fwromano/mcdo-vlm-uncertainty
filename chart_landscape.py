"""Reliability-Validity Landscape scatter plot.
Shows the fundamental tradeoff between reliability (Spearman) and validity (ablation %).
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Data: (reliability_spearman, validity_pct, label)
points = [
    (0.430, 96.4, "CLIP B/32\n12-c_proj+wt"),
    (0.430, 93.6, "CLIP B/32\n12-c_proj"),
    (0.574, 86.8, "CLIP B/32\nuniform"),
    (0.750, 78.2, "CLIP L/14\nc_proj"),
    (0.821, 71.6, "CLIP L/14\nuniform"),
    (0.820, 94.4, "PE-Core\nlate-3+wt"),
    (0.977, 59.5, "Gaussian\nall modules"),
    (0.998, 25.4, "Gaussian\nblk11 only"),
]

fig, ax = plt.subplots(figsize=(10, 6.5))

# Style
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

# Validity threshold
ax.axhline(y=75, color='#888888', linestyle='--', linewidth=1, zorder=1)
ax.text(0.25, 76.5, 'Validity threshold (75%)', fontsize=9, color='#888888')

# Plot points
xs = [p[0] for p in points]
ys = [p[1] for p in points]
labels = [p[2] for p in points]

# Separate into above/below threshold for marker style
for x, y, label in points:
    marker = 'o' if y >= 75 else 'x'
    color = 'black' if y >= 75 else '#999999'
    size = 80 if y >= 75 else 60
    ax.scatter(x, y, s=size, c=color, marker=marker, zorder=3, linewidths=1.5)

    # Label offsets (hand-tuned)
    dx, dy = 0.02, 0
    ha = 'left'
    if 'PE-Core' in label:
        dx, dy = -0.02, 2
        ha = 'right'
    elif '12-c_proj+wt' in label:
        dx, dy = 0.02, 2
    elif '12-c_proj' in label and 'wt' not in label:
        dx, dy = 0.02, -4
    elif 'uniform' in label and 'B/32' in label:
        dx, dy = 0.02, 2
    elif 'L/14' in label and 'c_proj' in label and 'uniform' not in label:
        dx, dy = -0.02, -4
        ha = 'right'
    elif 'Gaussian' in label and 'all' in label:
        dx, dy = -0.02, 2
        ha = 'right'
    elif 'blk11' in label:
        dx, dy = -0.02, 0
        ha = 'right'

    ax.annotate(label, (x, y), xytext=(x + dx, y + dy),
                fontsize=8, color='#333333', ha=ha, va='center',
                fontfamily='monospace')

ax.set_xlabel('Reliability (Spearman rank correlation)', fontsize=11, fontfamily='sans-serif')
ax.set_ylabel('Validity (ablation pass %)', fontsize=11, fontfamily='sans-serif')
ax.set_title('Reliability–Validity Landscape', fontsize=14, fontweight='bold', fontfamily='sans-serif')

ax.set_xlim(0.2, 1.05)
ax.set_ylim(15, 105)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#333333')
ax.spines['bottom'].set_color('#333333')
ax.tick_params(colors='#333333', labelsize=9)
ax.grid(axis='both', alpha=0.15, color='#000000')

plt.tight_layout()
plt.savefig('/sessions/confident-hopeful-noether/charts/chart_landscape.png', dpi=200, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print("Saved chart_landscape.png")
