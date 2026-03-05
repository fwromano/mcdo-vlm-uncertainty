"""Cross-Model Validation: 3 models on reliability-validity axes.
Confirms the theory generalizes across contrastive VLMs.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Data from STATE_OF_EXPLORATION
models = [
    ('CLIP B/32', 0.430, 93.6, 'o'),
    ('CLIP L/14', 0.750, 78.2, 's'),
    ('PE-Core B/16', 0.820, 94.4, 'D'),
]

fig, ax = plt.subplots(figsize=(7, 5))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

# Validity threshold
ax.axhline(y=75, color='#888888', linestyle='--', linewidth=1)
ax.text(0.38, 76, '75% threshold', fontsize=8, color='#888888')

for name, rel, val, marker in models:
    ax.scatter(rel, val, s=120, c='black', marker=marker, zorder=3, linewidths=1.5)
    ax.annotate(name, (rel, val), xytext=(10, -5), textcoords='offset points',
                fontsize=10, fontweight='bold')

# Annotations for key insight
ax.annotate('Larger model =\nmore robust features', xy=(0.75, 78.2), xytext=(0.55, 65),
            fontsize=8, color='#666666', ha='center',
            arrowprops=dict(arrowstyle='->', color='#999999', lw=1))

ax.set_xlabel('Reliability (Spearman)', fontsize=11)
ax.set_ylabel('Validity (ablation pass %)', fontsize=11)
ax.set_title('Cross-Model Validation', fontsize=14, fontweight='bold')

ax.set_xlim(0.35, 0.95)
ax.set_ylim(55, 105)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#333333')
ax.spines['bottom'].set_color('#333333')
ax.tick_params(colors='#333333', labelsize=9)
ax.grid(axis='both', alpha=0.15, color='#000000')

plt.tight_layout()
plt.savefig('/sessions/confident-hopeful-noether/charts/chart_cross_model.png', dpi=200, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print("Saved chart_cross_model.png")
