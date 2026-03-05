"""Dropout Rate Cliff: validity collapses above p=0.01.
Shows the narrow operating window for valid MC dropout.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

p_rates = [0.01, 0.05, 0.10, 0.20]
validity = [86.8, 43.5, 13.0, 0.5]

fig, ax = plt.subplots(figsize=(7, 5))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

bars = ax.bar(range(len(p_rates)), validity, color=['black', '#888888', '#BBBBBB', '#DDDDDD'],
              edgecolor='black', linewidth=0.8, width=0.6)

# Validity threshold
ax.axhline(y=75, color='#888888', linestyle='--', linewidth=1)
ax.text(3.3, 76.5, '75%', fontsize=8, color='#888888')

# Value labels
for i, (bar, v) in enumerate(zip(bars, validity)):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
            f'{v}%', ha='center', fontsize=10, fontweight='bold')

ax.set_xticks(range(len(p_rates)))
ax.set_xticklabels([f'p={p}' for p in p_rates], fontsize=10)
ax.set_ylabel('Validity (ablation pass %)', fontsize=11)
ax.set_title('Dropout Rate: Sharp Validity Cliff', fontsize=14, fontweight='bold')
ax.set_ylim(0, 105)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#333333')
ax.spines['bottom'].set_color('#333333')
ax.tick_params(colors='#333333', labelsize=9)

plt.tight_layout()
plt.savefig('/sessions/confident-hopeful-noether/charts/chart_dropout_cliff.png', dpi=200, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print("Saved chart_dropout_cliff.png")
