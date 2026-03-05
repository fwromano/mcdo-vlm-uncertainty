"""T-Scaling: Reliability scales as O(sqrt(T)).
Shows Spearman correlation vs number of MC passes for two configs.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

T = [16, 32, 64, 128, 256]
spearman_cproj = [0.307, 0.311, 0.430, 0.581, 0.739]
spearman_uniform = [0.433, 0.446, 0.574, 0.705, 0.821]

fig, ax = plt.subplots(figsize=(8, 5.5))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

ax.plot(T, spearman_cproj, 'o-', color='black', linewidth=1.8, markersize=6, label='12-c_proj (high validity)')
ax.plot(T, spearman_uniform, 's--', color='#666666', linewidth=1.8, markersize=6, label='Uniform (baseline)')

# sqrt(T) reference curve, fitted to uniform at T=64
ref_T = np.linspace(16, 256, 100)
scale = spearman_uniform[2] / np.sqrt(64)
ax.plot(ref_T, scale * np.sqrt(ref_T), ':', color='#AAAAAA', linewidth=1.2, label=r'$O(\sqrt{T})$ reference')

ax.set_xlabel('MC Passes (T)', fontsize=11)
ax.set_ylabel('Spearman Rank Correlation', fontsize=11)
ax.set_title('Reliability Scaling with MC Passes', fontsize=14, fontweight='bold')

ax.set_xscale('log', base=2)
ax.set_xticks(T)
ax.set_xticklabels([str(t) for t in T])
ax.set_ylim(0.2, 0.9)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#333333')
ax.spines['bottom'].set_color('#333333')
ax.tick_params(colors='#333333', labelsize=9)
ax.grid(axis='both', alpha=0.15, color='#000000')
ax.legend(fontsize=9, frameon=False)

# Cost annotation
for i, t in enumerate(T):
    cost = t / 64
    ax.annotate(f'{cost:.1f}x' if cost != 1 else '1x', (t, spearman_cproj[i]),
                xytext=(0, -18), textcoords='offset points', fontsize=7.5,
                ha='center', color='#666666')

ax.text(0.98, 0.02, 'Labels: relative cost (1x = T=64)', transform=ax.transAxes,
        fontsize=7.5, ha='right', color='#999999')

plt.tight_layout()
plt.savefig('/sessions/confident-hopeful-noether/charts/chart_t_scaling.png', dpi=200, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print("Saved chart_t_scaling.png")
