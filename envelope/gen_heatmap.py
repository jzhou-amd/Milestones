import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as ticker

# --- Data ---
s_range = np.logspace(0, 6, 800)   # 1 to 1M tokens
c_range = np.logspace(0, 7, 900)   # 1 to 10M tokens
S, C = np.meshgrid(s_range, c_range)

attn = 0.00718 * (S**2 + S * C)
kv   = 2.28 * C + 3.67 * S
ratio = attn / kv

# --- Custom diverging colormap: teal -> white -> red-orange ---
colors_list = [
    (0.00, '#0a4046'),   # deep teal
    (0.20, '#1e7a82'),   # teal
    (0.35, '#4ecdc4'),   # bright teal
    (0.50, '#f5f5f0'),   # warm white
    (0.65, '#ff9650'),   # orange
    (0.80, '#ff5a3c'),   # red-orange
    (1.00, '#8c1414'),   # deep red
]
cmap = mcolors.LinearSegmentedColormap.from_list(
    'attn_kv',
    [(t, c) for t, c in colors_list],
    N=512
)

# --- Figure ---
fig, ax = plt.subplots(figsize=(9.5, 7.5), dpi=200)
fig.patch.set_facecolor('#ffffff')
ax.set_facecolor('#fafafa')

# Log ratio for symmetric color mapping
log_ratio = np.log10(ratio)
vmin, vmax = -3, 3

im = ax.pcolormesh(
    S, C, log_ratio,
    cmap=cmap, vmin=vmin, vmax=vmax,
    shading='gouraud', rasterized=True
)

# Contour at ratio = 1
contour = ax.contour(
    S, C, log_ratio, levels=[0],
    colors='#222222', linewidths=1.6, linestyles='--'
)
ax.clabel(contour, fmt={0: 'ratio = 1'}, fontsize=9,
          manual=[(450, 3e6)],
          colors='#222222')

# Log scale
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(1, 1e6)
ax.set_ylim(1, 1e7)

# Axis labels
ax.set_xlabel('s  (new prompt tokens)', fontsize=12, fontfamily='monospace',
              color='#333333', labelpad=10)
ax.set_ylabel('c  (shared context tokens)', fontsize=12, fontfamily='monospace',
              color='#333333', labelpad=10)

# Tick styling
ax.tick_params(colors='#555555', which='both', labelsize=10)
for spine in ax.spines.values():
    spine.set_color('#cccccc')

ax.grid(True, which='major', color='#e0e0e0', linewidth=0.5, alpha=0.7)
ax.grid(True, which='minor', color='#f0f0f0', linewidth=0.3, alpha=0.5)

# Colorbar
cbar = fig.colorbar(im, ax=ax, pad=0.02, aspect=30, shrink=0.92)
cbar.set_label('attention time / KV$ creation time', fontsize=10,
               fontfamily='monospace', color='#444444', labelpad=12)

# Custom colorbar ticks showing actual ratio values
cbar_ticks = [-3, -2, -1, 0, 1, 2, 3]
cbar_labels = ['0.001x', '0.01x', '0.1x', '1x', '10x', '100x', '1000x']
cbar.set_ticks(cbar_ticks)
cbar.set_ticklabels(cbar_labels)
cbar.ax.tick_params(labelsize=9, colors='#555555')
cbar.outline.set_edgecolor('#cccccc')

# Title
ax.set_title('KV$ Sharing: When Does Attention Dominate Runtime?',
             fontsize=14, fontfamily='monospace', color='#222222',
             pad=16, fontweight='500')

# Equations annotation
eq_text = (
    'attention = 0.00718 × (s² + s·c)\n'
    'KV$ creation = 2.28 × c + 3.67 × s'
)
ax.annotate(eq_text, xy=(0.02, 0.02), xycoords='axes fraction',
            fontsize=9, fontfamily='monospace', color='#666666',
            verticalalignment='bottom',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                      edgecolor='#dddddd', alpha=0.9))

# Region labels
ax.text(5, 3e6, 'KV$ creation\ndominates', fontsize=11, color='#0a6066',
        fontfamily='monospace', fontweight='bold', alpha=0.7,
        ha='left', va='center')
ax.text(2e4, 30, 'Attention\ndominates', fontsize=11, color='#8c1414',
        fontfamily='monospace', fontweight='bold', alpha=0.7,
        ha='center', va='center')

plt.tight_layout()
plt.savefig('/sessions/pensive-eloquent-hamilton/mnt/outputs/kv-cache-heatmap.png',
            dpi=200, bbox_inches='tight', facecolor='#ffffff')
print('saved')
