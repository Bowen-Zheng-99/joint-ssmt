#!/usr/bin/env python3
"""
Panel B: Joint inference schematic.
Large version for methods figure (~half a publication page: 3.5" × 4.5").
Spacious layout, minimal whitespace at borders, uniform font size.
"""

import os
import argparse
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import FancyBboxPatch

# Figure style
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Computer Modern Roman', 'CMU Serif', 'DejaVu Serif'],
    'mathtext.fontset': 'cm',
})
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# Colors
latent_color = '#7B2D8E'
lfp_color = '#2E86AB'
spike_color = '#E94F37'
pg_color = '#F5A623'
kalman_color = '#5D8A66'

# UNIFORM FONT SIZE
FONT_SIZE = 8


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, default='./final_figure_data/figures')
    parser.add_argument('--figsize_w', type=float, default=3.5)
    parser.add_argument('--figsize_h', type=float, default=4.5)
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    
    # Create figure with minimal margins
    fig, ax = plt.subplots(figsize=(args.figsize_w, args.figsize_h))
    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
    
    # Coordinate system: x=[0, 10], y=[0, 14]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 14)
    ax.axis('off')
    
    # =========================================================================
    # LATENT BOX (top, full width, taller)
    # =========================================================================
    latent_box = FancyBboxPatch((0.2, 11.2), 9.6, 2.6, boxstyle="round,pad=0.15",
                                 facecolor=latent_color, alpha=0.25,
                                 edgecolor=latent_color, linewidth=1.5)
    ax.add_patch(latent_box)
    ax.text(5.0, 13.2, 'Latent OU Process', fontsize=FONT_SIZE, ha='center', fontweight='bold')
    ax.text(5.0, 12.2, r'$dZ_t(\omega_j) = -\lambda_j Z_t \, dt + \sigma_{v,j} \, dB_t$',
            fontsize=FONT_SIZE, ha='center')

    
    # =========================================================================
    # LFP BOX (left middle, wider)
    # =========================================================================
    lfp_box = FancyBboxPatch((0.2, 6.3), 4.5, 3.2, boxstyle="round,pad=0.15",
                              facecolor=lfp_color, alpha=0.2,
                              edgecolor=lfp_color, linewidth=1.5)
    ax.add_patch(lfp_box)
    ax.text(2.45, 8.8, 'Multitaper Spectrogram', fontsize=FONT_SIZE, ha='center', fontweight='bold')
    ax.text(2.45, 7.8, r'$Y_k^{(m)} = Z_{t_k} + \varepsilon_k$', fontsize=FONT_SIZE, ha='center')
    ax.text(2.45, 6.9, r'(Gaussian)', fontsize=FONT_SIZE, ha='center', color='#555555')
    
    # Arrow latent -> LFP
    ax.annotate('', xy=(2.45, 9.4), xytext=(3.8, 11.3),
                arrowprops=dict(arrowstyle='->', color='#333333', lw=1.2,
                              connectionstyle='arc3,rad=0.12'))
    
    # =========================================================================
    # SPIKE BOX (right middle, wider)
    # =========================================================================
    spike_box = FancyBboxPatch((5.3, 6.3), 4.5, 3.2, boxstyle="round,pad=0.15",
                                facecolor=spike_color, alpha=0.2,
                                edgecolor=spike_color, linewidth=1.5)
    ax.add_patch(spike_box)
    ax.text(7.55, 8.8, 'Spike Train', fontsize=FONT_SIZE, ha='center', fontweight='bold')
    ax.text(7.55, 7.8, r'$S_n \sim \mathrm{Bern}(\sigma(\psi_n))$', fontsize=FONT_SIZE, ha='center')

    ax.text(7.55, 6.9, r'(Bernoulli)', fontsize=FONT_SIZE, ha='center', color='#555555')
    
    # Arrow latent -> Spike
    ax.annotate('', xy=(7.55, 9.4), xytext=(6.2, 11.3),
                arrowprops=dict(arrowstyle='->', color='#333333', lw=1.2,
                              connectionstyle='arc3,rad=-0.12'))
    
    # =========================================================================
    # POLYA-GAMMA BOX (right lower, wider)
    # =========================================================================
    pg_box = FancyBboxPatch((5.3, 2.8), 4.5, 2.8, boxstyle="round,pad=0.15",
                             facecolor=pg_color, alpha=0.25,
                             edgecolor=pg_color, linewidth=1.5)
    ax.add_patch(pg_box)
    ax.text(7.55, 5.0, 'Pólya-Gamma Augmentation', fontsize=FONT_SIZE, ha='center', fontweight='bold')
    ax.text(7.55, 4.0, r'$\xi_n \sim \mathrm{PG}(1, \psi_n)$', fontsize=FONT_SIZE, ha='center')
    ax.text(7.55, 3.15, r'$\rightarrow$ Gaussian pseudo-obs', fontsize=FONT_SIZE, ha='center', 
            color='#8B4000')
    
    # Arrow Spike -> PG
    ax.annotate('', xy=(7.55, 5.55), xytext=(7.55, 6.4),
                arrowprops=dict(arrowstyle='->', color='#333333', lw=1.2))
    
    # =========================================================================
    # KALMAN BOX (bottom, full width, taller)
    # =========================================================================
    kalman_box = FancyBboxPatch((0.2, 0.2), 9.6, 2.2, boxstyle="round,pad=0.15",
                                 facecolor=kalman_color, alpha=0.25,
                                 edgecolor=kalman_color, linewidth=1.5)
    ax.add_patch(kalman_box)
    ax.text(5.0, 1.8, 'Kalman (RTS) Smoother', fontsize=FONT_SIZE, ha='center', fontweight='bold')
    ax.text(5.0, 0.9, r'$p(Z_{1:T} \mid Y, S)$', fontsize=FONT_SIZE, ha='center')

    
    # Arrow LFP -> Kalman
    ax.annotate('', xy=(3.2, 2.3), xytext=(2.45, 6.4),
                arrowprops=dict(arrowstyle='->', color=lfp_color, lw=1.5,
                              connectionstyle='arc3,rad=0.06'))
    
    # Arrow PG -> Kalman
    ax.annotate('', xy=(6.8, 2.3), xytext=(7.55, 2.9),
                arrowprops=dict(arrowstyle='->', color='#8B4000', lw=1.5,
                              connectionstyle='arc3,rad=-0.06'))
    
    # Save with tight bounding box and minimal padding
    plt.savefig(os.path.join(args.output, 'panel_b_schematic.png'), dpi=150, 
                bbox_inches='tight', pad_inches=0.02)
    plt.savefig(os.path.join(args.output, 'panel_b_schematic.pdf'), 
                bbox_inches='tight', pad_inches=0.02)
    plt.close()
    print(f"Saved panel_b_schematic.png/pdf to {args.output}")


if __name__ == '__main__':
    main()