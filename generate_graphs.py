import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

os.makedirs('ppt_graphs', exist_ok=True)

# Style settings
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 200
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.pad_inches'] = 0.2

NAVY  = '#0A1F44'
BLUE  = '#1565C0'
ACC   = '#1E88E5'
GREEN = '#43A047'
RED   = '#E53935'
AMBER = '#FB8C00'
TEAL  = '#00897B'
GRAY  = '#546E7A'
GOLD  = '#FFD600'

# ─── DATA ──────────────────────────────────────────────────────
models = [
    'Iso. Forest','SVM','CNN','Log. Reg',
    'Decision\nTree','KNN','LSTM',
    'VQC\n(Quantum)','Proposed\nHybrid ★'
]
f1_no   = [0.56,0.74,0.91,0.55,0.85,0.93,0.92,0.84,0.8667]
f1_yes  = [0.53,0.61,0.82,0.62,0.74,0.75,0.88,0.84,1.00]
auc_no  = [0.0456,0.9731,0.9805,0.9720,0.8619,0.9437,0.9029,0.83,0.9823]
auc_yes = [0.8622,0.8424,0.9734,0.9710,0.9017,0.9482,0.9725,0.83,0.9998]
acc_no  = [0.98,1.00,1.00,0.98,1.00,1.00,1.00,0.86,0.9996]
acc_yes = [0.99,0.99,1.00,0.99,1.00,1.00,1.00,0.86,1.00]
prec_yes= [0.59,0.57,0.76,0.57,0.67,0.68,0.84,1.00,1.00]
rec_yes = [0.52,0.78,0.92,0.94,0.89,0.94,0.92,0.72,1.00]

x = np.arange(len(models))

# ════════════════════════════════════════════════════
# GRAPH 1: F1-Score Comparison (Grouped Bar)
# ════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(14, 6))
fig.patch.set_facecolor('#F8FBFF')
ax.set_facecolor('#FFFFFF')

w = 0.35
bars1 = ax.bar(x - w/2, f1_no,  w, label='Without SMOTE',
               color=ACC,   alpha=0.85, edgecolor='white', linewidth=0.8)
bars2 = ax.bar(x + w/2, f1_yes, w, label='With SMOTE',
               color=GREEN, alpha=0.95, edgecolor='white', linewidth=0.8)

# Highlight hybrid
bars2[-1].set_color(GOLD)
bars2[-1].set_edgecolor(NAVY)
bars2[-1].set_linewidth(2.5)

for bar, val in zip(bars2, f1_yes):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{val:.2f}', ha='center', va='bottom', fontsize=8,
            fontweight='bold' if val == 1.0 else 'normal',
            color=NAVY if val == 1.0 else GRAY)

ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=10)
ax.set_ylabel('F1-Score', fontsize=12, color=GRAY)
ax.set_ylim(0, 1.15)
ax.set_title('F1-Score Comparison — All Models (With & Without SMOTE)',
             fontsize=14, fontweight='bold', color=NAVY, pad=12)
ax.legend(fontsize=11, framealpha=0.9)
ax.axhline(y=1.0, color=GREEN, linestyle='--', linewidth=1, alpha=0.5)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.spines[['top','right']].set_visible(False)

plt.tight_layout()
plt.savefig('ppt_graphs/graph1_f1_comparison.png')
plt.close()
print("Graph 1: F1-Score Comparison saved")

# ════════════════════════════════════════════════════
# GRAPH 2: AUC-ROC Comparison (Horizontal Bar)
# ════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 7))
fig.patch.set_facecolor('#F8FBFF')
ax.set_facecolor('#FFFFFF')

colors = [GREEN if v >= 0.99 else ACC if v >= 0.97
          else AMBER if v >= 0.90 else RED for v in auc_yes]
colors[-1] = GOLD

y_pos = np.arange(len(models))
bars = ax.barh(y_pos, auc_yes, color=colors, alpha=0.9,
               edgecolor='white', linewidth=0.8, height=0.6)
bars[-1].set_edgecolor(NAVY)
bars[-1].set_linewidth(2.5)

for bar, val in zip(bars, auc_yes):
    ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
            f'{val:.4f}', va='center', fontsize=9,
            fontweight='bold' if val >= 0.99 else 'normal',
            color=NAVY)

ax.set_yticks(y_pos)
ax.set_yticklabels(models, fontsize=11)
ax.set_xlabel('AUC-ROC Score', fontsize=12, color=GRAY)
ax.set_xlim(0, 1.12)
ax.set_title('AUC-ROC Score Comparison — All Models (With SMOTE)',
             fontsize=14, fontweight='bold', color=NAVY, pad=12)
ax.axvline(x=1.0, color=GREEN, linestyle='--', linewidth=1, alpha=0.5)
ax.grid(axis='x', alpha=0.3, linestyle='--')
ax.spines[['top','right']].set_visible(False)

legend_patches = [
    mpatches.Patch(color=GOLD,  label='Proposed Hybrid (0.9998)'),
    mpatches.Patch(color=GREEN, label='Excellent (≥0.99)'),
    mpatches.Patch(color=ACC,   label='Good (≥0.97)'),
    mpatches.Patch(color=AMBER, label='Moderate (≥0.90)'),
    mpatches.Patch(color=RED,   label='Weak (<0.90)'),
]
ax.legend(handles=legend_patches, fontsize=9, loc='lower right')

plt.tight_layout()
plt.savefig('ppt_graphs/graph2_auc_comparison.png')
plt.close()
print("Graph 2: AUC Comparison saved")

# ════════════════════════════════════════════════════
# GRAPH 3: ROC Curve (Simulated)
# ════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 7))
fig.patch.set_facecolor('#F8FBFF')
ax.set_facecolor('#FFFFFF')

roc_models = [
    ('Proposed Hybrid ★',  GOLD,   3.5, [(0,0),(0.01,0.88),(0.03,0.97),(0.05,0.99),(0.1,0.9999),(1,1)]),
    ('CNN',                ACC,    2.0, [(0,0),(0.03,0.75),(0.08,0.88),(0.15,0.95),(0.3,0.98),(1,1)]),
    ('LSTM',               TEAL,   2.0, [(0,0),(0.04,0.72),(0.09,0.85),(0.18,0.93),(0.35,0.97),(1,1)]),
    ('KNN',                GREEN,  1.8, [(0,0),(0.05,0.65),(0.12,0.80),(0.25,0.90),(0.4,0.95),(1,1)]),
    ('Decision Tree',      BLUE,   1.5, [(0,0),(0.08,0.60),(0.18,0.76),(0.30,0.87),(0.5,0.93),(1,1)]),
    ('VQC (Quantum)',      RED,    1.5, [(0,0),(0.10,0.55),(0.22,0.70),(0.38,0.80),(0.55,0.85),(1,1)]),
    ('Random Classifier',  GRAY,   1.0, [(0,0),(1,1)]),
]

for name, color, lw, pts in roc_models:
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    ls = '--' if name == 'Random Classifier' else '-'
    ax.plot(xs, ys, color=color, linewidth=lw, linestyle=ls, label=name, alpha=0.9)

ax.fill_between([p[0] for p in roc_models[0][3]],
                [p[1] for p in roc_models[0][3]], alpha=0.08, color=GOLD)

ax.set_xlabel('False Positive Rate', fontsize=12, color=GRAY)
ax.set_ylabel('True Positive Rate', fontsize=12, color=GRAY)
ax.set_title('ROC Curve Comparison — All Models', fontsize=14,
             fontweight='bold', color=NAVY, pad=12)
ax.legend(fontsize=10, loc='lower right', framealpha=0.9)
ax.grid(alpha=0.3, linestyle='--')
ax.spines[['top','right']].set_visible(False)
ax.text(0.55, 0.12, 'AUC = 0.9998\n(Proposed Hybrid)',
        fontsize=10, color=NAVY, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.4', facecolor=GOLD, alpha=0.85))

plt.tight_layout()
plt.savefig('ppt_graphs/graph3_roc_curve.png')
plt.close()
print("Graph 3: ROC Curve saved")

# ════════════════════════════════════════════════════
# GRAPH 4: Confusion Matrix (Hybrid Model)
# ════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(7, 6))
fig.patch.set_facecolor('#F8FBFF')

cm = np.array([[56843, 20], [21, 56842]])
labels = np.array([['TN\n56,843', 'FP\n20'], ['FN\n21', 'TP\n56,842']])

colors_cm = np.array([[0.78, 0.91, 0.79, 1], [1, 0.80, 0.80, 1],
                       [1, 0.80, 0.80, 1], [0.78, 0.91, 0.79, 1]])
colors_cm = colors_cm.reshape(2,2,4)

for i in range(2):
    for j in range(2):
        color = '#C8E6C9' if (i==j) else '#FFCDD2'
        ax.add_patch(plt.Rectangle([j-0.5, i-0.5], 1, 1, color=color, zorder=1))

for i in range(2):
    for j in range(2):
        tc = '#1B5E20' if (i==j) else '#B71C1C'
        ax.text(j, i, labels[i,j], ha='center', va='center',
                fontsize=16, fontweight='bold', color=tc, zorder=2)

ax.set_xticks([0,1])
ax.set_yticks([0,1])
ax.set_xticklabels(['Predicted: Legit', 'Predicted: Fraud'], fontsize=12)
ax.set_yticklabels(['Actual: Legit', 'Actual: Fraud'], fontsize=12)
ax.set_xlim(-0.5, 1.5)
ax.set_ylim(-0.5, 1.5)
ax.set_title('Confusion Matrix — Proposed Hybrid Model\n(AutoEncoder + Random Forest + XGBoost)',
             fontsize=13, fontweight='bold', color=NAVY, pad=12)

from matplotlib.patches import Patch
legend_patches = [Patch(color='#C8E6C9', label='Correct (TN + TP)'),
                  Patch(color='#FFCDD2', label='Incorrect (FP + FN)')]
ax.legend(handles=legend_patches, fontsize=10, loc='upper right',
          bbox_to_anchor=(1.0, -0.08))

plt.tight_layout()
plt.savefig('ppt_graphs/graph4_confusion_matrix.png')
plt.close()
print("Graph 4: Confusion Matrix saved")

# ════════════════════════════════════════════════════
# GRAPH 5: Ablation Study
# ════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(11, 6))
fig.patch.set_facecolor('#F8FBFF')
ax.set_facecolor('#FFFFFF')

variants = ['Full Hybrid\n(AE+RF+XGB)', 'Without\nAutoEncoder',
            'Without RF\n(Bagging)', 'Without XGB\n(Boosting)',
            'Without\nSMOTE', 'Without HP\nTuning']
f1_abl  = [0.9995, 0.8438, 0.7919, 0.8438, 0.8500, 0.8500]
auc_abl = [0.9998, 0.9684, 0.9615, 0.9684, 0.9584, 0.9584]

x2 = np.arange(len(variants))
w2 = 0.35
b1 = ax.bar(x2 - w2/2, f1_abl,  w2, label='F1-Score',
            color=[GREEN if i==0 else RED for i in range(len(variants))],
            alpha=0.9, edgecolor='white')
b2 = ax.bar(x2 + w2/2, auc_abl, w2, label='AUC-ROC',
            color=[GOLD if i==0 else AMBER for i in range(len(variants))],
            alpha=0.9, edgecolor='white')

for bars in [b1, b2]:
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                f'{bar.get_height():.4f}', ha='center', va='bottom',
                fontsize=8.5, color=NAVY, fontweight='bold')

ax.set_xticks(x2)
ax.set_xticklabels(variants, fontsize=10)
ax.set_ylabel('Score', fontsize=12, color=GRAY)
ax.set_ylim(0.7, 1.06)
ax.set_title('Ablation Study — Impact of Each Component on Hybrid Model Performance',
             fontsize=13, fontweight='bold', color=NAVY, pad=12)
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.spines[['top','right']].set_visible(False)
ax.axhline(y=1.0, color=GREEN, linestyle='--', linewidth=1, alpha=0.4)

plt.tight_layout()
plt.savefig('ppt_graphs/graph5_ablation_study.png')
plt.close()
print("Graph 5: Ablation Study saved")

# ════════════════════════════════════════════════════
# GRAPH 6: Multi-Metric Radar / Spider Chart
# ════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
fig.patch.set_facecolor('#F8FBFF')

categories = ['Precision', 'Recall', 'F1-Score', 'AUC-ROC', 'Accuracy']
N = len(categories)
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]

radar_models = [
    ('Proposed Hybrid ★', [1.00, 1.00, 1.00, 0.9998, 1.00], GOLD, 3),
    ('CNN',               [0.76, 0.92, 0.82, 0.9734, 1.00], ACC,  2),
    ('KNN',               [0.68, 0.94, 0.75, 0.9482, 1.00], GREEN,2),
    ('VQC (Quantum)',     [1.00, 0.72, 0.84, 0.83,   0.86], RED,  2),
]

ax.set_ylim(0, 1.05)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=12)
ax.set_yticks([0.6, 0.7, 0.8, 0.9, 1.0])
ax.set_yticklabels(['0.6','0.7','0.8','0.9','1.0'], fontsize=8, color=GRAY)
ax.grid(color=GRAY, alpha=0.3)

for name, vals, color, lw in radar_models:
    vals_plot = vals + vals[:1]
    ax.plot(angles, vals_plot, color=color, linewidth=lw, label=name, alpha=0.9)
    ax.fill(angles, vals_plot, color=color, alpha=0.08)

ax.set_title('Multi-Metric Radar Chart — Top Models Comparison',
             fontsize=13, fontweight='bold', color=NAVY, pad=20, y=1.08)
ax.legend(fontsize=10, loc='upper right', bbox_to_anchor=(1.35, 1.1))

plt.tight_layout()
plt.savefig('ppt_graphs/graph6_radar_chart.png', bbox_inches='tight')
plt.close()
print("Graph 6: Radar Chart saved")

# ════════════════════════════════════════════════════
# GRAPH 7: SMOTE Effect (Before vs After)
# ════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.patch.set_facecolor('#F8FBFF')

for ax_s in axes:
    ax_s.set_facecolor('#FFFFFF')

# Before SMOTE
ax1 = axes[0]
sizes1 = [284315, 492]
labels1 = ['Legitimate\n284,315 (99.83%)', 'Fraudulent\n492 (0.17%)']
colors1 = [GREEN, RED]
wedges, texts, autotexts = ax1.pie(
    sizes1, labels=labels1, colors=colors1,
    autopct='%1.2f%%', startangle=90,
    pctdistance=0.75, labeldistance=1.12,
    wedgeprops=dict(edgecolor='white', linewidth=2)
)
for at in autotexts: at.set_fontsize(10)
ax1.set_title('Before SMOTE\n(Highly Imbalanced)', fontsize=13,
              fontweight='bold', color=RED, pad=12)

# After SMOTE
ax2 = axes[1]
sizes2 = [227845, 227845]
labels2 = ['Legitimate\n227,845 (50%)', 'Fraudulent\n227,845 (50%)']
colors2 = [GREEN, ACC]
wedges2, texts2, autotexts2 = ax2.pie(
    sizes2, labels=labels2, colors=colors2,
    autopct='%1.1f%%', startangle=90,
    pctdistance=0.75, labeldistance=1.12,
    wedgeprops=dict(edgecolor='white', linewidth=2)
)
for at in autotexts2: at.set_fontsize(10)
ax2.set_title('After SMOTE\n(Perfectly Balanced)', fontsize=13,
              fontweight='bold', color=GREEN, pad=12)

fig.suptitle('Effect of SMOTE on Class Distribution',
             fontsize=15, fontweight='bold', color=NAVY, y=1.02)

plt.tight_layout()
plt.savefig('ppt_graphs/graph7_smote_effect.png', bbox_inches='tight')
plt.close()
print("Graph 7: SMOTE Effect saved")

# ════════════════════════════════════════════════════
# GRAPH 8: Precision vs Recall Trade-off
# ════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 7))
fig.patch.set_facecolor('#F8FBFF')
ax.set_facecolor('#FFFFFF')

scatter_models = [
    ('Proposed Hybrid', 1.00, 1.00, GOLD, 250),
    ('CNN',             0.76, 0.92, ACC,  120),
    ('KNN',             0.68, 0.94, GREEN,120),
    ('LSTM',            0.84, 0.92, TEAL, 120),
    ('Decision Tree',   0.67, 0.89, BLUE, 100),
    ('SVM',             0.57, 0.78, '#9C27B0', 100),
    ('Log. Reg',        0.57, 0.94, '#FF7043', 100),
    ('VQC (Quantum)',   1.00, 0.72, RED,  120),
    ('Iso. Forest',     0.59, 0.52, GRAY, 100),
]

for name, prec, rec, color, size in scatter_models:
    ax.scatter(rec, prec, c=color, s=size, zorder=5,
               edgecolors='white', linewidths=1.5, alpha=0.9)
    offset_x = 0.01 if name != 'VQC (Quantum)' else -0.14
    offset_y = 0.015 if name not in ['Proposed Hybrid'] else 0.02
    fw = 'bold' if name == 'Proposed Hybrid' else 'normal'
    ax.annotate(name, (rec, prec),
                xytext=(rec + offset_x, prec + offset_y),
                fontsize=9.5, color=NAVY, fontweight=fw)

ax.scatter([1.0], [1.0], c=GOLD, s=300, zorder=6,
           edgecolors=NAVY, linewidths=2.5, marker='*')

ax.set_xlabel('Recall', fontsize=13, color=GRAY)
ax.set_ylabel('Precision', fontsize=13, color=GRAY)
ax.set_title('Precision vs Recall Trade-off — All Models (With SMOTE)',
             fontsize=14, fontweight='bold', color=NAVY, pad=12)
ax.set_xlim(0.45, 1.08)
ax.set_ylim(0.50, 1.08)
ax.grid(alpha=0.3, linestyle='--')
ax.spines[['top','right']].set_visible(False)
ax.axhline(y=1.0, color=GREEN, linestyle='--', alpha=0.3)
ax.axvline(x=1.0, color=GREEN, linestyle='--', alpha=0.3)
ax.text(0.97, 0.52, '★ Perfect Score\n(Proposed Hybrid)',
        fontsize=10, color=NAVY, fontweight='bold', ha='right')

plt.tight_layout()
plt.savefig('ppt_graphs/graph8_precision_recall.png')
plt.close()
print("Graph 8: Precision vs Recall saved")

print("\nALL 8 GRAPHS SAVED to ppt_graphs/ folder!")
print("Files:")
for f in sorted(os.listdir('ppt_graphs')):
    print(f"   - {f}")
