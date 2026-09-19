import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-v0_8-whitegrid')
df = pd.read_csv('image_tcav_analysis.csv')
concept_cols = ['lung_opacity', 'consolidation', 'pleural_effusion']

fig, axes = plt.subplots(figsize=(5, 5), dpi=300)
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

# A. Violin Plot
# df_melted = df.melt(id_vars=['filename'], value_vars=concept_cols, 
#                     var_name='Clinical Concept', value_name='Directional Derivative')
# sns.violinplot(data=df_melted, x='Clinical Concept', y='Directional Derivative', ax=axes[0], palette=colors, cut=0)
# axes[0].axhline(0, color='red', linestyle='--', linewidth=1.5, label='Zero Sensitivity')
# axes[0].set_title('A. Continuous Sensitivity Distributions', fontweight='bold')
# axes[0].legend()

# B. Mean TCAV Score Bar Plot
mean_scores = [0.9819, 0.9956, 0.8913]
std_devs = [0.0230, 0.0061, 0.0687]
bars = axes.bar(concept_cols, mean_scores, yerr=std_devs, capsize=6, color=colors, alpha=0.85, edgecolor='black', width=0.5)
axes.axhline(0.50, color='black', linestyle=':', linewidth=2, label='Random Baseline (0.50)')
axes.set_ylim(0, 1.15)
axes.set_title('Mean TCAV Score (5 Runs)', fontweight='bold')
axes.legend()

for bar, mean_val in zip(bars, mean_scores):
    axes.text(bar.get_x() + bar.get_width()/2.0, mean_val + 0.08, f'{mean_val:.4f}', ha='center', va='bottom', fontweight='bold')

# C. Concept Independence Matrix
# corr_matrix = df[concept_cols].corr()
# sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='Blues', ax=axes[2], cbar=False, vmin=0, vmax=1)
# axes[2].set_title('C. Feature Independence Heatmap', fontweight='bold')

plt.tight_layout()
# plt.savefig('tcav_publication_summary.png', bbox_inches='tight')
plt.savefig('tcav_summary.png', bbox_inches='tight')

print("✅ Saved publication plot to 'tcav_publication_summary.png'")