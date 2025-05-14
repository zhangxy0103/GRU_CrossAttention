import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import os

def load_and_preprocess_data():
    handwriting_df = pd.read_csv('path')
    speech_df = pd.read_csv('path')

    hw_features = handwriting_df.drop(columns=['id', 'class'])
    sp_features = speech_df.drop(columns=['id', 'class'])

    scaler = StandardScaler()
    hw_scaled = scaler.fit_transform(hw_features)
    sp_scaled = scaler.fit_transform(sp_features)

    return hw_scaled, sp_scaled, handwriting_df['id'], handwriting_df['class'], hw_features.columns, sp_features.columns

def save_pca_results(pca_obj, features, ids, labels, prefix, folder, n_components):
    pca_features = pca_obj.transform(features)

    columns = [f'{prefix}_pc{i + 1}' for i in range(n_components)]
    pca_df = pd.DataFrame(pca_features[:, :n_components], columns=columns)
    pca_df['id'] = ids
    pca_df['class'] = labels

    filename = f'pca_results/{folder}/{prefix}_pca_{n_components}.csv'
    pca_df.to_csv(filename, index=False)
    print(f'Saved: {filename}')

def save_pca_loadings(pca_obj, feature_names, prefix, folder, n_components):
    loadings = pca_obj.components_[:n_components]
    pc_labels = [f'PC{i+1}' for i in range(n_components)]

    loading_df = pd.DataFrame(loadings.T, index=feature_names, columns=pc_labels)

    filename = f'pca_results/{folder}/{prefix}_loadings_{n_components}.csv'
    loading_df.to_csv(filename)
    print(f'Saved Loadings: {filename}')

def perform_pca_and_plot(hw_scaled, sp_scaled, hw_ids, hw_labels, hw_columns, sp_columns):
    fused_features = np.hstack((hw_scaled, sp_scaled))

    pca_hw = PCA(n_components=0.95).fit(hw_scaled)
    pca_sp = PCA(n_components=0.95).fit(sp_scaled)
    pca_fused = PCA(n_components=0.95).fit(fused_features)

    for n_components in [16, 27]:
        if pca_hw.n_components_ >= n_components:
            save_pca_results(pca_hw, hw_scaled, hw_ids, hw_labels, 'handwriting', f'pc{n_components}', n_components)
            save_pca_loadings(pca_hw, hw_columns, 'handwriting', f'pc{n_components}', n_components)

        if pca_sp.n_components_ >= n_components:
            save_pca_results(pca_sp, sp_scaled, hw_ids, hw_labels, 'speech', f'pc{n_components}', n_components)
            save_pca_loadings(pca_sp, sp_columns, 'speech', f'pc{n_components}', n_components)

        if pca_fused.n_components_ >= n_components:
            save_pca_results(pca_fused, fused_features, hw_ids, hw_labels, 'fused', f'pc{n_components}', n_components)

    hw_cumsum = pca_hw.explained_variance_ratio_.cumsum()
    sp_cumsum = pca_sp.explained_variance_ratio_.cumsum()
    fused_cumsum = pca_fused.explained_variance_ratio_.cumsum()

    plt.figure(figsize=(10, 6), dpi=300)
    ax = plt.gca()
    for spine in ['top', 'right', 'bottom', 'left']:
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_linewidth(1.5)

    colors = ['#4E79A7', '#F28E2B', '#59A14F']
    plt.plot(range(1, len(hw_cumsum) + 1), hw_cumsum, 'o-', color=colors[0], markersize=6, linewidth=2, label='Handwriting PCA')
    plt.plot(range(1, len(sp_cumsum) + 1), sp_cumsum, 's-', color=colors[1], markersize=6, linewidth=2, label='Speech PCA')
    plt.plot(range(1, len(fused_cumsum) + 1), fused_cumsum, 'D-', color=colors[2], markersize=6, linewidth=2, label='Fused PCA')

    def annotate_pc(n, y_val, offset_x=0.5):
        plt.plot(n, y_val, '*', color='#D62728', markersize=12, markeredgewidth=1)
        plt.text(n + offset_x, y_val - 0.03, f'PC{n}: {y_val:.1%}', fontsize=11, color='#D62728',
                 bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#D62728', lw=1, alpha=0.8))

    if len(fused_cumsum) >= 10:
        annotate_pc(10, fused_cumsum[9])
    threshold_idx = np.argmax(fused_cumsum >= 0.8)
    if threshold_idx.size > 0:
        annotate_pc(threshold_idx + 1, fused_cumsum[threshold_idx], offset_x=-3 if (threshold_idx + 1) > 20 else 0.5)
    if len(fused_cumsum) > 0:
        annotate_pc(len(fused_cumsum), fused_cumsum[-1], offset_x=-4)

    plt.axhline(0.8, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)
    plt.text(1, 0.82, '80% Variance Threshold', fontsize=10, color='gray')

    max_pc = max(len(hw_cumsum), len(sp_cumsum), len(fused_cumsum))
    plt.xticks(np.arange(0, max_pc + 1, 5))
    plt.xlim(0.5, max_pc + 0.5)
    plt.ylim(0, 1.05)
    plt.xlabel('Number of Principal Components', fontsize=12, labelpad=10)
    plt.ylabel('Cumulative Explained Variance', fontsize=12, labelpad=10)
    plt.title('Multimodal PCA Variance Comparison', fontsize=14, pad=15)
    plt.legend(fontsize=10, loc='lower right', frameon=True)
    plt.grid(False)
    plt.tight_layout()
    plt.savefig('multimodal_pca_comparison_pro.png', dpi=300, bbox_inches='tight')
    plt.show()

    return pca_hw, pca_sp, pca_fused

if __name__ == "__main__":
    hw_scaled, sp_scaled, hw_ids, hw_labels, hw_columns, sp_columns = load_and_preprocess_data()
    pca_hw, pca_sp, pca_fused = perform_pca_and_plot(hw_scaled, sp_scaled, hw_ids, hw_labels, hw_columns, sp_columns)
