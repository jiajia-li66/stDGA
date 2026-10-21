import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
import numpy as np
from umap import UMAP
from matplotlib.cm import get_cmap

def save_clustering_plot(data, labels, method="pca", path="clustering_plot.png", title="Clustering Result"):
    # 如果是 tensor 转 numpy
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    # 降维方法选择
    if method == "pca":
        reducer = PCA(n_components=2)
    elif method == "tsne":
        reducer = TSNE(n_components=2, random_state=123, init='pca', learning_rate='auto')
    elif method == "umap":
        reducer = UMAP(n_components=2, random_state=123)
    else:
        raise ValueError("method must be 'pca', 'tsne' or 'umap'")

    # 降维处理
    reduced = reducer.fit_transform(data)

    # 配色与样式
    sns.set(style="whitegrid", font_scale=1.2)
    unique_labels = np.unique(labels)
    n_colors = len(unique_labels)
    palette = sns.color_palette("tab20", n_colors=n_colors)
    color_dict = {label: palette[i % len(palette)] for i, label in enumerate(unique_labels)}
    label_colors = [color_dict[l] for l in labels]

    # 绘图
    plt.figure(figsize=(6, 5))
    plt.scatter(
        reduced[:, 0], reduced[:, 1],
        c=label_colors, s=25, alpha=0.8, edgecolors='w', linewidths=0.4
    )
    plt.title(title, fontsize=14)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight', transparent=True)
    plt.close()


def save_evaluation_results(metrics, params, save_path="evaluation_log.csv"):
    record = {**metrics, **params}
    df_new = pd.DataFrame([record])
    if os.path.exists(save_path):
        df_old = pd.read_csv(save_path)
        df_all = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_all = df_new
    df_all.to_csv(save_path, index=False)
    return df_all.tail(1)
