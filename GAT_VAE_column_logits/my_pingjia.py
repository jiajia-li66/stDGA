import torch
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import adjusted_rand_score, silhouette_score, normalized_mutual_info_score
from sklearn.metrics import adjusted_mutual_info_score, homogeneity_score, completeness_score, v_measure_score
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, fowlkes_mallows_score
from scipy.spatial.distance import jensenshannon
import pandas as pd
from sklearn.neighbors import kneighbors_graph
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering


CELLTYPE_COLOR_MAP = {
    "Astro-1": "#1f77b4",
    "Astro-2": "#aec7e8",
    "Endo": "#ff7f0e",
    "HPC": "#ffbb78",
    "Micro": "#2ca02c",
    "Oligo": "#98df8a",
    "PVALB": "#d62728",
    "Reln": "#ff9896",
    "SST": "#9467bd",
    "Smc": "#c5b0d5",
    "VIP": "#8c564b",
    "Vip": "#8c564b",
    "eL2/3": "#c49c94",
    "eL4": "#e377c2",
    "eL5": "#f7b6d2",
    "eL6-1": "#7f7f7f",
    "eL6-2": "#c7c7c7",
}

CELLTYPE_LEGEND_ORDER = [
    "Astro-1", "Astro-2", "Endo", "HPC", "Micro", "Oligo", "PVALB", "Reln",
    "SST", "Smc", "VIP", "eL2/3", "eL4", "eL5", "eL6-1", "eL6-2",
]


def _display_label(label):
    label = str(label)
    return "VIP" if label == "Vip" else label


def _ordered_labels(labels):
    labels = np.asarray(labels)
    unique_labels = list(np.unique(labels))
    label_by_display = {_display_label(label): label for label in unique_labels}
    ordered = [label_by_display[name] for name in CELLTYPE_LEGEND_ORDER if name in label_by_display]
    ordered.extend([label for label in unique_labels if label not in ordered])
    return ordered


def _scatter_kwargs(label):
    color = CELLTYPE_COLOR_MAP.get(str(label), CELLTYPE_COLOR_MAP.get(_display_label(label)))
    return {"color": color} if color is not None else {}


def _clusters_to_majority_celltypes(cluster_labels, true_label_names):
    cluster_labels = np.asarray(cluster_labels)
    true_label_names = np.asarray(true_label_names).astype(str)
    mapped = np.empty(cluster_labels.shape, dtype=object)

    for cluster in np.unique(cluster_labels):
        mask = cluster_labels == cluster
        names, counts = np.unique(true_label_names[mask], return_counts=True)
        mapped[mask] = names[np.argmax(counts)]

    return mapped


def expression_umap_plot(imputed_X, labels, path=None, title="UMAP of Expression Space", legend_prefix="Label "):
    """与最初版本一致：matplotlib 默认颜色循环，按 np.unique(labels) 顺序着色。"""
    import umap

    if isinstance(imputed_X, torch.Tensor):
        imputed_X = imputed_X.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    reducer = umap.UMAP(n_components=2, random_state=123)
    embedding = reducer.fit_transform(imputed_X)

    df = pd.DataFrame(embedding, columns=["x", "y"])
    df['label'] = labels

    plt.figure(figsize=(7, 5))
    for lbl in _ordered_labels(labels):
        subset = df[df['label'] == lbl]
        plt.scatter(
            subset['x'],
            subset['y'],
            label=f"{legend_prefix}{_display_label(lbl)}",
            alpha=0.7,
            s=20,
            edgecolors="#333333",
            linewidths=0.2,
            **_scatter_kwargs(lbl),
        )
    plt.title(title)
    plt.grid(False)
    plt.legend(
        bbox_to_anchor=(1.05, 0.5),
        loc='center left',
        borderaxespad=0.,
    )
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    if path:
        plt.savefig(path, dpi=300, bbox_inches='tight')
        print(f"🖼 UMAP表达空间图已保存: {path}")
    plt.close()


def expression_spatial_plot(spatial_coords, labels, path=None, title="Spatial Distribution", legend_prefix="Label "):
    """与 UMAP 相同绘图方式，保证默认配色一致；仅坐标换为空间 X–Y（前两列）。"""
    if isinstance(spatial_coords, torch.Tensor):
        spatial_coords = spatial_coords.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    xy = np.asarray(spatial_coords)
    if xy.ndim != 2 or xy.shape[1] < 2:
        raise ValueError("spatial_coords 需要至少两列作为 X、Y")
    df = pd.DataFrame(xy[:, :2], columns=["x", "y"])
    df['label'] = labels

    plt.figure(figsize=(7, 5))
    for lbl in _ordered_labels(labels):
        subset = df[df['label'] == lbl]
        plt.scatter(
            subset['x'],
            subset['y'],
            label=f"{legend_prefix}{_display_label(lbl)}",
            alpha=0.7,
            s=20,
            edgecolors="#333333",
            linewidths=0.2,
            **_scatter_kwargs(lbl),
        )
    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(False)
    plt.legend(
        bbox_to_anchor=(1.05, 0.5),
        loc='center left',
        borderaxespad=0.,
    )
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    if path:
        plt.savefig(path, dpi=300, bbox_inches='tight')
        print(f"🖼 空间坐标图已保存: {path}")
    plt.close()



def compute_similarity_matrix(matrix, axis=0):
    matrix = np.asarray(matrix, dtype=np.float32)
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    corr = np.corrcoef(matrix if axis == 1 else matrix.T)
    return np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)


def safe_pearsonr(x, y):
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    finite_mask = np.isfinite(x) & np.isfinite(y)
    x = x[finite_mask]
    y = y[finite_mask]

    if x.size < 2 or y.size < 2:
        return 0.0
    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0

    try:
        corr, _ = pearsonr(x, y)
        if np.isnan(corr) or np.isinf(corr):
            return 0.0
        return float(corr)
    except Exception:
        return 0.0


def evaluate_similarity(test_result, test_true):
    assert test_result.shape == test_true.shape, "输入 shape 不匹配"
    cell_sim_result = compute_similarity_matrix(test_result, axis=0)
    cell_sim_true = compute_similarity_matrix(test_true, axis=0)
    gene_sim_result = compute_similarity_matrix(test_result, axis=1)
    gene_sim_true = compute_similarity_matrix(test_true, axis=1)
    cell_corr = safe_pearsonr(cell_sim_result.flatten(), cell_sim_true.flatten())
    gene_corr = safe_pearsonr(gene_sim_result.flatten(), gene_sim_true.flatten())
    return cell_corr, gene_corr


def compute_js_divergence(p, q):
    p = np.clip(p, 1e-10, 1)
    q = np.clip(q, 1e-10, 1)
    return jensenshannon(p, q) ** 2


def compute_morans_I(x, coords, k=10):
    if len(coords) <= 1:
        return float('nan')
    k = min(k, len(coords) - 1)
    if k < 1:
        return float('nan')
    W = kneighbors_graph(coords, n_neighbors=k, mode='connectivity', include_self=False).toarray()
    np.fill_diagonal(W, 0)
    if W.sum() == 0 or np.allclose(x, x[0]):
        return float('nan')
    W = W / W.sum()
    x_mean = np.mean(x)
    num = np.sum(W * np.outer(x - x_mean, x - x_mean))
    denom = np.sum((x - x_mean) ** 2)
    return (len(x) / W.sum()) * (num / denom) if denom != 0 else float('nan')


def summarize(values):
    values = np.array(values)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return {'mean': float('nan'), 'std': float('nan'), 'min': float('nan'),
                '25%': float('nan'), '50%': float('nan'), '75%': float('nan'), 'max': float('nan')}
    return {
        'mean': np.mean(values),
        'std': np.std(values),
        'min': np.min(values),
        '25%': np.percentile(values, 25),
        '50%': np.percentile(values, 50),
        '75%': np.percentile(values, 75),
        'max': np.max(values),
    }


def _kmeans_labels_or_agglo(X, n_clusters, random_state=123, n_init=20):
    """KMeans with fallback to AgglomerativeClustering if empty clusters."""
    pred = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=n_init).fit_predict(X)
    if len(np.unique(pred)) < n_clusters:
        pred = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward').fit_predict(X)
    return pred


def evaluate_gat_autoencoder(X_imputed, original_x, mask, mark, true_labels, spatial_coords,
                             plot_path=None, only_clustering=False, save_filled_path=None):
    logits = X_imputed[:original_x.shape[0], :original_x.shape[1]]
    X_imputed = X_imputed[:original_x.shape[0], :]

    X_for_clustering = X_imputed.detach().cpu().numpy()
    X_common_genes = X_imputed[:, :original_x.shape[1]].detach().cpu().numpy()
    mark = mark[:original_x.shape[0], :original_x.shape[1]]
    eval_mask = mask[:original_x.shape[0], :original_x.shape[1]]
    logits_final = logits * (1 - eval_mask) + original_x * eval_mask

    true_label_names = np.array(true_labels[:original_x.shape[0]]).astype(str)
    _, true_labels = np.unique(true_label_names, return_inverse=True)

    n_clusters = len(np.unique(true_labels))

    predicted_labels = _kmeans_labels_or_agglo(X_for_clustering, n_clusters)

    scaler = StandardScaler(with_mean=True, with_std=True)
    X_scaled = scaler.fit_transform(X_for_clustering)
    pca_dim = int(min(50, X_scaled.shape[1], max(2, X_scaled.shape[0] - 1)))
    X_pca = PCA(n_components=pca_dim, random_state=123).fit_transform(X_scaled)
    predicted_labels_pca = _kmeans_labels_or_agglo(X_pca, n_clusters)
    predicted_label_names = _clusters_to_majority_celltypes(predicted_labels, true_label_names)
    predicted_label_names_pca = _clusters_to_majority_celltypes(predicted_labels_pca, true_label_names)

    pred_unique = np.unique(predicted_labels)
    pred_unique_pca = np.unique(predicted_labels_pca)

    print(f"📌 True labels: {n_clusters} 类, Predicted (no PCA): {len(pred_unique)} 类, Predicted (PCA): {len(pred_unique_pca)} 类")
    ari = adjusted_rand_score(true_labels, predicted_labels)
    try:
        sc = silhouette_score(X_for_clustering, predicted_labels)
    except Exception:
        sc = float('nan')
    nmi = normalized_mutual_info_score(true_labels, predicted_labels)

    # === 可视化 ===
    if plot_path:
        # 拆分出路径前缀和扩展名
        if plot_path.endswith('.png'):
            plot_prefix = plot_path[:-4]
        else:
            plot_prefix = plot_path

        # 画真实标签图
        expression_umap_plot(
            X_for_clustering,
            true_label_names,
            path=plot_prefix + "_true.png",
            title=f"UMAP - True Labels ({n_clusters} classes)",
            legend_prefix=""
        )

        # 画预测标签图
        expression_umap_plot(
            X_for_clustering,
            predicted_label_names,
            path=plot_prefix + "_pred.png",
            title=f"UMAP - Predicted KMeans ({len(pred_unique)} clusters)",
            legend_prefix=""
        )
        expression_umap_plot(
            X_for_clustering,
            predicted_label_names_pca,
            path=plot_prefix + "_pred_pca.png",
            title=f"UMAP - Predicted PCA+KMeans ({len(pred_unique_pca)} clusters)",
            legend_prefix=""
        )

        # 画原始空间坐标图，配色顺序与原始 UMAP 绘图一致
        expression_spatial_plot(
            spatial_coords,
            true_label_names,
            path=plot_prefix + "_true_spatial.png",
            title=f"Filled Spatial (True Labels, {n_clusters} classes)",
            legend_prefix=""
        )
        expression_spatial_plot(
            spatial_coords,
            predicted_label_names,
            path=plot_prefix + "_pred_spatial.png",
            title=f"Filled Spatial (Pred , {len(pred_unique)} clusters)",
            legend_prefix=""
        )
        expression_spatial_plot(
            spatial_coords,
            predicted_label_names_pca,
            path=plot_prefix + "_pred_spatial_pca.png",
            title=f"Filled Spatial (Pred PCA, {len(pred_unique_pca)} clusters)",
            legend_prefix=""
        )

    if save_filled_path:
        import os
        out = os.path.abspath(save_filled_path)
        parent = os.path.dirname(out)
        if parent:
            os.makedirs(parent, exist_ok=True)
        np.savez_compressed(
            out,
            imputed_X=X_common_genes.astype(np.float32),
            n_cells=int(X_common_genes.shape[0]),
            n_genes=int(X_common_genes.shape[1]),
        )
        print(f"📁 已保存填充后表达矩阵: {out} (keys: imputed_X, n_cells, n_genes)")

    clustering_metrics = {
        'ARI': ari,
        'NMI': nmi,
        'Silhouette': sc,
    }
    # 额外聚类指标
    try:
        ami = adjusted_mutual_info_score(true_labels, predicted_labels)
    except Exception:
        ami = float('nan')
    try:
        homogeneity = homogeneity_score(true_labels, predicted_labels)
    except Exception:
        homogeneity = float('nan')
    try:
        completeness = completeness_score(true_labels, predicted_labels)
    except Exception:
        completeness = float('nan')
    try:
        vmeasure = v_measure_score(true_labels, predicted_labels)
    except Exception:
        vmeasure = float('nan')
    try:
        ch_score = calinski_harabasz_score(X_for_clustering, predicted_labels)
    except Exception:
        ch_score = float('nan')
    try:
        db_score = davies_bouldin_score(X_for_clustering, predicted_labels)
    except Exception:
        db_score = float('nan')
    try:
        fmi = fowlkes_mallows_score(true_labels, predicted_labels)
    except Exception:
        fmi = float('nan')
    # purity
    try:
        n_samples = len(predicted_labels)
        total_max = 0
        for lbl in np.unique(predicted_labels):
            cluster_mask = predicted_labels == lbl
            true_subset = true_labels[cluster_mask]
            if len(true_subset) == 0:
                continue
            counts = np.bincount(true_subset)
            total_max += counts.max()
        purity = total_max / n_samples if n_samples > 0 else float('nan')
    except Exception:
        purity = float('nan')

    clustering_metrics.update({
        'AMI': ami,
        'Homogeneity': homogeneity,
        'Completeness': completeness,
        'V-Measure': vmeasure,
        'Calinski-Harabasz': ch_score,
        'Davies-Bouldin': db_score,
        'Fowlkes-Mallows': fmi,
        'Purity': purity,
    })
    if only_clustering:
        return pd.DataFrame(clustering_metrics, index=[0]), {"global_metrics": clustering_metrics}

    # == 全指标 ==
    per_gene_df, extended_metrics = evaluate_gene_level_metrics(
        logits_final, original_x, eval_mask, spatial_coords
    )
    global_metrics = extended_metrics["global_metrics"]
    global_metrics.update(clustering_metrics)

    # 📊 打印每列指标的统计摘要
    print("\n🧬 Per-gene Evaluation Summary:")
    print(per_gene_df.describe().loc[["mean", "std", "25%", "50%", "75%"]].round(6))

    # 🌍 打印全局指标
    print("\n🌍 Flattened Global Metrics:")
    for k, v in global_metrics.items():
        print(f"{k:<12}: {v:.6f}")

    return per_gene_df, {"global_metrics": global_metrics}




def evaluate_gene_level_metrics(logits, original_x, mask, spatial_coords):
    num_genes = original_x.shape[1]

    mse_list, rmse_list, mae_list, pcc_list, cos_list, js_div_list, moran_I_list = [], [], [], [], [], [], []

    logits_np = logits.detach().cpu().numpy() if hasattr(logits, "detach") else np.array(logits)
    original_np = original_x.detach().cpu().numpy() if hasattr(original_x, "detach") else np.array(original_x)
    mask_np = mask.detach().cpu().numpy() if hasattr(mask, "detach") else np.array(mask)
    logits_np = np.nan_to_num(logits_np, nan=0.0, posinf=0.0, neginf=0.0)
    original_np = np.nan_to_num(original_np, nan=0.0, posinf=0.0, neginf=0.0)
    mask_np = np.nan_to_num(mask_np, nan=0.0, posinf=0.0, neginf=0.0)
    mask_np = 1 - mask_np
    coords_np = spatial_coords

    for gene_idx in range(num_genes):
        gene_mask = mask_np[:, gene_idx]
        if np.sum(gene_mask) == 0:
            continue
        original_col = original_np[:, gene_idx][gene_mask.astype(bool)]
        pred_col = logits_np[:, gene_idx][gene_mask.astype(bool)]
        coords_col = coords_np[gene_mask.astype(bool)]

        if np.all(original_col == 0) and np.all(pred_col == 0):
            continue

        mse_list.append(np.mean((pred_col - original_col) ** 2))
        rmse_list.append(np.sqrt(np.mean((pred_col - original_col) ** 2)))
        mae_list.append(mean_absolute_error(original_col, pred_col))

        pcc = np.corrcoef(original_col, pred_col)[0, 1] if np.std(original_col) > 0 and np.std(pred_col) > 0 else 0
        pcc_list.append(pcc)

        cos_sim = cosine_similarity(pred_col.reshape(1, -1), original_col.reshape(1, -1))[0, 0]
        cos_list.append(cos_sim)

        p_soft = F.softmax(torch.tensor(original_col), dim=0).numpy()
        q_soft = F.softmax(torch.tensor(pred_col), dim=0).numpy()
        js_div_list.append(compute_js_divergence(p_soft, q_soft))

        moran_I_list.append(compute_morans_I(pred_col, coords_col, k=10))

    stats = {
        'gene_rmse': summarize(rmse_list),
        'gene_mae': summarize(mae_list),
        'gene_pcc': summarize(pcc_list),
        'gene_cos': summarize(cos_list),
        'gene_js_divergence': summarize(js_div_list),
        'gene_morans_I': summarize(moran_I_list),
    }

    flattened_mask = mask_np.astype(bool)
    all_true = original_np[flattened_mask]
    all_pred = logits_np[flattened_mask]

    global_stats = {
        'RMSE': np.sqrt(np.mean((all_pred - all_true) ** 2)),
        'MAE': mean_absolute_error(all_true, all_pred),
        'MSE': np.mean((all_pred - all_true) ** 2),
        'PCC': safe_pearsonr(all_true, all_pred),
        'Cosine': cosine_similarity(all_pred.reshape(1, -1), all_true.reshape(1, -1))[0, 0],
        'CellSim': evaluate_similarity(logits_np, original_np)[0],
        'GeneSim': evaluate_similarity(logits_np, original_np)[1],
    }

    return pd.DataFrame({
        'RMSE': rmse_list,
        'MAE': mae_list,
        'PCC': pcc_list,
        'CosineSim': cos_list,
        'JS_Divergence': js_div_list,
        'Morans_I': moran_I_list,
    }), {
        'gene_stat_summary': stats,
        'global_metrics': global_stats
    }
