import torch
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import h5py
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from scipy.stats import pearsonr
from sklearn.metrics import adjusted_rand_score, silhouette_score, normalized_mutual_info_score
from sklearn.metrics import adjusted_mutual_info_score, homogeneity_score, completeness_score, v_measure_score
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, fowlkes_mallows_score
from utils import save_clustering_plot
import scipy.sparse as sp
import os

def compute_similarity_matrix(matrix, axis=0):
    # 转为 numpy
    if isinstance(matrix, torch.Tensor):
        matrix = matrix.cpu().numpy()

    # 转为 float32 并保证密度合理
    matrix = matrix.astype(np.float32)

    # 如果 axis=1，表示按基因比较，需要转置为基因 × 样本
    if axis == 1:
        matrix = matrix.T
    num_samples = matrix.shape[0]
    similarity_matrix = np.zeros((num_samples, num_samples))

    for i in range(num_samples):
        for j in range(i, num_samples):  # 只计算上三角，减少冗余
            corr, _ = pearsonr(matrix[i], matrix[j])
            similarity_matrix[i, j] = corr
            similarity_matrix[j, i] = corr  # 对称赋值
    return similarity_matrix

def evaluate_similarity(test_result, test_true):
    if isinstance(test_result, torch.Tensor):
        test_result = test_result.detach().cpu().numpy()
    if isinstance(test_true, torch.Tensor):
        test_true = test_true.detach().cpu().numpy()

    # 相似度矩阵
    cell_sim_result = compute_similarity_matrix(test_result, axis=0)
    cell_sim_true = compute_similarity_matrix(test_true, axis=0)
    gene_sim_result = compute_similarity_matrix(test_result, axis=1)
    gene_sim_true = compute_similarity_matrix(test_true, axis=1)

    # 展开成一维向量
    # 写入 H5 必须保持固定长度 n*n，因此写入用向量不做过滤，只把 NaN/inf 替换为 0。
    # 相关系数计算仍然基于 mask 过滤无效值。
    cell_sim_result_flat_raw = cell_sim_result.flatten()
    cell_sim_true_flat_raw = cell_sim_true.flatten()
    gene_sim_result_flat_raw = gene_sim_result.flatten()
    gene_sim_true_flat_raw = gene_sim_true.flatten()

    cell_sim_result_flat_save = np.nan_to_num(cell_sim_result_flat_raw, nan=0.0, posinf=0.0, neginf=0.0)
    gene_sim_result_flat_save = np.nan_to_num(gene_sim_result_flat_raw, nan=0.0, posinf=0.0, neginf=0.0)

    valid_cell_mask = (
        ~np.isnan(cell_sim_result_flat_raw)
        & ~np.isnan(cell_sim_true_flat_raw)
        & ~np.isinf(cell_sim_result_flat_raw)
        & ~np.isinf(cell_sim_true_flat_raw)
    )
    cell_sim_result_flat = cell_sim_result_flat_raw[valid_cell_mask]
    cell_sim_true_flat = cell_sim_true_flat_raw[valid_cell_mask]

    valid_gene_mask = (
        ~np.isnan(gene_sim_result_flat_raw)
        & ~np.isnan(gene_sim_true_flat_raw)
        & ~np.isinf(gene_sim_result_flat_raw)
        & ~np.isinf(gene_sim_true_flat_raw)
    )
    gene_sim_result_flat = gene_sim_result_flat_raw[valid_gene_mask]
    gene_sim_true_flat = gene_sim_true_flat_raw[valid_gene_mask]
    # 如果有效数据不足，或输入恒定，返回默认值
    if len(cell_sim_result_flat) == 0 or np.std(cell_sim_result_flat) == 0 or np.std(cell_sim_true_flat) == 0:
        cell_similarity_score = 0.0
    else:
        cell_similarity_score, _ = pearsonr(cell_sim_result_flat, cell_sim_true_flat)

    if len(gene_sim_result_flat) == 0 or np.std(gene_sim_result_flat) == 0 or np.std(gene_sim_true_flat) == 0:
        gene_similarity_score = 0.0
    else:
        gene_similarity_score, _ = pearsonr(gene_sim_result_flat, gene_sim_true_flat)


    return cell_similarity_score, gene_similarity_score, cell_sim_result_flat_save, gene_sim_result_flat_save


def evaluate_gat_autoencoder(model, x, A_combined, mask, mark, original_x, device, true_labels,predict=0, plot_path=None, enable_clustering=False):
    model.eval()
    with torch.no_grad():
        logits = model(x, A_combined)
        logits_final = logits * (1 - mark) + original_x * mark
        logits_cpu = logits_final.detach().cpu().numpy()
        # 初始化聚类指标
        ari = sc = nmi = None
        ami = homogeneity = completeness = vmeasure = None
        ch_score = db_score = fmi = purity = None

        if enable_clustering:
            true_labels = true_labels.iloc[:, 0].values
            # convert true labels to integer indices for some metrics
            _, true_int = np.unique(true_labels, return_inverse=True)
            n_clusters = len(np.unique(true_int))
            kmeans = KMeans(n_clusters=n_clusters, random_state=123, n_init=20)
            predicted_labels = kmeans.fit_predict(logits_cpu)

            if plot_path:
                base, ext = os.path.splitext(plot_path)
                pred_path = base + "_pred" + ext
                true_path = base + "_true" + ext

                save_clustering_plot(logits_cpu, predicted_labels, path=pred_path, title="Predicted Clustering")
                save_clustering_plot(logits_cpu, true_labels, path=true_path, title="True Label Distribution")


            ari = adjusted_rand_score(true_int, predicted_labels)
            sc = silhouette_score(logits_cpu, predicted_labels)
            nmi = normalized_mutual_info_score(true_int, predicted_labels)
            # 额外聚类指标
            try:
                ami = adjusted_mutual_info_score(true_int, predicted_labels)
            except Exception:
                ami = None
            try:
                homogeneity = homogeneity_score(true_int, predicted_labels)
            except Exception:
                homogeneity = None
            try:
                completeness = completeness_score(true_int, predicted_labels)
            except Exception:
                completeness = None
            try:
                vmeasure = v_measure_score(true_int, predicted_labels)
            except Exception:
                vmeasure = None
            try:
                ch_score = calinski_harabasz_score(logits_cpu, predicted_labels)
            except Exception:
                ch_score = None
            try:
                db_score = davies_bouldin_score(logits_cpu, predicted_labels)
            except Exception:
                db_score = None
            try:
                fmi = fowlkes_mallows_score(true_int, predicted_labels)
            except Exception:
                fmi = None
            # purity
            try:
                n_samples = len(predicted_labels)
                total_max = 0
                for lbl in np.unique(predicted_labels):
                    mask_cl = predicted_labels == lbl
                    true_subset = true_int[mask_cl]
                    if len(true_subset) == 0:
                        continue
                    counts = np.bincount(true_subset)
                    total_max += counts.max()
                purity = total_max / n_samples if n_samples > 0 else None
            except Exception:
                purity = None
            # attach extras to return vars by overwriting or adding locals
            # we'll return these by appending after existing return tuple

        mask = mask.to(device)
        if mask.dtype != torch.bool and mask.dtype != torch.long:
            mask = mask.bool()
        mask = ~mask

        masked_logits = logits[mask]
        masked_original_x = original_x[mask].to(device)

        mse = F.mse_loss(masked_logits, masked_original_x).item()
        rmse = torch.sqrt(F.mse_loss(masked_logits, masked_original_x)).item()
        mae = mean_absolute_error(masked_original_x.cpu().numpy(), masked_logits.cpu().numpy())
        numerator = torch.sum(logits_final * original_x)
        denominator = torch.norm(logits_final) * torch.norm(original_x)
        cosine_sim = (numerator / (denominator + 1e-8)).item()
        x_centered = (original_x - original_x.mean()).flatten()
        y_centered = (logits_final - logits_final.mean()).flatten()
        pcc = (torch.dot(x_centered, y_centered) /
            (torch.norm(x_centered) * torch.norm(y_centered) + 1e-8)).item()
        cell_sim=0
        gene_sim=0
        # if predict == 2:#仿真数据才计算
        #     # 计算相似度
        #     cell_sim, gene_sim, cell_sim_result_flat, gene_sim_result_flat = evaluate_similarity(
        #         logits_cpu,original_x.detach().cpu().numpy() if isinstance(original_x, torch.Tensor) else original_x)
        #     # HDF5 文件路径
        #     h5file_path = "D:/stDGA/sim/cell_sim_dgy_30%.h5"
        #     # 读取 R 保存的 h5 文件
        #     with h5py.File(h5file_path, "r") as f:
        #         data_matrix = f["cell_sim_data"][:]  # 取出矩阵
        #         colnames = [x.decode("utf-8") for x in f["colnames"][:]]  # 取出列名（字节转字符串）
        #     # 构造 DataFrame：自动判断是否需要转置，确保行数能对上 n*n
        #     expected_len = len(cell_sim_result_flat)
        #     if data_matrix.ndim != 2:
        #         raise ValueError(f"cell_sim_data 维度错误: {data_matrix.shape}")
        #     if data_matrix.shape[0] == expected_len and data_matrix.shape[1] == len(colnames):
        #         aligned_matrix = data_matrix
        #     elif data_matrix.shape[1] == expected_len and data_matrix.shape[0] == len(colnames):
        #         aligned_matrix = data_matrix.T
        #     else:
        #         file_n = int(np.sqrt(data_matrix.shape[0]))
        #         calc_n = int(np.sqrt(expected_len))
        #         raise ValueError(
        #             "细胞相似性维度不匹配："
        #             f"cell_sim_data.shape={data_matrix.shape}, colnames={len(colnames)}, "
        #             f"期望行数={expected_len}(≈{calc_n}^2)。"
        #             f"若你看到 90000，通常表示细胞数≈300(300^2=90000)。"
        #         )
        #     sim = pd.DataFrame(aligned_matrix, columns=colnames)
        #     # 添加新的一列 'stDGA'
        #     sim['stDGA'] = cell_sim_result_flat
        #     # 调整列顺序，把 'stDGA' 移到第二列
        #     cols = sim.columns.tolist()
        #     cols.insert(1, cols.pop(cols.index('stDGA')))
        #     sim = sim[cols]
        #     # 保存回 h5 文件（覆盖原文件）
        #     with h5py.File(h5file_path, "w") as f:
        #         # 保存矩阵
        #         f.create_dataset("cell_sim_data", data=sim.values)
        #         # 保存列名
        #         f.create_dataset("colnames", data=[c.encode("utf-8") for c in sim.columns])
        #         print('stDGA的细胞相似性已被写入H5文件')
            # # 基因相似性写入 H5 文件
            # gene_h5_path = "D:/stDGA/sim/gene_sim_10X_30%.h5"
            # try:
            #     with h5py.File(gene_h5_path, "r") as f:
            #         data_matrix = f["gene_sim_data"][:]
            #         colnames = [x.decode("utf-8") for x in f["colnames"][:]]
            #     # 构造 DataFrame：自动判断是否需要转置，确保行数能对上 n*n
            #     expected_len = len(gene_sim_result_flat)
            #     if data_matrix.ndim != 2:
            #         raise ValueError(f"gene_sim_data 维度错误: {data_matrix.shape}")
            #     if data_matrix.shape[0] == expected_len and data_matrix.shape[1] == len(colnames):
            #         aligned_matrix = data_matrix
            #     elif data_matrix.shape[1] == expected_len and data_matrix.shape[0] == len(colnames):
            #         aligned_matrix = data_matrix.T
            #     else:
            #         file_n = int(np.sqrt(data_matrix.shape[0]))
            #         calc_n = int(np.sqrt(expected_len))
            #         raise ValueError(
            #             "基因相似性维度不匹配："
            #             f"gene_sim_data.shape={data_matrix.shape}, colnames={len(colnames)}, "
            #             f"期望行数={expected_len}(≈{calc_n}^2)。"
            #             "\n如果你确认基因数是 2000，那么期望行数应为 4,000,000；"
            #             f"当前 H5 行数={data_matrix.shape[0]}(≈{file_n}^2)，说明生成该 H5 时用的基因子集数量不同。"
            #         )
            #     sim_gene = pd.DataFrame(aligned_matrix, columns=colnames)
            #     sim_gene['stDGA'] = gene_sim_result_flat
            #     with h5py.File(gene_h5_path, "w") as f:
            #         f.create_dataset("gene_sim_data", data=sim_gene.values)
            #         f.create_dataset("colnames", data=[c.encode("utf-8") for c in sim_gene.columns])
            #         print('stDGA的基因相似性已被写入H5文件')
            # except Exception as e:
            #     print(f'写入基因H5文件时出错: {e}')
            # with h5py.File("D:/stDGA/sim/cell_sim_52_6drop1.h5") as f:
            #     print(f['cell_sim_data'].shape)
            # with h5py.File("D:/stDGA/sim/gene_sim_52_6drop1.h5") as f:
            #     print(f['gene_sim_data'].shape)
            # print("cell_sim_result_flat:", len(cell_sim_result_flat))
            # print("gene_sim_result_flat:", len(gene_sim_result_flat))
    return cosine_sim, rmse, mae, pcc, mse, cell_sim, gene_sim, ari, sc, nmi, ami, homogeneity, completeness, vmeasure, ch_score, db_score, fmi, purity
