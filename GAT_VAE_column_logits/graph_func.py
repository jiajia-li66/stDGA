import numpy as np
import scipy.sparse as sp
import torch
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import GridSearchCV
import pandas as pd

# 1. 基于空间位置构造 KNN 邻接矩阵
from sklearn.neighbors import NearestNeighbors


def generate_adj_mat(adata, k=10, include_self=False):
    if 'spatial' in adata.obsm:
        coords = adata.obsm['spatial']
    elif {'X', 'Y'}.issubset(adata.obs.columns):
        coords = adata.obs[['X', 'Y']].values
    else:
        raise ValueError("空间坐标不存在")

    n = coords.shape[0]

    # 不能超过总数 - 1（除非 include_self）
    if include_self:
        n_neighbors = min(k, n)
    else:
        n_neighbors = min(k + 1, n)

    nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(coords)
    distances, indices = nbrs.kneighbors(coords)

    if not include_self:
        indices = indices[:, 1:]

    actual_k = indices.shape[1]

    rows = np.repeat(np.arange(n), actual_k)
    cols = indices.reshape(-1)
    data = np.ones_like(cols, dtype=np.float32)

    assert len(rows) == len(cols) == len(data), f"行数: {len(rows)}, 列数: {len(cols)}"

    adj = sp.coo_matrix((data, (rows, cols)), shape=(n, n))
    adj = adj.maximum(adj.T)  # 转为无向图
    return adj


def construct_adj_matrix_from_seurat(adata):
    # 获取聚类标签
    labels = adata.obs['celltype'].values
    n_cells = len(labels)

    # 初始化邻接矩阵
    adj_mat = np.zeros((n_cells, n_cells), dtype=np.float32)

    # 遍历每个簇，构建邻接矩阵
    for label in set(labels):
        cluster_indices = np.where(labels == label)[0]
        for i in cluster_indices:
            adj_mat[i, cluster_indices] = 1

    # 返回稀疏矩阵格式的邻接矩阵
    return sp.csr_matrix(adj_mat)


def gene_expression_adj(adata, similarity_metric='cosine', threshold=0.5):
    # 提取表达矩阵，确保为 numpy array
    gene_expr = adata.X.toarray() if not isinstance(adata.X, np.ndarray) else adata.X

    # 计算相似度矩阵
    if similarity_metric == 'cosine':
        sim_matrix = 1 - pairwise_distances(gene_expr, metric='cosine')
    elif similarity_metric == 'euclidean':
        dist = pairwise_distances(gene_expr, metric='euclidean')
        sim_matrix = 1 / (1 + dist)  # 转为相似度
    elif similarity_metric == 'pearson':
        sim_matrix = np.corrcoef(gene_expr)
    else:
        raise ValueError(f"不支持的相似性度量: {similarity_metric}")

    # 阈值处理：>= threshold 为1，其余为0
    binary_adj = (sim_matrix >= threshold).astype(np.float32)

    # 构造稀疏矩阵
    return sp.csr_matrix(binary_adj)


def generate_mask(adata):
    data = adata.X  # 获取数据
    # 创建 mask，标记数据中为 0 的位置
    mask = (data == 0).astype(np.float32)
    # 转换为 PyTorch 张量
    mask = torch.tensor(mask, dtype=torch.float32)
    return mask


# 5. 将稀疏矩阵转换为 PyTorch 张量（支持 GPU 加速）
def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape)


# 图拉普拉斯平滑和归一化
def preprocess_graph(adj):
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    # 检查 rowsum 中的值，避免出现零值
    rowsum[rowsum == 0] = 1e-10  # 用一个很小的正数替代 0
    # 处理零值
    epsilon = 1e-8
    rowsum = np.where(rowsum == 0, epsilon, rowsum)
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_mx_to_torch_sparse_tensor(adj_normalized)
    # return sparse_mx_to_torch_sparse_tensor(adj_normalized)


# GPU 加速计算
def enable_gpu_support(adj):
    if torch.cuda.is_available():
        return adj.cuda()
    else:
        return adj


import numpy as np
import scipy.sparse as sp
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr


def keep_topk_per_row_sparse(mat, top_k):
    """对每一行仅保留 top_k 大权重边（稀疏矩阵）。"""
    if top_k is None or top_k <= 0:
        return mat
    mat = mat.tocsr()
    n_rows, n_cols = mat.shape
    top_k = int(min(top_k, n_cols))
    rows, cols, vals = [], [], []
    for i in range(n_rows):
        start, end = mat.indptr[i], mat.indptr[i + 1]
        row_cols = mat.indices[start:end]
        row_vals = mat.data[start:end]
        if row_vals.size == 0:
            continue
        if row_vals.size > top_k:
            keep_idx = np.argpartition(row_vals, -top_k)[-top_k:]
            row_cols = row_cols[keep_idx]
            row_vals = row_vals[keep_idx]
        rows.extend([i] * row_vals.size)
        cols.extend(row_cols.tolist())
        vals.extend(row_vals.tolist())
    return sp.csr_matrix((vals, (rows, cols)), shape=mat.shape, dtype=mat.dtype)


def generate_combined_adj_matrix(spatial_adj, seq_adj, spatial_data, matching_seq_data):
    """
    生成一个大的邻接矩阵，维度是 (spatial_cells + seq_cells, spatial_cells + seq_cells)
    - spatial_adj: 空间数据计算的邻接矩阵
    - seq_adj: 序列数据计算的邻接矩阵
    - spatial_data: 空间细胞的基因表达数据，shape=(spatial_cells, num_genes)
    - seq_data: 序列细胞的基因表达数据，shape=(seq_cells, num_genes)
    """
    spatial_cells = spatial_adj.shape[0]
    seq_cells = seq_adj.shape[0]

    print("spatial_data shape:", spatial_data.X.shape)
    print("matching_seq_data shape:", matching_seq_data.X.shape)

    # 计算空间细胞与序列细胞之间的 Pearson 相关系数
    inter_connection = calculate_inter_cell_similarity(spatial_data.X, matching_seq_data.X)

    # 拼接邻接矩阵
    top_left = spatial_adj  # 空间数据邻接矩阵
    top_right = inter_connection  # 空间与序列之间的连接
    print("st和sc的pcc:", inter_connection.shape)
    bottom_left = inter_connection.T  # 序列与空间之间的连接（对称矩阵）
    bottom_right = seq_adj  # 序列数据邻接矩阵

    # 将这些矩阵拼接起来
    upper = sp.hstack([top_left, top_right])
    lower = sp.hstack([bottom_left, bottom_right])
    combined_adj = sp.vstack([upper, lower])  # 最终合并成一个大的邻接矩阵

    return sp.csr_matrix(combined_adj)


def calculate_inter_cell_similarity(spatial_data, seq_data):
    # 稀疏矩阵直接 np.array 会变成 object 标量，必须显式转为 dense
    if sp.issparse(spatial_data):
        spatial_data = spatial_data.toarray()
    else:
        spatial_data = np.asarray(spatial_data, dtype=np.float32)
    if sp.issparse(seq_data):
        seq_data = seq_data.toarray()
    else:
        seq_data = np.asarray(seq_data, dtype=np.float32)

    spatial_data = spatial_data.astype(np.float32, copy=False)
    seq_data = seq_data.astype(np.float32, copy=False)

    # 行标准化（避免 std=0 导致 NaN）
    spatial_std = spatial_data.std(axis=1, keepdims=True)
    seq_std = seq_data.std(axis=1, keepdims=True)
    spatial_std = np.where(spatial_std < 1e-8, 1.0, spatial_std)
    seq_std = np.where(seq_std < 1e-8, 1.0, seq_std)
    spatial_data = (spatial_data - spatial_data.mean(axis=1, keepdims=True)) / spatial_std
    seq_data = (seq_data - seq_data.mean(axis=1, keepdims=True)) / seq_std

    # Pearson-like 相似度
    sim = np.dot(spatial_data, seq_data.T) / max(spatial_data.shape[1], 1)

    # 仅保留每个空间细胞最相近的 top-k 序列细胞，防止跨模态近乎全连接
    k = int(min(20, sim.shape[1]))
    if k <= 0:
        return sp.csr_matrix(sim.shape, dtype=np.float32)
    topk_idx = np.argpartition(sim, -k, axis=1)[:, -k:]
    rows = np.repeat(np.arange(sim.shape[0]), k)
    cols = topk_idx.reshape(-1)
    vals = sim[np.arange(sim.shape[0])[:, None], topk_idx].reshape(-1)
    # 仅保留正相关连接
    pos_mask = vals > 0
    rows = rows[pos_mask]
    cols = cols[pos_mask]
    vals = vals[pos_mask]
    inter_connection = sp.coo_matrix((vals, (rows, cols)), shape=sim.shape, dtype=np.float32)
    return inter_connection.tocsr()

