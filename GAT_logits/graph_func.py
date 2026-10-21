import numpy as np
import scipy.sparse as sp
import torch
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import GridSearchCV
import pandas as pd


def generate_adj_mat(adata, k=10, include_self=False):
    from sklearn.metrics import pairwise_distances
    from scipy.sparse import lil_matrix

    assert 'spatial' in adata.obsm, 'AnnData object should provide spatial information'

    coords = adata.obsm['spatial']
    dist = pairwise_distances(coords)  # shape: (n_cells, n_cells)

    n = dist.shape[0]
    adj = lil_matrix((n, n), dtype=np.float32)

    for i in range(n):
        # 获取当前节点的距离向量
        d = dist[i]

        # 排除自身（如果不包含）
        if not include_self:
            d[i] = np.inf

        # 获取前 k 个最小值的索引（邻居）
        knn_idx = np.argpartition(d, kth=k)[:k]

        # 设置邻接为1
        adj[i, knn_idx] = 1.0

    return adj.tocsr()




def construct_adj_matrix_from_seurat(adata):
    # 获取空间坐标
    spatial_coords = adata.obsm['spatial']
    # 获取聚类标签
    labels = adata.obs['ClusterLabel'].values  
    n_cells = len(spatial_coords)

    # 初始化邻接矩阵
    adj_mat = np.zeros((n_cells, n_cells), dtype=np.float32)

    # 遍历每个簇，构建邻接矩阵
    for label in set(labels):
        cluster_indices = np.where(labels == label)[0]
        for i in cluster_indices:
            adj_mat[i, cluster_indices] = 1

    # 返回稀疏矩阵格式的邻接矩阵
    return sp.csr_matrix(adj_mat)


from sklearn.metrics import pairwise_distances
import numpy as np
import scipy.sparse as sp

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
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_mx_to_torch_sparse_tensor(adj_normalized)
    #return sparse_mx_to_torch_sparse_tensor(adj_normalized)


# GPU 加速计算
def enable_gpu_support(adj):
    if torch.cuda.is_available():
        return adj.cuda()
    else:
        return adj

