from turtle import pd
import numpy as np
import scanpy as sc
import h5py
import graph_func
import torch


def AnnData_generate(gene_expr, spatial_coords, labels):
    # 创建 AnnData 对象
    adata = sc.AnnData(X=gene_expr.astype(np.float32), dtype=np.float32)
    # 直接使用行索引对齐数据
    adata.obs_names = [f"cell_{i}" for i in range(len(gene_expr))]  # 创建默认的 obs_names
    # 添加空间坐标到 adata.obsm
    adata.obsm['spatial'] = spatial_coords[['0', '1']].values          #空间位置坐标0=x,1=y
    # print("空间坐标已添加到 AnnData 对象中的 .obsm。")
    adata.obs["ClusterLabel"] = labels.values
    # print("聚类标签已添加到 AnnData 对象中的 .obs。")
    return adata


def prepare_graph_data(adata):
    import torch

    # 生成邻接矩阵（均为 0-1 稀疏图）
    A_spatial = graph_func.generate_adj_mat(adata)
    A_cluster = graph_func.construct_adj_matrix_from_seurat(adata)
    A_expr = graph_func.gene_expression_adj(adata)

    def to_sparse_tensor(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
        indices = torch.tensor([sparse_mx.row, sparse_mx.col], dtype=torch.int64)
        values = torch.tensor(sparse_mx.data, dtype=torch.float32)
        return torch.sparse_coo_tensor(indices, values, sparse_mx.shape).coalesce()

    A_spatial = to_sparse_tensor(A_spatial)
    A_cluster = to_sparse_tensor(A_cluster)
    A_expr = to_sparse_tensor(A_expr)

    return A_spatial, A_cluster, A_expr



