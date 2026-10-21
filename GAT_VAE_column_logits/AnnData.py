from turtle import pd
import numpy as np
import scanpy as sc
import h5py
import graph_func
import torch
from scipy.sparse import block_diag
import scipy.sparse as sp


def AnnData_generate(gene_expr, spatial_coords, labels):
    # 创建 AnnData 对象
    adata = sc.AnnData(X=gene_expr)
    print(f"空间坐标的列名：{spatial_coords.columns}")
    print(f"标签的列名：{labels.columns}")
    # 直接使用行索引对齐数据
    adata.obs_names = [f"cell_{i}" for i in range(gene_expr.shape[0])]  # 创建默认的 obs_names
    print(f"adata.obs_names: {adata.obs_names[:5]}")
    # 添加空间坐标到 adata.obsm
    # 修改为使用字符串列名选取列
    adata.obsm['spatial'] = spatial_coords[['X', 'Y']].values
    print("空间坐标已添加到 AnnData 对象中的 .obsm。")
    adata.obs['cluster'] = labels['Celltype'].values
    print("聚类标签已添加到 AnnData 对象中的 .obs。")
    return adata


import torch
import numpy as np
import scipy.sparse as sp
from scipy.sparse import bmat


def binarize_adj(adj):
    return (adj != 0).astype(np.int8)


def prepare_graph_data(
        st_adata, sc_adata, matching_seq_data,
        use_logic='and',  # 'and' or 'or'
        top_k1=50,
        top_k2=50,
):
    st_expr = binarize_adj(graph_func.gene_expression_adj(st_adata))
    seq_expr = binarize_adj(graph_func.gene_expression_adj(sc_adata))
    st_cluster = binarize_adj(graph_func.construct_adj_matrix_from_seurat(st_adata))
    sc_cluster = binarize_adj(graph_func.construct_adj_matrix_from_seurat(sc_adata))
    st_spatial = binarize_adj(graph_func.generate_adj_mat(st_adata))

    A_expr = binarize_adj(graph_func.generate_combined_adj_matrix(st_expr, seq_expr, st_adata, matching_seq_data))
    A_cluster = binarize_adj(prepare_graph_cluster(st_adata, sc_adata))
    A_spatial = pad_sparse_matrix(st_spatial, A_cluster)

    A_spatial = A_spatial.tocsr()
    A_cluster = A_cluster.tocsr()
    A_expr = A_expr.tocsr()

    n_st = st_adata.shape[0]
    n_sc = sc_adata.shape[0]

    A_expr_st = A_expr[:n_st, :n_st]
    A_expr_sc = A_expr[n_st:, n_st:]
    A_expr_cross_st2sc = A_expr[:n_st, n_st:]
    A_expr_cross_sc2st = A_expr[n_st:, :n_st]

    # 边裁剪策略：
    # 1) st-st 保留全部（不裁剪）
    # 2) sc-sc 仅保留 top_k1
    # 3) st-sc 仅保留 top_k2（按 st->sc 行裁剪，并保持反向对称）
    A_expr_sc = graph_func.keep_topk_per_row_sparse(A_expr_sc, top_k1)
    A_expr_cross_st2sc = graph_func.keep_topk_per_row_sparse(A_expr_cross_st2sc, top_k2)
    A_expr_cross_sc2st = A_expr_cross_st2sc.T.tocsr()
    cross_density = A_expr_cross_st2sc.nnz / max(A_expr_cross_st2sc.shape[0] * A_expr_cross_st2sc.shape[1], 1)
    print(f"📌 跨模态连接密度: {cross_density:.6f}, top_k1={top_k1}, top_k2={top_k2}")

    A_spatial_st = A_spatial[:n_st, :n_st]
    A_cluster_st = A_cluster[:n_st, :n_st]

    if use_logic == 'or':
        A_st_combined = ((A_spatial_st + A_cluster_st + A_expr_st) != 0).astype(np.int8).tocoo()
        A_sc_combined = ((A_cluster[n_st:, n_st:] + A_expr_sc) != 0).astype(np.int8).tocoo()
    elif use_logic == 'and':
        A_st_combined = A_spatial_st.multiply(A_cluster_st).multiply(A_expr_st).tocoo()
        A_sc_combined = A_cluster[n_st:, n_st:].multiply(A_expr_sc).tocoo()
    else:
        raise ValueError("use_logic must be 'or' or 'and'")

    # 如果 "and" 交集图过稀，会导致消息传递几乎失效，自动退回到 "or"
    if use_logic == 'and':
        st_density = A_st_combined.nnz / max(A_st_combined.shape[0] * A_st_combined.shape[1], 1)
        sc_density = A_sc_combined.nnz / max(A_sc_combined.shape[0] * A_sc_combined.shape[1], 1)
        if st_density < 1e-4 or sc_density < 1e-4:
            print(f"⚠️ 'and' 图过稀 (st={st_density:.2e}, sc={sc_density:.2e})，自动切换为 'or'。")
            A_st_combined = ((A_spatial_st + A_cluster_st + A_expr_st) != 0).astype(np.int8).tocoo()
            A_sc_combined = ((A_cluster[n_st:, n_st:] + A_expr_sc) != 0).astype(np.int8).tocoo()

    A_final = bmat([
        [A_st_combined, A_expr_cross_st2sc],
        [A_expr_cross_sc2st, A_sc_combined]
    ], format='coo')

    row = np.array(A_final.row)
    col = np.array(A_final.col)
    indices = torch.from_numpy(np.vstack((row, col))).long()
    values = torch.tensor(A_final.data, dtype=torch.float32)
    A_final_torch = torch.sparse_coo_tensor(indices, values, A_final.shape).coalesce()

    # ✅ 检查是否为01矩阵
    unique_values = torch.unique(A_final_torch.values())
    print("🔍 A_final_torch.values() 中的唯一值:", unique_values)

    if torch.all((unique_values == 0) | (unique_values == 1)):
        print("✅ A_final_torch 是 01 矩阵（布尔结构）")
    else:
        print("⚠️ A_final_torch 中存在非 0/1 的值")

    return A_final_torch


def prepare_graph_cluster(st_adata, sc_adata):
    # 构造邻接矩阵
    A_cluster_st = graph_func.construct_adj_matrix_from_seurat(st_adata)
    A_cluster_sc = graph_func.construct_adj_matrix_from_seurat(sc_adata)

    # 对角拼接两个邻接矩阵（返回一个新的稀疏矩阵）
    A_combined = block_diag([A_cluster_st, A_cluster_sc], format='csr')

    return A_combined


import scipy.sparse as sp


def pad_sparse_matrix(mat, target):
    """
    将稀疏矩阵 mat 补零扩展到 target_shape 的维度
    """
    cur_shape = mat.shape
    target_shape = target.shape
    pad_bottom = target_shape[0] - cur_shape[0]
    pad_right = target_shape[1] - cur_shape[1]

    if pad_bottom > 0:
        bottom_padding = sp.csr_matrix((pad_bottom, cur_shape[1]))
        mat = sp.vstack([mat, bottom_padding])

    if pad_right > 0:
        right_padding = sp.csr_matrix((mat.shape[0], pad_right))
        mat = sp.hstack([mat, right_padding])

    return mat
