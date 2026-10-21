import h5py
import time
import os
import numpy as np
import pandas as pd
import torch
import scipy.sparse as sp
from scipy.sparse import csr_matrix, hstack
import scanpy as sc
import graph_func
from AnnData import prepare_graph_data


# 自动将细胞类型列统一命名为 "celltype"
def standardize_celltype_column(adata,
                                possible_names=['celltype', 'class_label', 'Cluster_Name', 'ClusterName', 'subclass_label', 'subclass',
                                                'cell_type', 'CellType', 'label']):
    print("📋 当前 obs 列名：", adata.obs.columns.tolist())
    for col in possible_names:
        if col in adata.obs.columns:
            adata = adata.copy()
            adata.obs['celltype'] = adata.obs[col]
            print(f"✅ 使用 {col} 重命名为 celltype")
            return adata
    raise ValueError("❌ 未找到可识别的细胞类型列，请检查 obs 中的列名")


# 应用到你的数据


def extract_hvg_and_split(file_path_spatial, file_path_seq, test_genes=None, val_genes=None, save_padded_path=None, n_hvg=2000):
    # 加载两个数据集
    spatial_data = sc.read_h5ad(file_path_spatial)
    seq_data = sc.read_h5ad(file_path_seq)
    spatial_data = standardize_celltype_column(spatial_data)
    seq_data = standardize_celltype_column(seq_data)

    # 提取共有基因并根据其子集重构数据
    shared_genes = spatial_data.var_names.intersection(seq_data.var_names)
    if n_hvg is not None and len(shared_genes) > int(n_hvg):
        # 以方差作为稳健的高变基因近似指标（避免不同归一化流程导致的 HVG 不稳定）
        shared_genes = np.array(shared_genes)
        st_x = spatial_data[:, shared_genes].X
        sc_x = seq_data[:, shared_genes].X
        st_x = st_x.toarray() if sp.issparse(st_x) else np.asarray(st_x)
        sc_x = sc_x.toarray() if sp.issparse(sc_x) else np.asarray(sc_x)
        st_var = np.nan_to_num(np.var(st_x, axis=0), nan=0.0, posinf=0.0, neginf=0.0)
        sc_var = np.nan_to_num(np.var(sc_x, axis=0), nan=0.0, posinf=0.0, neginf=0.0)
        merged_var = st_var + sc_var
        top_idx = np.argsort(-merged_var)[:int(n_hvg)]
        shared_genes = shared_genes[top_idx]
        print(f"✅ 已按方差选取 HVG: {len(shared_genes)}")
    spatial_data = spatial_data[:, shared_genes]
    seq_data = seq_data[:, shared_genes]

    # 打印基因数量用于后续判断
    n_shared_genes = len(shared_genes)
    print(f"共有基因数量: {n_shared_genes}")
    matching_seq_data = seq_data[:, shared_genes]
    A_combined = prepare_graph_data(
        spatial_data,
        seq_data,
        matching_seq_data,
        use_logic='or',
        top_k1=50,
        top_k2=50,
    )

    n_cells = spatial_data.n_obs
    n_genes = spatial_data.n_vars

    # 初始化掩码矩阵：全1
    val_mask_matrix = np.ones((n_cells, n_genes), dtype=np.float32)
    test_mask_matrix = np.ones((n_cells, n_genes), dtype=np.float32)

    # 基因名 -> 索引映射
    gene_name_to_idx = {gene: idx for idx, gene in enumerate(spatial_data.var_names)}

    # 判断并应用掩码逻辑
    if n_shared_genes <= 50:
        for gene in val_genes:
            if gene in gene_name_to_idx:
                idx = gene_name_to_idx[gene]
                val_mask_matrix[:, idx] = 0
        test_genes = []  # 无测试集
    else:
        for gene in val_genes:
            if gene in gene_name_to_idx:
                idx = gene_name_to_idx[gene]
                val_mask_matrix[:, idx] = 0
        if test_genes is not None:
            for gene in test_genes:
                if gene in gene_name_to_idx:
                    idx = gene_name_to_idx[gene]
                    test_mask_matrix[:, idx] = 0

    # 构造表达矩阵和掩码
    mark_train = val_mask_matrix * test_mask_matrix
    X_full_share = spatial_data.X
    X_ref = seq_data.X

    X_full = pad_columns_to_match(X_full_share, X_ref)
    X_impute_combine = concatenate_matrices(X_full, X_ref, axis=0)

    if 'spatial' in spatial_data.obsm:
        spatial_coords = spatial_data.obsm['spatial']
    else:
        spatial_coords = spatial_data.obs[['X', 'Y']].values

    X_labels = spatial_data.obs['celltype']
    X_mark = (X_full != 0).astype(int)
    ref_mask = csr_matrix(np.ones_like(X_ref))

    X_val_mask = concatenate_matrices(val_mask_matrix, ref_mask)
    X_test_mask = concatenate_matrices(test_mask_matrix, ref_mask)
    X_mark_combine = concatenate_matrices(X_mark, ref_mask)  # 标记本身有表达值的位置
    X_mask_all = concatenate_matrices(mark_train, ref_mask)  # 标记验证集和测试集总的掩码

    print("A_combined shape:", A_combined.shape)
    print("spatial_coords shape:", spatial_coords.shape)
    print("X_full_share shape:", X_full_share.shape)
    print("X_impute_combine shape:", X_impute_combine.shape)
    print("X_val_mask shape:", X_val_mask.shape)
    print("X_test_mask shape:", X_test_mask.shape)
    print("X_mask_all shape:", X_mask_all.shape)
    print("X_mark_combine shape:", X_mark_combine.shape)
    print("X_labels shape:", X_labels.shape)

    # 如果请求，保存填充后的矩阵到指定目录（使用 scipy.sparse.save_npz）
    if save_padded_path is not None:
        try:
            os.makedirs(save_padded_path, exist_ok=True)
            # 保存稀疏矩阵
            sp.save_npz(os.path.join(save_padded_path, "X_full_padded.npz"), X_full)
            sp.save_npz(os.path.join(save_padded_path, "X_impute_combine.npz"), X_impute_combine)
            sp.save_npz(os.path.join(save_padded_path, "X_val_mask.npz"), X_val_mask)
            sp.save_npz(os.path.join(save_padded_path, "X_test_mask.npz"), X_test_mask)
            sp.save_npz(os.path.join(save_padded_path, "X_mask_all.npz"), X_mask_all)
            sp.save_npz(os.path.join(save_padded_path, "X_mark_combine.npz"), X_mark_combine)
            # labels and spatial coords 可以用 numpy 保存
            np.savez_compressed(os.path.join(save_padded_path, "meta.npz"), spatial_coords=spatial_coords,
                                labels=np.array(X_labels))
            print(f"✅ 已保存填充结果到: {save_padded_path}")
        except Exception as e:
            print(f"⚠️ 保存填充结果失败: {e}")

    return A_combined, spatial_coords, X_full_share, X_impute_combine, X_val_mask, X_test_mask, X_mask_all, X_mark_combine, X_labels


import numpy as np
from scipy.sparse import csr_matrix, vstack, hstack

# Module-level timing accumulator for padding/concatenation operations
_padding_time_seconds = 0.0


def get_padding_time_seconds():
    """Return total time (seconds) spent inside padding/concatenation helpers."""
    return _padding_time_seconds


def reset_padding_time_seconds():
    global _padding_time_seconds
    _padding_time_seconds = 0.0


def concatenate_matrices(X_fill, X_reference, fill_value=0, axis=0):
    """
    按指定维度拼接两个稀疏矩阵，自动补齐列数或行数。
    - axis=0：纵向拼接（行数相加）
    - axis=1：横向拼接（列数相加）
    """
    global _padding_time_seconds
    t0 = time.time()
    X_fill = csr_matrix(X_fill) if not isinstance(X_fill, csr_matrix) else X_fill
    X_reference = csr_matrix(X_reference) if not isinstance(X_reference, csr_matrix) else X_reference

    if axis == 0:
        # 补齐列数
        if X_fill.shape[1] < X_reference.shape[1]:
            padding = csr_matrix((X_fill.shape[0], X_reference.shape[1] - X_fill.shape[1]))
            X_fill = hstack([X_fill, padding], format='csr')
        elif X_fill.shape[1] > X_reference.shape[1]:
            padding = csr_matrix((X_reference.shape[0], X_fill.shape[1] - X_reference.shape[1]))
            X_reference = hstack([X_reference, padding], format='csr')
        res = vstack([X_fill, X_reference], format='csr')
        _padding_time_seconds += (time.time() - t0)
        return res
    elif axis == 1:
        res = hstack([X_fill, X_reference], format='csr')
        _padding_time_seconds += (time.time() - t0)
        return res
    else:
        raise ValueError("axis must be 0 or 1")


def pad_columns_to_match(X_full, X_ref):
    """
    补齐 X_full 的列数以匹配 X_ref。
    """
    global _padding_time_seconds
    t0 = time.time()
    X_full = csr_matrix(X_full) if not isinstance(X_full, csr_matrix) else X_full
    X_ref = csr_matrix(X_ref) if not isinstance(X_ref, csr_matrix) else X_ref

    if X_full.shape[1] < X_ref.shape[1]:
        zero_padding = csr_matrix((X_full.shape[0], X_ref.shape[1] - X_full.shape[1]))
        X_full = hstack([X_full, zero_padding], format='csr')

    _padding_time_seconds += (time.time() - t0)
    return X_full

# def expand_spatial(spatial_data, X_reference):
#     """
#     扩展空间信息：将空间坐标扩展到拼接后的大矩阵中。
#     对于参考矩阵没有空间信息的位置，填充[0, 0]。
#     """
#     # 获取参考矩阵的行数
#     n_ref = X_reference.shape[0]

#     # 创建一个新的空间信息矩阵，大小为 (spatial_data 行数 + X_reference 行数, 2)
#     rows = len(spatial_data.iloc[:, 0])
#     spatial_combined = np.zeros((rows + n_ref, 2), dtype=spatial_data.values.dtype)

#     # 将原始空间数据填充到左上角（空间数据包含x和y坐标）
#     spatial_combined[:spatial_data.shape[0], :] = spatial_data.to_numpy()

#     # 对于参考矩阵的行数部分，填充[0, 0]
#     spatial_combined[spatial_data.shape[0]:, :] = 0

#     return pd.DataFrame(spatial_combined, columns=spatial_data.columns)


# import numpy as np
# import pandas as pd

# def expand_labels(labels_data, labels_ref):
#     # 获取参考矩阵的行数（即 X_reference 的行数）
#     n_ref = labels_ref.shape[0]
#     # 将 CategoricalDtype 转换为 object 类型
#     dtype = object if pd.api.types.is_categorical_dtype(labels_data) else labels_data.values.dtype
#     labels_combined = np.zeros((labels_data.shape[0] + n_ref, 1), dtype=dtype)

#     # 将标签数据填充到标签矩阵的前部分
#     labels_combined[:labels_data.shape[0], 0] = labels_data.to_numpy().flatten()  # 确保将标签数据转换为一维数组
#     labels_combined[labels_data.shape[0]:, 0] = labels_ref.to_numpy().flatten()

#     # 返回一个 Pandas DataFrame，保留原始列名
#     # 直接指定列名
#     return pd.DataFrame(labels_combined, columns=['Celltype'])


# from scipy.sparse import csr_matrix, vstack


# import numpy as np
# from scipy.sparse import csr_matrix


# def generate_random_mask(adata, mask_ratio):
#     """
#     根据设置的掩码率，随机选择一部分基因（列），将它们的所有表达值设置为0。
#     - adata: AnnData 对象，包含基因表达矩阵
#     - mask_ratio: 掩码比例，即要掩码的基因列的比例
#     """
#     X = adata.X  # 获取基因表达矩阵
#     num_genes = X.shape[1]  # 获取基因数量（列数）

#     # 计算掩码的数量，基于掩码比例
#     num_masked_genes = int(num_genes * mask_ratio)  # 计算要掩码的基因列数

#     # 随机选择要掩码的基因列索引
#     mask_indices = np.random.choice(num_genes, num_masked_genes, replace=False)

#     # 创建一个掩码矩阵，所有列默认是1，表示保留
#     mask = np.ones(X.shape, dtype=int)

#     # 将选择的基因列的掩码值设置为0，表示掩码
#     mask[:, mask_indices] = 0

#     # 创建掩码后的基因表达矩阵，选择的列的值将变为0
#     X_mask = X.multiply(mask)  # 使用乘法应用掩码

#     # 返回掩码和掩码后的基因表达矩阵
#     return mask, X_mask


