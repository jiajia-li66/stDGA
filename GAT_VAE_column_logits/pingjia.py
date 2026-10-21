# == 引入库 ==
import torch
import numpy as np
import random
import scipy.sparse as sp
import scanpy as sc
import pandas as pd
import os

from my_pingjia import evaluate_gat_autoencoder
from data_prepare import extract_hvg_and_split


# =========================
# 1. 基础设置：不训练，只读取 npz
# =========================
device = "cpu"

# 建议固定随机种子，保证 train / val / test 基因划分尽量一致
SEED = 123
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)

file_path_spatial = "D:/stDGA/data/data_column/starmap/starmap_spatial_processed.h5ad"
file_path_seq = "D:/stDGA/data/data_column/starmap/VISp_RNA_processed.h5ad"

save_dir = "D:/stDGA/data/data_column/result_column/stDGA"
plot_path = "D:/stDGA/data/data_column/result_column/stDGA"

os.makedirs(save_dir, exist_ok=True)
os.makedirs(plot_path, exist_ok=True)

# 这里是之前 evaluate_gat_autoencoder 保存出来的填充结果
filled_npz_path = os.path.join(save_dir, "imputed_spatial_filled_stDGA.npz")

if not os.path.exists(filled_npz_path):
    raise FileNotFoundError(f"没有找到填充结果 npz 文件：{filled_npz_path}")


# =========================
# 2. 读取原始数据
# =========================
spatial_data = sc.read_h5ad(file_path_spatial)
seq_data = sc.read_h5ad(file_path_seq)

print("seq_data shape:", seq_data.shape)
print("seq_data.obs head:")
print(seq_data.obs.head())
print("seq_data.obs columns:")
print(seq_data.obs.columns)

shared_genes = np.array(spatial_data.var_names.intersection(seq_data.var_names))
n_genes = len(shared_genes)
print(f"共有基因数量: {n_genes}")

use_test = n_genes > 50
use_val = True


# =========================
# 3. 重新构建基因划分
#    注意：这里必须和之前生成 npz 时的划分逻辑一致
# =========================
if n_genes <= 50:
    n_train = int(n_genes * 0.8)
    idx_train = np.random.choice(range(n_genes), n_train, replace=False)
    idx_val = sorted(set(range(n_genes)) - set(idx_train))

    train_genes = shared_genes[idx_train]
    val_genes = shared_genes[idx_val]
    test_genes = []

else:
    n_train = int(n_genes * 0.7)
    n_val = int(n_genes * 0.15)

    idx_perm = np.random.permutation(range(n_genes))
    idx_train = idx_perm[:n_train]
    idx_val = idx_perm[n_train:n_train + n_val]
    idx_test = sorted(set(range(n_genes)) - set(idx_train) - set(idx_val))

    train_genes = shared_genes[idx_train]
    val_genes = shared_genes[idx_val]
    test_genes = shared_genes[idx_test]

    print("测试基因：")
    print(test_genes)


# =========================
# 4. 调用 data_prepare，生成 X_full / mask / labels / spatial_coords
#    这里不训练，只是为了拿到评价所需对象
# =========================
tup = extract_hvg_and_split(
    file_path_spatial,
    file_path_seq,
    test_genes if use_test else [],
    val_genes,
    save_padded_path=os.path.join(save_dir, "padded_results"),
    n_hvg=2000,
)

A_combined, spatial_coords, X_full, X_input, val_mask, test_mask, full_mask, mark, labels = tup

tensors = [X_full, X_input, val_mask, test_mask, full_mask, mark]
X_full, X_input, val_mask, test_mask, full_mask, mark = [
    torch.tensor(t.toarray() if sp.issparse(t) else t, dtype=torch.float32).to(device)
    for t in tensors
]


# =========================
# 5. 直接读取 npz 里的填充结果作为 X_imputed
# =========================
npz_data = np.load(filled_npz_path)

print("npz keys:", npz_data.files)

if "imputed_X" in npz_data.files:
    X_imputed_np = npz_data["imputed_X"]
else:
    # 防止有些 npz 文件第一个 key 不是 imputed_X
    first_key = npz_data.files[0]
    print(f"未找到 imputed_X，使用第一个 key: {first_key}")
    X_imputed_np = npz_data[first_key]

X_imputed_np = np.asarray(X_imputed_np, dtype=np.float32)

print("X_full shape:", tuple(X_full.shape))
print("X_imputed from npz shape:", X_imputed_np.shape)

if X_imputed_np.shape[0] != X_full.shape[0]:
    raise ValueError(
        f"细胞数量不一致：X_imputed 有 {X_imputed_np.shape[0]} 行，"
        f"但 X_full 有 {X_full.shape[0]} 行。请检查 npz 是否对应当前数据。"
    )

if X_imputed_np.shape[1] < X_full.shape[1]:
    raise ValueError(
        f"基因数量不够：X_imputed 有 {X_imputed_np.shape[1]} 列，"
        f"但 X_full 至少需要 {X_full.shape[1]} 列用于评价。"
    )

X_imputed = torch.tensor(X_imputed_np, dtype=torch.float32).to(device)


# =========================
# 6. 统一调用 evaluate_gat_autoencoder 画图 + 计算指标
# =========================
final_mask = test_mask if use_test else val_mask

plot_file = "cluster_result_test_starmap_from_npz.png" if use_test else "cluster_result_val_starmap_from_npz.png"

per_gene_df, extended_metrics = evaluate_gat_autoencoder(
    X_imputed,
    X_full,
    final_mask,
    mark,
    labels,
    spatial_coords,
    plot_path=os.path.join(plot_path, plot_file),
    only_clustering=(n_genes <= 50),
    save_filled_path=None,   # 这里已经是读取 npz，不需要再保存一次
)


# =========================
# 7. 输出结果
# =========================
print("✅ 评估完成，指标如下：")

if isinstance(per_gene_df, pd.DataFrame) and not per_gene_df.empty:
    print("📊 Per-gene Evaluation Summary:")
    print(per_gene_df.describe().loc[["mean", "std", "25%", "50%", "75%"]].round(6))

print("\n🌍 Flattened Global Metrics:")
for k, v in extended_metrics["global_metrics"].items():
    try:
        print(f"{k:<20}: {v:.6f}")
    except Exception:
        print(f"{k:<20}: {v}")