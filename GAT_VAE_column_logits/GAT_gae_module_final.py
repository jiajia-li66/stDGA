# == 引入库 ==
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GATConv
import torch.nn.functional as F
import numpy as np
import random
import scipy.sparse as sp
import scanpy as sc
import pandas as pd
import os
from collections import defaultdict
from my_pingjia import evaluate_gat_autoencoder
from data_prepare import extract_hvg_and_split

# == 固定随机种子 ==
seed = 123
torch.cuda.empty_cache()
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

class GATAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads, num_layers):
        super(GATAutoencoder, self).__init__()
        self.gat_encoders = nn.ModuleList()
        for _ in range(num_layers):
            self.gat_encoders.append(GATConv(input_dim, hidden_dim, heads=num_heads, dropout=0.6))
            input_dim = hidden_dim * num_heads
        self.linear_decoders = nn.ModuleList()
        for _ in range(num_layers):
            self.linear_decoders.append(nn.Linear(input_dim, hidden_dim))
            input_dim = hidden_dim
        self.output_layer = nn.Linear(input_dim, output_dim)
        self.alpha_weights = None

    def forward(self, x, A_combined, print_once=False):
        # `A_combined` here is expected to be an `edge_index` (COO) tensor for GATConv
        if print_once:
            try:
                print("▶️ forward 启动，x:", x.shape, "edge_index:", A_combined.shape)
            except Exception:
                print("▶️ forward 启动，x:", x.shape, "edge_index: (unknown shape)")
        for encoder in self.gat_encoders:
            x = encoder(x, A_combined)
        for decoder in self.linear_decoders:
            x = F.relu(decoder(x))
        return self.output_layer(x)


def masked_mse_loss(pred, target, mask):
    pred = pred[:target.shape[0], :target.shape[1]]
    mask = mask[:target.shape[0], :target.shape[1]]
    return torch.sum((mask * (pred - target)) ** 2) / torch.sum(mask)


# == 主流程 ==
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"
file_path_spatial = "D:/stDGA/data/data_column/starmap/starmap_spatial_processed.h5ad"
file_path_seq = "D:/stDGA/data/data_column/starmap/VISp_RNA_processed.h5ad"
save_dir = "D:/stDGA/data/data_column/result_column"
plot_path = "D:/stDGA/data/data_column/result_column/image"
os.makedirs(save_dir, exist_ok=True)
os.makedirs(plot_path, exist_ok=True)

spatial_data = sc.read_h5ad(file_path_spatial)
seq_data = sc.read_h5ad(file_path_seq)
print(seq_data.shape)                  # 总览结构
print(seq_data.obs.head())      # 看前几行内容
print(seq_data.obs.columns)     # 看列名
shared_genes = np.array(spatial_data.var_names.intersection(seq_data.var_names))
n_genes = len(shared_genes)
print(f"共有基因数量: {n_genes}")

use_test = n_genes > 50
use_val = True

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
    idx_train, idx_val = idx_perm[:n_train], idx_perm[n_train:n_train+n_val]
    idx_test = sorted(set(range(n_genes)) - set(idx_train) - set(idx_val))
    train_genes = shared_genes[idx_train]
    val_genes = shared_genes[idx_val]
    test_genes = shared_genes[idx_test]
    print('测试基因：')
    print(test_genes)

tup = extract_hvg_and_split(
    file_path_spatial, file_path_seq, test_genes if use_test else [], val_genes,
    save_padded_path=os.path.join(save_dir, "padded_results"),
    n_hvg=2000,
)
# unpack
A_combined, spatial_coords, X_full, X_input, val_mask, test_mask, full_mask, mark, labels = tup
tensors = [X_full, X_input, val_mask, test_mask, full_mask, mark]
X_full, X_input, val_mask, test_mask, full_mask, mark = [torch.tensor(t.toarray() if sp.issparse(t) else t, dtype=torch.float32).to(device) for t in tensors]

# Convert adjacency to edge_index (COO format) for torch_geometric GATConv
edge_index = None
try:
    # Case 1: A_combined is a torch sparse tensor (returned by prepare_graph_data)
    if isinstance(A_combined, torch.Tensor) and A_combined.is_sparse:
        coalesced = A_combined.coalesce()
        indices = coalesced.indices()  # shape [2, E]
        edge_index = indices.to(device).long()
    # Case 2: A_combined is a scipy sparse matrix
    elif sp.issparse(A_combined):
        A_coo = A_combined.tocoo()
        edge_index = torch.tensor([A_coo.row.astype(np.int64), A_coo.col.astype(np.int64)], dtype=torch.long).to(device)
    else:
        # Try converting a dense/array-like adjacency
        A_arr = np.array(A_combined)
        A_coo = sp.coo_matrix(A_arr)
        edge_index = torch.tensor([A_coo.row.astype(np.int64), A_coo.col.astype(np.int64)], dtype=torch.long).to(device)
except Exception as e:
    print(f"⚠️ 无法将 A_combined 转换为 edge_index: {e}")
    edge_index = None

if edge_index is not None:
    try:
        print(f"edge_index shape: {edge_index.shape}, num_edges: {edge_index.size(1)}")
    except Exception:
        pass

model = GATAutoencoder(X_input.shape[1], hidden_dim=28, output_dim=X_input.shape[1], num_heads=4, num_layers=3).to(device)
# use `edge_index` (sparse COO) with GATConv to avoid dense adjacency allocations
if edge_index is None:
    raise RuntimeError("需要有效的 edge_index（从 A_combined 转换），但转换失败")

optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
import time
try:
    import psutil
except Exception:
    psutil = None
from data_prepare import extract_hvg_and_split, get_padding_time_seconds
# == Early stopping based on val loss ==
val_gt = torch.tensor(X_full.cpu().numpy(), dtype=torch.float32).to(device)
val_mask_tensor = val_mask
best_loss = float('inf')
best_state_dict = None
best_output = None
best_epoch = -1
patience = 100
epochs_without_improve = 0
start_time = time.time()
# Memory tracking
peak_rss_bytes = 0
process = psutil.Process(os.getpid()) if psutil else None
gpu_peak_bytes = 0
if torch.cuda.is_available():
    try:
        torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass
print("🚀🚀 开始训练循环...")
for epoch in range(1000):
    print(f"🔁 Epoch {epoch} START ======")
    model.train()
    out = model(X_input, edge_index, print_once=(epoch == 0))
    loss = masked_mse_loss(out, X_full, full_mask)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # sample process memory after update
    if process is not None:
        try:
            rss = process.memory_info().rss
            if rss > peak_rss_bytes:
                peak_rss_bytes = rss
        except Exception:
            pass
    if torch.cuda.is_available():
        try:
            curr_gpu = torch.cuda.max_memory_allocated()
            if curr_gpu and curr_gpu > gpu_peak_bytes:
                gpu_peak_bytes = curr_gpu
        except Exception:
            pass

    model.eval()
    with torch.no_grad():
        val_out = model(X_input, edge_index)  # 重新前向，确保是 eval 态输出
        val_loss = masked_mse_loss(val_out, val_gt, 1 - val_mask_tensor)

    if val_loss < best_loss:
        best_loss = val_loss.item()
        best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        best_output = val_out.detach().clone()  # 保存 eval 态输出
        best_epoch = epoch
        epochs_without_improve = 0

    else:
        epochs_without_improve += 1


    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Train Loss = {loss.item():.4f}, Val Loss = {val_loss.item():.4f}")
    if epochs_without_improve >= patience:
        print(f"Early stopping at epoch {epoch}, Best Val Loss = {best_loss:.4f}")
        break

# === 输出训练耗时 ===
end_time = time.time()
print(f"\n训练总耗时: {end_time - start_time:.2f} 秒")
# Print padding time and peak memory usage
try:
    pad_time = get_padding_time_seconds()
    print(f"填充/拼接操作总耗时: {pad_time:.4f} 秒")
except Exception:
    print("无法获取填充时间（get_padding_time_seconds 未找到）")

if peak_rss_bytes > 0:
    print(f"进程峰值常驻内存(RSS): {peak_rss_bytes / (1024**2):.2f} MB")
else:
    if psutil is None:
        print("psutil 未安装，无法获取进程内存使用情况")

if gpu_peak_bytes > 0:
    print(f"GPU 峰值分配内存: {gpu_peak_bytes / (1024**2):.2f} MB")
# == Evaluation ==
if best_state_dict is None:
    raise RuntimeError("训练过程中没有得到有效的 best_state_dict")

model.load_state_dict(best_state_dict)
model.to(device)
model.eval()
with torch.no_grad():
    X_imputed = model(X_input, edge_index)  # 最终使用最优参数的 eval 输出
from my_pingjia import evaluate_gat_autoencoder

final_mask = test_mask if use_test else val_mask
plot_file = "cluster_result_test_starmap.png" if use_test else "cluster_result_val_starmap.png"   #stDGA


per_gene_df, extended_metrics = evaluate_gat_autoencoder(
    X_imputed,
    X_full,
    final_mask,
    mark,
    labels,
    spatial_coords,
    plot_path=os.path.join(plot_path, plot_file),
    only_clustering=(n_genes <= 50),
    save_filled_path=os.path.join(save_dir, "imputed_spatial_filled.npz"),
)

print("✅ 评估完成，指标如下：")
print("📊 Per-gene Evaluation Summary:")
print(per_gene_df.describe().loc[["mean", "std", "25%", "50%", "75%"]].round(6))

print("\n🌍 Flattened Global Metrics:")
for k, v in extended_metrics["global_metrics"].items():
    print(f"{k:<12}: {v:.6f}")


import json
import numpy as np

def save_metrics_and_params(metrics: dict, model_config: dict, save_path: str):
    def convert_to_builtin(obj):
        if isinstance(obj, dict):
            return {k: convert_to_builtin(v) for k, v in obj.items()}
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    output = {
        "metrics": convert_to_builtin(metrics),
        "model_config": convert_to_builtin(model_config),
    }

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4)
    print(f"📁 模型参数与指标已保存到: {save_path}")



save_metrics_and_params(
    metrics=extended_metrics["global_metrics"],
    model_config={
        "input_dim": int(X_input.shape[1]),
        "hidden_dim": 28,
        "output_dim": int(X_input.shape[1]),
        "num_heads": 4,
        "num_layers": 3,
        "device": str(device),
        "n_genes": int(n_genes),
        "best_epoch": int(best_epoch),
        "early_stop_val_loss": float(best_loss),
    },
    save_path=os.path.join(save_dir, "starmap_results.json"),
)
