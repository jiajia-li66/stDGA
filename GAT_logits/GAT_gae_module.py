import torch.nn as nn
from torch_geometric.nn import GATConv
import torch.nn.functional as F
from AnnData import prepare_graph_data
from AnnData import AnnData_generate
from hvg_gene import extract_hvg_and_split
from my_pingjia import evaluate_gat_autoencoder
import numpy as np
import random
import torch
from sklearn.model_selection import train_test_split
import os
import copy
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
import psutil
import pandas as pd

# 固定保存路径（按需修改）
SAVE_DIR = "D:/stDGA/output/stDGA/imputed_csv"
os.makedirs(SAVE_DIR, exist_ok=True)
MEMORY_LOG_PATH = os.path.join(SAVE_DIR, "training_memory_usage_decidua basalis_10%.csv")  #改运行内存的保存地址

os.environ["OMP_NUM_THREADS"] = "1"

# ========= 设置 =========
seed = 123
use_simulated = False  # 设置为 True 则输出全部指标(仿真)，False 则仅输出真实数据所需指标（真实)
choose = 0 if use_simulated else 1

# ========= 固定随机种子 =========
torch.cuda.empty_cache()
print(f"Available GPU Memory: {torch.cuda.memory_allocated() / 1024 ** 2} MB")
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


# ========= 工具函数 =========
def to_tensor_if_needed(x, device):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float().to(device)
    elif isinstance(x, torch.Tensor):
        return x.float().to(device)
    else:
        raise TypeError("Unsupported type: {}".format(type(x)))


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

        # ✅ 默认不设置，需用户手动设置
        self.alpha_weights = None

    def forward(self, x, A_combined):
        for encoder in self.gat_encoders:
            x = encoder(x, A_combined)
        for decoder in self.linear_decoders:
            x = torch.relu(decoder(x))
        return self.output_layer(x)

    def combine_adj_matrices(self, A_cluster, A_spatial, A_expr, mode='and'):
        """
        使用逻辑与/或合并三个邻接矩阵（0/1稀疏图）
        :param mode: 'or' 或 'and'
        """
        assert mode in ['or', 'and'], "mode 只能是 'or' 或 'and'"

        # 获取所有索引集合（转为 tuple 方便 set 操作）
        idx_c = set(map(tuple, A_cluster._indices().cpu().numpy().T))
        idx_s = set(map(tuple, A_spatial._indices().cpu().numpy().T))
        idx_e = set(map(tuple, A_expr._indices().cpu().numpy().T))

        if mode == 'or':
            combined_idx = idx_c | idx_s | idx_e  # 并集
        else:
            combined_idx = idx_c & idx_s & idx_e  # 交集

        if not combined_idx:
            raise ValueError("合并后的邻接为空，请检查是否所有邻接都为0或交集太小")

        # 转为 Tensor 格式
        indices = torch.tensor(list(zip(*combined_idx)), dtype=torch.long).to(A_cluster.device)
        values = torch.ones(indices.shape[1], dtype=torch.float32).to(A_cluster.device)

        A_fused = torch.sparse_coo_tensor(indices, values, size=A_cluster.shape).coalesce()
        print(f"✅ 融合后非零边数: {A_fused._nnz()} (模式: {mode})")

        return A_fused


def masked_mse_loss(pred, target, mask):
    loss = torch.sum(((1 - mask) * (pred - target)) ** 2) / torch.sum(mask)
    return loss


# ========= 数据加载 =========
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
print(f"Using device: {device}")
file_path = "D:/stDGA/data/Sporadic_data/decidua basalis_10%/processed_data.h5"
data = extract_hvg_and_split(file_path)
X_train, X_train_full, X_train_mask, spatial_train, labels_train = data["train"]
X_val, X_val_full, X_val_mask, spatial_val, labels_val = data["val"]
X_test, X_test_full, X_test_mask, spatial_test, labels_test = data["test"]

adata_train = AnnData_generate(X_train, spatial_train, labels_train)
adata_val = AnnData_generate(X_val, spatial_val, labels_val)
adata_test = AnnData_generate(X_test, spatial_test, labels_test)

A_spatial_train, A_cluster_train, A_expr_train = prepare_graph_data(adata_train)
A_spatial_val, A_cluster_val, A_expr_val = prepare_graph_data(adata_val)
A_spatial_test, A_cluster_test, A_expr_test = prepare_graph_data(adata_test)

mark_train = (X_train_full != 0).astype(int) if choose else X_train_mask
mark_val = (X_val_full != 0).astype(int) if choose else X_val_mask
mark_test = (X_test_full != 0).astype(int) if choose else X_test_mask
mark_train_all = mark_train * X_train_mask if choose else X_train_mask
mark_val_all = mark_val * X_val_mask if choose else X_val_mask
mark_test_all = mark_test * X_test_mask if choose else X_test_mask

# ========= 转 tensor =========
X_train = to_tensor_if_needed(X_train, device)
X_val = to_tensor_if_needed(X_val, device)
X_test = to_tensor_if_needed(X_test, device)
X_train_full = to_tensor_if_needed(X_train_full, device)
X_val_full = to_tensor_if_needed(X_val_full, device)
X_test_full = to_tensor_if_needed(X_test_full, device)
X_val_mask = to_tensor_if_needed(X_val_mask, device)
X_test_mask = to_tensor_if_needed(X_test_mask, device)
X_train_mask = to_tensor_if_needed(X_train_mask, device)
mark_train_all = to_tensor_if_needed(mark_train_all, device)
mark_val_all = to_tensor_if_needed(mark_val_all, device)
mark_test_all = to_tensor_if_needed(mark_test_all, device)
mark_val = to_tensor_if_needed(mark_val, device)
mark_test = to_tensor_if_needed(mark_test, device)

# ========= 模型参数 =========
input_dim = X_train.shape[1]
hidden_dim = 23
output_dim = input_dim
num_heads = 6
num_layers = 4
model = GATAutoencoder(input_dim, hidden_dim, output_dim, num_heads, num_layers).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
# ===== 显存 & 参数量检查 =====
total_params = sum(p.numel() for p in model.parameters())
print(f"📊 模型参数总量: {total_params / 1e6:.2f} M")

if device.type == "cuda":
    print(f"🧠 当前 CUDA 显存使用: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")
    print(f"🧠 当前 CUDA 显存缓存: {torch.cuda.memory_reserved() / 1024 ** 2:.2f} MB")

reg_lambda = 0.01
patience = 20
best_val_loss = float('inf')
min_delta = 1e-5
epochs_without_improvement = 0
num_epochs = 1000

# 训练前融合邻接矩阵
A_combined_train = model.combine_adj_matrices(A_cluster_train, A_spatial_train, A_expr_train).to(device)
A_combined_val = model.combine_adj_matrices(A_cluster_val, A_spatial_val, A_expr_val).to(device)
A_combined_test = model.combine_adj_matrices(A_cluster_test, A_spatial_test, A_expr_test).to(device)

evaluate_every = 10  # 每隔多少轮评估一次

# ========= 训练过程 =========
import time
import matplotlib.pyplot as plt

val_loss_list = []
best_model = None
start_time = time.time()

memory_log = []
# 在训练前（训练循环外）如果使用 GPU，请重置峰值统计
if device.type == "cuda":
    torch.cuda.reset_peak_memory_stats()

for epoch in range(num_epochs):
    model.train()
    imputed_X_train = model(X_train, A_combined_train)
    final_X_train = X_train + (1 - mark_train_all) * imputed_X_train
    mse_loss = masked_mse_loss(final_X_train, X_train_full, X_train_mask)
    total_loss = mse_loss

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    # 在 optimizer.step() 之后添加
    proc = psutil.Process()
    cpu_mem_mb = proc.memory_info().rss / 1024 ** 2
    sys_mem_pct = psutil.virtual_memory().percent
    if device.type == "cuda":
        gpu_mem_mb = torch.cuda.memory_allocated() / 1024 ** 2
        gpu_peak_mb = torch.cuda.max_memory_allocated() / 1024 ** 2
    else:
        gpu_mem_mb = None
        gpu_peak_mb = None

    print(
        f"Epoch {epoch + 1}: CPU_RSS={cpu_mem_mb:.1f} MB, Sys%={sys_mem_pct:.1f}%, GPU={gpu_mem_mb if gpu_mem_mb is None else f'{gpu_mem_mb:.1f} MB'}, GPU_peak={gpu_peak_mb if gpu_peak_mb is None else f'{gpu_peak_mb:.1f} MB'}")

    memory_log.append({
        "epoch": epoch + 1,
        "cpu_mem_mb": cpu_mem_mb,
        "sys_mem_pct": sys_mem_pct,
        "gpu_mem_mb": gpu_mem_mb,
        "gpu_peak_mb": gpu_peak_mb
    })

    model.eval()
    with torch.no_grad():
        imputed_val = model(X_val, A_combined_val)
        final_X_val = X_val + (1 - mark_val_all) * imputed_val
        val_loss_value = masked_mse_loss(final_X_val, X_val_full, X_val_mask).item()
        val_loss_list.append(val_loss_value)

        should_evaluate = (epoch + 1) % evaluate_every == 0 or (epoch + 1) == num_epochs
        if should_evaluate:
            val_results = evaluate_gat_autoencoder(
                model, X_val, A_combined_val,
                X_val_mask, mark_val, X_val_full, device, labels_val,
                enable_clustering=False
            )

            print(f"\nEpoch {epoch + 1}, Train Loss: {total_loss.item():.4f}, Val Loss: {val_loss_value:.4f}")

            if use_simulated:
                print(
                    f"Val Metrics (Simulated): Cosine={val_results[0]:.4f}, RMSE={val_results[1]:.4f}, "
                    f"MAE={val_results[2]:.4f}, PCC={val_results[3]:.4f}, MSE={val_results[4]:.4f}, "
                    f"CellSim={val_results[5]:.4f}, GeneSim={val_results[6]:.4f}")
            else:
                print(f"Val Metrics (Real): RMSE={val_results[1]:.4f}, "
                      f"MAE={val_results[2]:.4f}, MSE={val_results[4]:.4f}")
        else:
            print(f"Epoch {epoch + 1}, Train Loss: {total_loss.item():.4f}, Val Loss: {val_loss_value:.4f}")

    if val_loss_value < best_val_loss - min_delta:
        best_val_loss = val_loss_value
        best_model = copy.deepcopy(model)
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= patience or epoch == num_epochs - 1:
        print(f"\nEarly stopping at epoch {epoch + 1}")
        break

end_time = time.time()
training_time = end_time - start_time
# pd.DataFrame(memory_log).to_csv(MEMORY_LOG_PATH, index=False)
# print(f"Saved memory log to {}")
print(f"\n训练总耗时: {training_time:.2f} 秒")
max_cpu = max(d["cpu_mem_mb"] for d in memory_log)
max_gpu_peak = max((d["gpu_peak_mb"] or 0) for d in memory_log)
print(f"Peak CPU RSS during training: {max_cpu:.1f} MB")
if device.type == "cuda":
    print(f"Peak GPU memory during training: {max_gpu_peak:.1f} MB")

# ========= 保存 val_loss 曲线图 =========
plt.figure(figsize=(6, 4))
plt.plot(range(1, len(val_loss_list) + 1), val_loss_list, label="Val Loss", marker='o')
plt.xlabel("Epoch")
plt.ylabel("Validation Loss")
plt.title("Validation Loss Curve")
plt.grid(True)
plt.tight_layout()
plt.legend()
plt.savefig("D:/stDGA/data/Sporadic_data/result/零星/img/val_loss_curve_decidua basalis_10%.png", dpi=300)
plt.close()

# ========= 测试 =========
model = best_model
model.eval()

from my_pingjia import evaluate_gat_autoencoder
from utils import save_clustering_plot, save_evaluation_results

eval_results = evaluate_gat_autoencoder(
    model,
    X_test, A_combined_test,
    X_test_mask, mark_test, X_test_full, device, labels_test, predict=2,
    plot_path="D:/stDGA/data/Sporadic_data/result/零星/image/cluster_decidua basalis_10%.png",
    enable_clustering=True
)

# Save all computed evaluation metrics and params to CSV
metrics_all = {
    "Cosine": eval_results[0], "RMSE": eval_results[1],
    "MAE": eval_results[2], "PCC": eval_results[3], "MSE": eval_results[4],
    "CellSim": eval_results[5], "GeneSim": eval_results[6],
    "ARI": eval_results[7], "Silhouette": eval_results[8], "NMI": eval_results[9],
    "AMI": eval_results[10], "Homogeneity": eval_results[11], "Completeness": eval_results[12],
    "V-Measure": eval_results[13],
    "Fowlkes-Mallows": eval_results[16], "Purity": eval_results[17]
}
params_all = {
    "hidden_dim": hidden_dim, "num_heads": num_heads,
    "num_layers": num_layers, "reg_lambda": reg_lambda,
}
eval_save_path = os.path.join("D:/stDGA/data/Sporadic_data/result/零星", "evaluation_log_decidua basalis_10%.csv")
out = {**metrics_all, **{f"param_{k}": v for k, v in params_all.items()}}
pd.DataFrame([out]).to_csv(eval_save_path, index=False)
print(f"Saved evaluation metrics and params to {eval_save_path}")

# ========= 打印测试指标 =========
if use_simulated:
    print(f"Test Results (Simulated): Cosine={eval_results[0]:.4f}, RMSE={eval_results[1]:.4f}, "
          f"MAE={eval_results[2]:.4f}, PCC={eval_results[3]:.4f}, MSE={eval_results[4]:.4f}, "
        f"ARI={eval_results[7]:.4f}, Silhouette={eval_results[8]:.4f}, NMI={eval_results[9]:.4f}, "
        f"AMI={eval_results[10]:.4f}, Purity={eval_results[17]:.4f}, "
        f"Homogeneity={eval_results[11]:.4f}, Completeness={eval_results[12]:.4f}, V-Measure={eval_results[13]:.4f}, "
        f"Calinski-Harabasz={eval_results[14]:.4f}, Davies-Bouldin={eval_results[15]:.4f}, Fowlkes-Mallows={eval_results[16]:.4f}, "
        f"CellSim={eval_results[5]:.4f}, GeneSim={eval_results[6]:.4f}")
else:
    print(f"Test Results (Real): RMSE={eval_results[1]:.4f}, MAE={eval_results[2]:.4f}, "
        f"MSE={eval_results[4]:.4f}, ARI={eval_results[7]:.4f}, Silhouette={eval_results[8]:.4f}, "
        f"Homogeneity={eval_results[11]:.4f}, Completeness={eval_results[12]:.4f}, "
        f"Fowlkes-Mallows={eval_results[16]:.4f}, "
        f"NMI={eval_results[9]:.4f}, AMI={eval_results[10]:.4f}, Purity={eval_results[17]:.4f}")

# 保存 train/val/test 的填充结果
train_save_path = os.path.join(SAVE_DIR, "train_decidua basalis_10%.csv") #stDGA
val_save_path = os.path.join(SAVE_DIR, "val_decidua basalis_10%.csv")
test_save_path = os.path.join(SAVE_DIR, "test_decidua basalis_10%.csv")

# 确保 tensor 转为 numpy（detach->cpu->numpy）
pd.DataFrame(final_X_train.detach().cpu().numpy()).to_csv(train_save_path, index=False)
pd.DataFrame(final_X_val.detach().cpu().numpy()).to_csv(val_save_path, index=False)

model = best_model
model.eval()
with torch.no_grad():
    imputed_test = model(X_test, A_combined_test)
    final_X_test = X_test + (1 - mark_test_all) * imputed_test
pd.DataFrame(final_X_test.detach().cpu().numpy()).to_csv(test_save_path, index=False)
print(f"Saved imputed CSVs to {SAVE_DIR}")
