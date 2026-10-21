import itertools
from gat_train_function import train_and_evaluate_gat  # 确保包含你封装的函数
import os
import json
import numpy as np

# 输入数据路径
file_path_spatial = "C:/Users/JGuan/Desktop/Lijiajia/data_column/SSp_osm/osm_processed.h5ad"
file_path_seq = "C:/Users/JGuan/Desktop/Lijiajia/data_column/SSp_osm/SSp_processed.h5ad"

# 定义超参数组合
param_grid = {
    'hidden_dim': [8, 15],
    'num_heads': [2, 4],
    'num_layers': [2, 4],
}

param_combinations = list(itertools.product(*param_grid.values()))
param_keys = list(param_grid.keys())

best_score = float('inf')
best_result = {}

results = []

for i, combo in enumerate(param_combinations):
    params = dict(zip(param_keys, combo))
    print(f"\n🔧 正在评估组合 {i + 1}/{len(param_combinations)}: {params}")

    try:
        val_loss, model_config = train_and_evaluate_gat(
            file_path_spatial=file_path_spatial,
            file_path_seq=file_path_seq,
            hidden_dim=params['hidden_dim'],
            num_heads=params['num_heads'],
            num_layers=params['num_layers'],
            verbose=False
        )
    except Exception as e:
        print(f"⚠️ 当前组合运行出错: {e}")
        continue

    result = {
        "val_loss": val_loss,
        "params": {
            **model_config
        }
    }

    results.append(result)

    if val_loss < best_score:
        best_score = val_loss
        best_result = result
        print("🎯 当前为最优组合")
    print(f"当前最佳验证集损失: {best_score:.4f}")

print("\n✅ 网格搜索完成！")
print(f"最优参数组合: {best_result['params']}")
print(f"最优验证集损失: {best_result['val_loss']:.4f}")

# with open(save_path, "w", encoding="utf-8") as f:
#     json.dump(convert_to_builtin(best_result), f, indent=4)

# print(f"\n📁 最优结果已保存至: {save_path}")
