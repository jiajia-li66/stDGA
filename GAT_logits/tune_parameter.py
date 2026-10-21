import itertools
from gat_train_function import train_and_evaluate_gat

# ✅ 超参数网格定义
param_grid = {
    'hidden_dim': [8, 12, 15],
    'num_heads': [2, 3, 4, 5],
    'num_layers': [2, 4, 6],
}


param_combinations = list(itertools.product(*param_grid.values()))
param_keys = list(param_grid.keys())

best_score = float('inf')
best_params = None

for i, combo in enumerate(param_combinations):
    params = dict(zip(param_keys, combo))
    print(f"\n🔧 正在评估组合 {i + 1}/{len(param_combinations)}: {params}")

    try:
        val_loss, _ = train_and_evaluate_gat(
            file_path="C:/Users/JGuan/Desktop/Lijiajia/data_unit/小鼠脑_30%/processed_data.h5",
            hidden_dim=params['hidden_dim'],
            num_heads=params['num_heads'],
            num_layers=params['num_layers'],
            use_simulated=True
        )
    except Exception as e:
        print(f"⚠️当前组合运行出错: {e}")
        continue

    if val_loss < best_score:
        best_score = val_loss
        best_params = params
        print("🎯 当前为最优组合")
    print(f"当前最佳验证集损失: {best_score:.4f}")

print("\n✅ 搜索完成")
print(f"最优参数组合: {best_params}")
print(f"对应验证集损失: {best_score:.4f}")