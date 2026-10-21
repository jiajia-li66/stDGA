def train_and_evaluate_gat(
    file_path,
    hidden_dim=15,
    num_heads=4,
    num_layers=3,
    reg_lambda=0.01,
    num_epochs=1000,
    patience=50,
    use_simulated=False,
    device=None
):
    import torch
    import torch.nn.functional as F
    import numpy as np
    import random
    import copy
    from AnnData import prepare_graph_data, AnnData_generate
    from hvg_gene import extract_hvg_and_split
    from my_pingjia import evaluate_gat_autoencoder
    from GAT_gae_module import GATAutoencoder, masked_mse_loss
    # ========= 设置种子与设备 =========
    seed = 123
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.empty_cache()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ========= 数据加载 =========
    choose = 0 if use_simulated else 1
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

    # ========= 掩码构造 =========
    if choose == 1:
        mark_train = (X_train_full != 0).astype(int)
        mark_val = (X_val_full != 0).astype(int)
        mark_test = (X_test_full != 0).astype(int)
        mark_train_all = mark_train * X_train_mask
        mark_val_all = mark_val * X_val_mask
        mark_test_all = mark_test * X_test_mask
    else:
        mark_val = X_val_mask
        mark_test = X_test_mask
        mark_train_all = X_train_mask
        mark_val_all = X_val_mask
        mark_test_all = X_test_mask

    # ========= 工具函数 =========
    def to_tensor(x):
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x).float().to(device)
        elif isinstance(x, torch.Tensor):
            return x.float().to(device)
        else:
            raise TypeError(f"不支持的类型: {type(x)}")

    # ========= 转 tensor =========
    X_train = to_tensor(X_train)
    X_val = to_tensor(X_val)
    X_test = to_tensor(X_test)
    X_train_full = to_tensor(X_train_full)
    X_val_full = to_tensor(X_val_full)
    X_test_full = to_tensor(X_test_full)
    X_train_mask = to_tensor(X_train_mask)
    X_val_mask = to_tensor(X_val_mask)
    X_test_mask = to_tensor(X_test_mask)
    mark_train_all = to_tensor(mark_train_all)
    mark_val_all = to_tensor(mark_val_all)
    mark_test_all = to_tensor(mark_test_all)
    mark_val = to_tensor(mark_val)
    mark_test = to_tensor(mark_test)

    A_spatial_train = A_spatial_train.to(device)
    A_cluster_train = A_cluster_train.to(device)
    A_expr_train = A_expr_train.to(device)
    A_spatial_val = A_spatial_val.to(device)
    A_cluster_val = A_cluster_val.to(device)
    A_expr_val = A_expr_val.to(device)
    A_spatial_test = A_spatial_test.to(device)
    A_cluster_test = A_cluster_test.to(device)
    A_expr_test = A_expr_test.to(device)

    # ========= 模型构建 =========
    input_dim = X_train.shape[1]
    output_dim = input_dim
    model = GATAutoencoder(input_dim, hidden_dim, output_dim, num_heads, num_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=reg_lambda)

    # ========= 融合邻接矩阵 =========
    A_combined_train = model.combine_adj_matrices(A_cluster_train, A_spatial_train, A_expr_train)
    A_combined_val = model.combine_adj_matrices(A_cluster_val, A_spatial_val, A_expr_val)

    # ========= 训练 =========
    min_delta = 1e-6
    import time
    best_val_loss = float("inf")
    best_model = None
    epochs_without_improvement = 0
    evaluate_every = 10  #
    start_time = time.time()

    for epoch in range(num_epochs):
        # === Train ===
        model.train()
        imputed_X_train = model(X_train, A_combined_train)
        final_X_train = X_train + (1 - mark_train_all) * imputed_X_train
        train_loss = masked_mse_loss(final_X_train, X_train_full, X_train_mask)

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        # === Validate ===
        model.eval()
        with torch.no_grad():
            val_output = model(X_val, A_combined_val)
            final_X_val = X_val + (1 - mark_val_all) * val_output
            val_loss = masked_mse_loss(final_X_val, X_val_full, X_val_mask)

        # === Early Stopping Logic ===
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            best_model = copy.deepcopy(model)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        # === Logging ===
        if (epoch + 1) % evaluate_every == 0 or (epoch + 1) == num_epochs:
            print(f"Epoch {epoch + 1}, Train Loss: {train_loss.item():.4f}, Val Loss: {val_loss.item():.4f}")

        if epochs_without_improvement >= patience:
            print(f"\nEarly stopping triggered at epoch {epoch + 1}")
            break

    # === 输出训练耗时 ===
    end_time = time.time()
    print(f"\n训练总耗时: {end_time - start_time:.2f} 秒")

    # === 恢复最佳模型（供测试使用） ===
    model = best_model

    return best_val_loss.item(), best_model
