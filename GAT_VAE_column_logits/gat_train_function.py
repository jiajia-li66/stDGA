def train_and_evaluate_gat(
    file_path_spatial,
    file_path_seq,
    hidden_dim,
    num_heads,
    num_layers,
    weight_decay=1e-4,
    num_epochs=1000,
    patience=50,
    device=None,
    verbose=False
):
    import torch
    import torch.nn.functional as F
    import numpy as np
    import os
    import scipy.sparse as sp
    from my_pingjia import evaluate_gat_autoencoder
    from AnnData import prepare_graph_data
    from data_prepare import extract_hvg_and_split
    import scanpy as sc
    from collections import defaultdict
    import torch.nn as nn
    import torch.optim as optim
    from torch_geometric.nn import GATConv

    # GAT Autoencoder Model Definition
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


        def forward(self, x, A_combined):
            for encoder in self.gat_encoders:
                x = encoder(x, A_combined)
            for decoder in self.linear_decoders:
                x = F.relu(decoder(x))
            return self.output_layer(x)

    # Loss Function: Masked MSE
    def masked_mse_loss(pred, target, mask):
        pred = pred[:target.shape[0], :target.shape[1]]
        mask = mask[:target.shape[0], :target.shape[1]]
        return torch.sum((mask * (pred - target)) ** 2) / torch.sum(mask)

    if device is None:
        device = torch.device("cpu")
        #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load spatial and seq data
    spatial_data = sc.read_h5ad(file_path_spatial)
    seq_data = sc.read_h5ad(file_path_seq)
    shared_genes = np.array(spatial_data.var_names.intersection(seq_data.var_names))
    n_genes = len(shared_genes)
    
    use_test = n_genes > 50
    if n_genes <= 50:
        n_train = int(n_genes * 0.8)
        n_val = int(n_genes * 0.2)
        idx_perm = np.random.permutation(range(n_genes))
        train_genes = shared_genes[idx_perm[:n_train]]
        val_genes = shared_genes[idx_perm[n_train:n_train+n_val]]
    else: 
        n_train = int(n_genes * 0.7)
        n_val = int(n_genes * 0.15)
        idx_perm = np.random.permutation(range(n_genes))
        train_genes = shared_genes[idx_perm[:n_train]]
        val_genes = shared_genes[idx_perm[n_train:n_train+n_val]]
        test_genes = shared_genes[idx_perm[n_train+n_val:]]

    # Extract data matrices
    A_combined, spatial_coords, X_full, X_input, val_mask, test_mask, full_mask, mark, labels = extract_hvg_and_split(
    file_path_spatial, file_path_seq, test_genes if use_test else [], val_genes
    )

    # Convert to torch tensors
    tensors = [X_full, X_input, val_mask, full_mask, mark]
    X_full, X_input, val_mask, full_mask, mark = [torch.tensor(t.toarray() if sp.issparse(t) else t, dtype=torch.float32).to(device) for t in tensors]
    A_combined = A_combined.to(device)

    # Initialize the model
    model = GATAutoencoder(
        input_dim=X_input.shape[1],
        hidden_dim=hidden_dim,
        output_dim=X_input.shape[1],
        num_heads=num_heads,
        num_layers=num_layers
    ).to(device)

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=0.01, weight_decay=weight_decay)

    best_loss = float('inf')
    best_output = None
    early_stop = 0

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        output = model(X_input, A_combined)
        train_loss = masked_mse_loss(output, X_full, full_mask)
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        # Validation step
        model.eval()
        with torch.no_grad():
            val_loss = masked_mse_loss(output, X_full, 1-val_mask)

        if val_loss < best_loss:
            best_loss = val_loss
            best_output = output.detach()
            early_stop = 0
        else:
            early_stop += 1

        if verbose and epoch % 10 == 0:
            print(f"[Epoch {epoch}] Val Loss: {val_loss.item():.4f}")

        if early_stop >= patience:
            break

    return float(best_loss), {
        "hidden_dim": hidden_dim,
        "num_heads": num_heads,
        "num_layers": num_layers,
    }


