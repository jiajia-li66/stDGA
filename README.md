## Usage

Two implementations are provided in this repository.

### 1. GAT_logits

The designated entry-point file is `GAT_logits/GAT_gae_module.py`.

Run the following command from the repository root:

```powershell
python GAT_logits/GAT_gae_module.py
```

### 2. GAT_VAE_column_logits

The designated entry-point file is `GAT_VAE_column_logits/GAT_gae_module_final.py`.

Run the following command from the repository root:

```powershell
python GAT_VAE_column_logits/GAT_gae_module_final.py
```

### Before Running

1. Activate the Conda environment configured for this project.
2. Install the required dependencies.
3. Check that all required local Python modules are present.
4. Update dataset paths to match the local machine.
5. Confirm that the required input datasets are available.

Some scripts currently contain machine-specific absolute paths. These must be updated before running the code on another computer.

The commands above identify the designated entry-point files. Successful execution should be verified in the target environment before considering the workflows fully reproducible.
