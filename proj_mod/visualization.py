import torch
import numpy as np
import matplotlib.pyplot as plt 

def training_plots(results_dict, titles=None, fig_width=15, subplot_aspect_ratio=0.75):
    # 07/21/2025 Created
    """
    Plots the training and validation loss for a dictionary of runs of models with keys (rnn,h0). 
    Each item in results_dict is a dictionary itself with keys "train_loss" and "val_los" 
    :param results_dict: Dictionary with keys (rnn,h0) and values equal to dictionaries with keys "train_loss" and "val_los" and values 'train_los' and 'val_los'
    :param titles: A list of titles for the subplots. If no list is provided, the titles are created using the keys of each run in results_dict and tfiguhe minimum validation loss
                    in the run
    :param fig_width: Determines the width of the figure containing the grid of plots
    :param subplot_aspect_ratio: Determines the aspect ratio of subplots
    """
    def to_cpu_list(loss_list):
        return [x.cpu() if hasattr(x, "cpu") else x for x in loss_list]

    keys = list(results_dict.keys())

    if titles is None:
        titles = [
            f"{rnn.upper()} | h0: {h0} | min val loss: {torch.stack(results_dict[(rnn, h0)]['val_loss']).min().item():.4f}" 
            for (rnn, h0) in results_dict]
    
    n = len(keys)
    n_cols = min(3, n)
    n_rows = int(np.ceil(n / n_cols))

    subplot_width = fig_width / n_cols
    subplot_height = subplot_width * subplot_aspect_ratio
    fig_height = n_rows * subplot_height

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(fig_width, fig_height),
        squeeze=False,
        sharex=True,
        sharey=True,
        constrained_layout=True
    )

    for idx, key in enumerate(keys):
        i, j = divmod(idx, n_cols)
        ax = axes[i][j]
        run_data = results_dict[key]

        tl = to_cpu_list(run_data["train_loss"])
        vl = to_cpu_list(run_data["val_loss"])
        x = np.arange(len(tl))
        
        ax.scatter(x, tl, c="blue", marker='+', s=60, linewidths=1.5, label="Train")
        ax.scatter(x, vl, c="orange", marker='+', s=60, linewidths=1.5, label="Val")

        ax.set_title(titles[idx])

        if i == n_rows - 1:
            ax.set_xlabel("Epoch")
        if j == 0:
            ax.set_ylabel("Loss")
        ax.grid(True)

    # Remove unused subplots
    for idx in range(n, n_rows * n_cols):
        i, j = divmod(idx, n_cols)
        fig.delaxes(axes[i][j])

    fig.suptitle("Training and Validation Loss", fontsize=16)
    plt.show()