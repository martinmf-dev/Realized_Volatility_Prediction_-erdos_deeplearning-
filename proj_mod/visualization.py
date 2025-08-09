import torch
import numpy as np
import matplotlib.pyplot as plt 
import os

def training_plots(results_dict, titles=None, fig_width=15, subplot_aspect_ratio=0.75, save_path=None):
    # 07/21/2025 Created
    # 08/07/2025 Adds option to save figure
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
            f"{rnn} |  {h0} | min val loss: {torch.stack(results_dict[(rnn, h0)]['val_loss']).min().item():.4f}" 
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
        
        ax.scatter(x, tl, c="blue", marker='+', s=100, linewidths=1.5, label="Train")
        ax.scatter(x, vl, c="orange", marker='+', s=100, linewidths=1.5, label="Val")

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


    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, format="png")

    plt.show()

def study_plots(study, param_for_scatter="lr", title_prefix="Optuna"):
    # 08/09/2025 Created
    """
    Plot progress for an Optuna Study:
      1) Best-so-far curve (trial vs best RMSPE so far)
      2) Trial values + best-so-far overlay
      3) Optional: param vs RMSPE scatter (log x if param looks like lr)

    Args:
        study: optuna.Study
        param_for_scatter: str or None, e.g. "lr"
        title_prefix: str for plot titles
    """
    # Completed trials only, sorted by trial number
    trials_complete = [t for t in study.trials if t.value is not None and t.state.name == "COMPLETE"]
    trials_complete.sort(key=lambda t: t.number)

    if not trials_complete:
        print("No completed trials to plot.")
        return

    trials = [t.number for t in trials_complete]
    values = [t.value for t in trials_complete]

    # Best-so-far
    best_so_far = []
    cur = float("inf")
    for v in values:
        cur = min(cur, v)
        best_so_far.append(cur)


    # val loss + best-so-far val loss
    plt.figure(figsize=(8, 5))
    plt.plot(trials, values, marker='o', label="Validation Loss")
    plt.plot(trials, best_so_far, marker='o', label="Best Validation Loss so far")
    plt.xlabel("Trial")
    plt.ylabel("Validation Loss")
    plt.title(f" Validation Loss and Best Validation Loss ({title_prefix})")
    plt.legend()
    plt.grid(True)
    plt.show()

    #  Optional: param vs val loss
    if param_for_scatter:
        have_param = [param_for_scatter in t.params for t in trials_complete]
        if any(have_param):
            xs = [t.params[param_for_scatter] for t in trials_complete if param_for_scatter in t.params]
            ys = [t.value for t in trials_complete if param_for_scatter in t.params]
            plt.figure(figsize=(8, 5))
            plt.scatter(xs, ys, alpha=0.6)
            if "lr" in param_for_scatter.lower():
                plt.xscale("log")
                plt.xlabel(f"{param_for_scatter} (log scale)")
            else:
                plt.xlabel(param_for_scatter)
            plt.ylabel("Validation Loss")
            plt.title(f"{param_for_scatter} vs Validation Loss ({title_prefix})")
            plt.grid(True, which="both")
            plt.show()