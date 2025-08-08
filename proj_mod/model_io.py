import os
import torch
import joblib
import optuna

def save_model_and_study(model, study, save_dir, weights_filename, study_filename):
    """
    Save a PyTorch model's state_dict and an Optuna study object.

    Parameters
    ----------
    model : torch.nn.Module
        The trained PyTorch model.
    study : optuna.Study
        The completed Optuna study.
    save_dir : str
        Directory where files will be saved.
    weights_filename : str, optional
        Filename for model weights (default: "best_weights.pth").
    study_filename : str, optional
        Filename for Optuna study (default: "study.pkl").
    """
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, weights_filename))
    joblib.dump(study, os.path.join(save_dir, study_filename))

def load_model_and_study(define_model_fn, device, save_dir, weights_filename="best_weights.pth", study_filename="study.pkl"):
    """
    Load a PyTorch model and an Optuna study from disk.

    Parameters
    ----------
    define_model_fn : callable
        Function that creates the model, should accept an Optuna trial as input.
    device : torch.device
        Device to load the model to.
    save_dir : str
        Directory where files are stored.
    weights_filename : str, optional
        Filename for model weights (default: "best_weights.pth").
    study_filename : str, optional
        Filename for Optuna study (default: "study.pkl").

    Returns
    -------
    model : torch.nn.Module
        The loaded model in evaluation mode.
    study : optuna.Study
        The loaded Optuna study.
    """
    # Load study
    study = joblib.load(os.path.join(save_dir, study_filename))
    best_trial = study.best_trial

    # Recreate model
    model = define_model_fn(optuna.trial.FixedTrial(best_trial.params)).to(device)

    # Load weights
    state = torch.load(os.path.join(save_dir, weights_filename), map_location=device)
    model.load_state_dict(state)
    model.eval()

    return model, study