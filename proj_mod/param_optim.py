import torch
import torch.optim as optim
import optuna
from proj_mod import training

def train_one_epoch(model, dataloader, optimizer, device, eps=0):
    model.train()
    loss_fn = training.RMSPELoss(eps=eps) 
    total_data_count=len(dataloader.dataset)
    sum_of_squares=0.0
    for batch in dataloader:
        x, y = batch
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            sum_of_squares+=torch.sum(torch.square((pred-y)/(y+eps)))
    with torch.no_grad():
        rmspe=torch.sqrt(sum_of_squares/total_data_count)
    return rmspe.item()

def validate(model, dataloader, device, eps=0):
    model.eval()
    sum_of_squares = 0.0
    total_data_count=len(dataloader.dataset)
    with torch.no_grad():
        for batch in dataloader:
            x, y = batch
            x, y = x.to(device), y.to(device)
            pred = model(x)
            sum_of_squares += torch.sum(torch.square((pred - y) / (y + eps)))
        rmspe = torch.sqrt(sum_of_squares / total_data_count)
    return rmspe.item()



def objective(trial, define_model, train_loader, test_loader, device):
    # Hyperparams (keep it minimal)
    lr = trial.suggest_float("lr", 1e-4, 3e-3, log=True)

    # Model & optimizer
    model = define_model(trial).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    # Early-stopping settings
    max_epochs   = 12
    patience     = 5
    min_delta    = 1e-4
    eps          = 1e-8

    best_val = float("inf")
    patience_ctr = 0

    for epoch in range(max_epochs):
        train_one_epoch(model, train_loader, optimizer, device, eps)
        val_loss = validate(model, test_loader, device, eps)

        if val_loss < best_val - min_delta:
            best_val = val_loss
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                break  # stop this trial early

    return best_val
