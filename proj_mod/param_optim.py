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

    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)

    trial.set_user_attr("scheduler_factor", 0.5)
    trial.set_user_attr("scheduler_patience", 3)
    trial.set_user_attr("threshold",1e-4)
    trial.set_user_attr("cooldown",1)
    trial.set_user_attr("scheduler_min_lr", 1e-7)
    
    model = define_model(trial).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler=optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,mode="min",
                                                   factor=trial.user_attrs["scheduler_factor"],
                                                   patience=trial.user_attrs["scheduler_patience"],
                                                   threshold=trial.user_attrs["threshold"],   # Require at least 0.0001 improvement to reset patience
                                                   cooldown=trial.user_attrs["cooldown"],       # Wait 1 epoch after LR drop before counting again
                                                   min_lr=trial.user_attrs["scheduler_min_lr"]
                                                  )

    eps = 1e-8
    best_val_loss = float('inf')

    patience = 7
    patience_counter = 0
    max_epochs = 10

    
    for epoch in range(max_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, eps)
        val_loss = validate(model, test_loader, device, eps)

        scheduler.step(val_loss)
        
        trial.report(val_loss, epoch)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        if val_loss < best_val_loss - 1e-4:  # added minimum delta threshold
            best_val_loss = val_loss
            patience_counter = 0          # reset counter on improvement
        else:
            patience_counter += 1         # increment if no improvement

            # Early stopping
            if patience_counter >= patience:
                break  
    
    return best_val_loss
