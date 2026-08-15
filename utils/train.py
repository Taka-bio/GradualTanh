from sklearn.metrics import roc_auc_score
import tqdm
import numpy as np
import torch

def train(model, train_loader, valid_dataset, optimizer, loss_fn, device, early_stop_patience=None, path=None, name=""):
    model.eval()
    initial_pred_train = model(train_loader.dataset.expr.to(device)).detach().cpu().numpy()
    initial_pred_valid = model(valid_dataset.expr.to(device)).detach().cpu().numpy()
    train_roc_auc_history = [roc_auc_score(train_loader.dataset.IC_binary.cpu().numpy(), initial_pred_train)]
    valid_roc_auc_history = [roc_auc_score(valid_dataset.IC_binary.cpu().numpy(), initial_pred_valid)]
    model.train()
    best_valid_auc = valid_roc_auc_history[0]
    torch.save(model.state_dict(), path / f"best_model{name}.pt")
    patience_counter = 0
    for i in range(1, 101):
        running_loss = 0.0
        for x, y, _ in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        model.eval()
        pred_train = model(train_loader.dataset.expr.to(device)).detach().cpu().numpy()
        pred_valid = model(valid_dataset.expr.to(device)).detach().cpu().numpy()
        train_roc_auc_history.append(roc_auc_score(train_loader.dataset.IC_binary.cpu().numpy(), pred_train))
        valid_roc_auc_history.append(roc_auc_score(valid_dataset.IC_binary.cpu().numpy(), pred_valid))
        if early_stop_patience is not None:
            if valid_roc_auc_history[-1] > best_valid_auc:
                best_valid_auc = valid_roc_auc_history[-1]
                torch.save(model.state_dict(), path / f"best_model{name}.pt")
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= early_stop_patience:
                break
        model.train()
    if early_stop_patience is not None:
        model.load_state_dict(torch.load(path / f"best_model{name}.pt"))
    return np.array(train_roc_auc_history), np.array(valid_roc_auc_history)
