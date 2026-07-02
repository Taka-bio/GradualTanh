from sklearn.metrics import roc_auc_score
import tqdm
import numpy as np

def train(model, train_loader, valid_dataset, optimizer, loss_fn, device):
    model.eval()
    initial_pred_train = model(train_loader.dataset.expr.to(device)).detach().cpu().numpy()
    initial_pred_valid = model(valid_dataset.expr.to(device)).detach().cpu().numpy()
    train_roc_auc_history = [roc_auc_score(train_loader.dataset.IC_binary.cpu().numpy(), initial_pred_train)]
    valid_roc_auc_history = [roc_auc_score(valid_dataset.IC_binary.cpu().numpy(), initial_pred_valid)]
    model.train()
    for i in tqdm.tqdm(range(1, 101)):
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
        model.train()
    return np.array(train_roc_auc_history), np.array(valid_roc_auc_history)
