import os
import sys
import pathlib
import argparse
import json

import pandas as pd
import numpy as np
import itertools

import torch

from sklearn.model_selection import StratifiedKFold

sys.path.append(str(pathlib.Path(__file__).parent.parent))
from utils.dataset import DRPDataset
from utils.train import train
from utils.plot import plot_auc_history

np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

device = "cuda" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser(description="Preprocess data for DRP.")
parser.add_argument("--drug_name", type=str, required=True, help="Drug name.")
parser.add_argument("--norm", type=str, default="gradual_tanh", choices=["tanh", "gradual_tanh", "layer_norm", "id", "classical"], help="Normalization method.")
args = parser.parse_args()

data_path = pathlib.Path(__file__).parent.parent / "data" / "preprocessed_data" / args.drug_name
log_path = pathlib.Path(__file__).parent.parent / "log" / args.drug_name / args.norm

train_data = pd.read_csv(data_path / "train_data.csv")
IC50_train = np.load(data_path / "train_IC50.npy")
IC50_train_binary = np.load(data_path / "train_IC50_binary.npy")

IC50_train_bin = pd.cut(IC50_train, bins=4, labels=False)

grid_search_range = {
    "num_qubits": [4, 8],
    "num_encoding_layer": [2, 4],
    "reps": [1, 2, 4],
    "lr": [1e-6, 1e-5, 1e-4],
    "width" : [20],
    "rotation" : [np.pi/2]
}
keys = grid_search_range.keys()
values = grid_search_range.values()

train_auc_history_all = np.zeros((len(list(itertools.product(*values))), 10*5, 101))
valid_auc_history_all = np.zeros((len(list(itertools.product(*values))), 10*5, 101))
for iter_num in range(10):
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=iter_num)
    for fold, (train_index, valid_index) in enumerate(kf.split(train_data, IC50_train_bin)):
        train_data_split, IC50_train_split, IC50_train_binary_split = train_data.iloc[train_index], IC50_train[train_index], IC50_train_binary[train_index]
        valid_data_split, IC50_valid_split, IC50_valid_binary_split = train_data.iloc[valid_index], IC50_train[valid_index], IC50_train_binary[valid_index]
        train_data_tensor, IC50_train_tensor, IC50_train_binary_tensor = torch.tensor(train_data_split.values, dtype=torch.float), torch.tensor(IC50_train_split, dtype=torch.float).unsqueeze(1), torch.tensor(IC50_train_binary_split, dtype=torch.float).unsqueeze(1)
        valid_data_tensor, IC50_valid_tensor, IC50_valid_binary_tensor = torch.tensor(valid_data_split.values, dtype=torch.float), torch.tensor(IC50_valid_split, dtype=torch.float).unsqueeze(1), torch.tensor(IC50_valid_binary_split, dtype=torch.float).unsqueeze(1)
        train_dataset = DRPDataset(train_data_tensor, IC50_train_tensor, IC50_train_binary_tensor)
        valid_dataset = DRPDataset(valid_data_tensor, IC50_valid_tensor, IC50_valid_binary_tensor)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, drop_last=False)
        for hyper_num, combination in enumerate(itertools.product(*values)):
            hyper_params = dict(zip(keys, combination))
            print(f"Iteration {iter_num}, Hyperparameters: {hyper_params}")
            if args.norm == "classical":
                from model.models import ClassicalNN
                model = ClassicalNN(input_size=train_data_tensor.shape[1], hyper_params=hyper_params).to(device)
            else:
                from model.models import HybridQNN
                model = HybridQNN(input_size=train_data_tensor.shape[1], hyper_params=hyper_params, norm=args.norm).to(device)
            loss_fn = torch.nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=hyper_params["lr"])
            train_roc_auc_each, valid_roc_auc_each = train(model, train_loader, valid_dataset, optimizer, loss_fn, device)
            train_auc_history_all[hyper_num, iter_num*5 + fold] = train_roc_auc_each
            valid_auc_history_all[hyper_num, iter_num*5 + fold] = valid_roc_auc_each
            break
        break
    break

train_auc_history_mean = np.mean(train_auc_history_all, axis=1)
valid_auc_history_mean = np.mean(valid_auc_history_all, axis=1)

train_auc_mean_max = np.max(train_auc_history_mean, axis=1)
valid_auc_mean_max = np.max(valid_auc_history_mean, axis=1)

max_index = np.argmax(valid_auc_mean_max)
best_hyper_params = dict(zip(keys, list(itertools.product(*values))[max_index]))
with open(log_path / f"best_hyper_params.json", "w") as f:
    json.dump(best_hyper_params, f, indent=4)

# plot_auc_history(train_auc_history_mean, [f"Hyper {i}" for i in range(len(train_auc_history_mean))], log_path, title=f"{args.norm}_train")
# plot_auc_history(valid_auc_history_mean, [f"Hyper {i}" for i in range(len(valid_auc_history_mean))], log_path, title=f"{args.norm}_valid")
np.save(log_path / f"{args.norm}_train_auc_history_all.npy", train_auc_history_all)
np.save(log_path / f"{args.norm}_valid_auc_history_all.npy", valid_auc_history_all)