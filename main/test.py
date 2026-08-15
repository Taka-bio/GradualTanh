import os
import sys
import pathlib
import argparse
import json
import copy

import pandas as pd
import numpy as np
import itertools

import torch

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

sys.path.append(str(pathlib.Path(__file__).parent.parent))
from utils.dataset import DRPDataset
from utils.train import train
from utils.plot import plot_auc_history
from utils import evaluation

np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

device = "cuda" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser(description="Preprocess data for DRP.")
parser.add_argument("--drug_name", type=str, required=True, help="Drug name.")
args = parser.parse_args()

data_path = pathlib.Path(__file__).parent.parent / "data" / "preprocessed_data" / args.drug_name
cv_log_path = pathlib.Path(__file__).parent.parent / "log" / args.drug_name 
results_path = pathlib.Path(__file__).parent.parent / "log" / args.drug_name / "results"

train_data = pd.read_csv(data_path / "train_data.csv")
IC50_train = np.load(data_path / "train_IC50.npy")
IC50_train_binary = np.load(data_path / "train_IC50_binary.npy")
test_data = pd.read_csv(data_path / "test_data.csv")
test_IC50 = np.load(data_path / "test_IC50_binary.npy")

from sklearn.model_selection import train_test_split
IC50_train_bin = pd.cut(IC50_train, bins=4, labels=False)
train_data, val_data, IC50_train, IC50_val, IC50_train_binary, IC50_val_binary = train_test_split(train_data, IC50_train, IC50_train_binary, test_size=0.2, random_state=0, stratify=IC50_train_bin)

train_data_tensor, IC50_train_tensor, IC50_train_binary_tensor = torch.tensor(train_data.values, dtype=torch.float), torch.tensor(IC50_train, dtype=torch.float).unsqueeze(1), torch.tensor(IC50_train_binary, dtype=torch.float).unsqueeze(1)
val_data_tensor, IC50_val_tensor, IC50_val_binary_tensor = torch.tensor(val_data.values, dtype=torch.float), torch.tensor(IC50_val, dtype=torch.float).unsqueeze(1), torch.tensor(IC50_val_binary, dtype=torch.float).unsqueeze(1)
train_dataset = DRPDataset(train_data_tensor, IC50_train_tensor, IC50_train_binary_tensor)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
valid_dataset = DRPDataset(val_data_tensor, IC50_val_tensor, IC50_val_binary_tensor)
test_data_tensor, test_IC50_tensor = torch.tensor(test_data.values, dtype=torch.float), torch.tensor(test_IC50, dtype=torch.float).unsqueeze(1)

model_type = ["classical",  "id", "layer_norm", "tanh", "gradual_tanh"]
model_param = {}
hyper_params_dict = {}
aucs_dict = {}
for norm in model_type:
    hyper_params_path = cv_log_path / norm / "best_hyper_params.json"
    with open(hyper_params_path, "r") as f:
        hyper_params = json.load(f)
    print(f"Training {norm} model with hyperparameters: {hyper_params}")
    if norm == "classical":
        from model.models import ClassicalNN
        model = ClassicalNN(input_size=train_data_tensor.shape[1], hyper_params=hyper_params).to(device)
    else:
        from model.models import HybridQNN
        model = HybridQNN(input_size=train_data_tensor.shape[1], hyper_params=hyper_params, norm=norm).to(device)
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=hyper_params["lr"])
    train_roc_auc_each, valid_roc_auc_each = train(model, train_loader, valid_dataset, optimizer, loss_fn, device, early_stop_patience=3, path=cv_log_path / norm)
    model_param[norm] = copy.deepcopy(model.state_dict())
    hyper_params_dict[norm] = hyper_params
    test_pred = model(test_data_tensor.to(device)).detach().cpu().numpy()
    test_roc_auc = roc_auc_score(test_IC50_tensor.cpu().numpy(), test_pred)
    aucs = evaluation.bootstrap_auc(test_IC50_tensor.cpu().numpy(), test_pred, log_dir=results_path, name=norm)
    aucs_dict[norm] = aucs
    print(f"Test ROC AUC for {norm} model: {test_roc_auc}")
evaluation.auc_summary(aucs_dict, log_dir=results_path)
model_tanh = HybridQNN(input_size=train_data_tensor.shape[1], hyper_params=hyper_params_dict["tanh"], norm="tanh").to(device)
model_gradual_tanh = HybridQNN(input_size=train_data_tensor.shape[1], hyper_params=hyper_params_dict["gradual_tanh"], norm="gradual_tanh").to(device)
model_tanh.load_state_dict(model_param["tanh"])
model_gradual_tanh.load_state_dict(model_param["gradual_tanh"])
# model_tanh, model_gradual_tanh, model_param, input_tensor, log_path
evaluation.check_value_distribution(model_tanh, train_data_tensor[:100].to(device), results_path, model_name="tanh", device=device)
evaluation.check_value_distribution(model_gradual_tanh, train_data_tensor[:100].to(device), results_path, model_name="gradual_tanh", device=device)
evaluation.ig_analyis(model_gradual_tanh, train_data, log_dir=results_path, baseline="mean", device=device)
# evaluation.run_on_real_device(model_gradual_tanh, test_data_tensor.to(device), target_class=test_IC50_tensor, device=device)
evaluation.coutour_plot(model_gradual_tanh, test_data_tensor.to(device), test_IC50_tensor, log_dir=results_path, device=device)