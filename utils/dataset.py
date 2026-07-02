# Dataset Class
from torch.utils.data import Dataset
class DRPDataset(Dataset):
    def __init__(self, expr, IC, IC_binary):
        self.expr = expr
        self.IC = IC
        self.IC_binary = IC_binary

    def __len__(self):
        return len(self.expr)

    def __getitem__(self, idx):
        return self.expr[idx], self.IC[idx], self.IC_binary[idx]