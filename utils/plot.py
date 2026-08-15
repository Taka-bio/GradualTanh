from matplotlib import pyplot as plt
def plot_auc_history(data, labels, path, title=""):
    for model, auc in zip(labels, data):
        plt.plot(auc, label=model)
    plt.xlabel("Epoch")
    plt.ylabel("AUC")
    plt.title("AUC history")
    plt.legend()
    plt.savefig(path / f"auc_history_{title}.png")
    plt.close()
