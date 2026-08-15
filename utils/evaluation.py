import pathlib

import torch
import numpy as np
import pandas as pd

def bootstrap_auc(y_true, y_pred, log_dir, name="", n_bootstraps=10000, random_seed=42):
    from sklearn.utils import resample
    from sklearn.metrics import roc_auc_score
    aucs = []
    counter = 0
    for i in range(n_bootstraps):
        while True:
            y_true_sampled, y_pred_sampled = resample(y_true, y_pred, random_state=random_seed+counter)
            if len(np.unique(y_true_sampled)) < 2:
                counter += 1
                continue
            else:
                break
        aucs.append(roc_auc_score(y_true_sampled, y_pred_sampled))
        counter += 1
    with open(log_dir / f"bootstrap_auc_{name}.txt", "w") as f:
        for auc in aucs:
            f.write(f"{auc}\n")
        f.write(f"AUC: {np.mean(aucs)} ± {np.std(aucs)}\n")
    return np.array(aucs)

def auc_summary(auc_dict, log_dir):
    with open(log_dir / "auc_summary.txt", "w") as f:
        for model_name, aucs in auc_dict.items():
            f.write(f"{model_name}: {np.mean(aucs)} ± {np.std(aucs)} (95% CI: [{np.percentile(aucs, 2.5)}, {np.percentile(aucs, 97.5)}])\n")
    return 0

def check_value_distribution(model, input_tensor, log_path, model_name="tanh", device="cuda"):
    encoder = model.encoder
    encoder.eval()
    with torch.no_grad():
        output = encoder(input_tensor.to(device))
        output_normalized = model.norm(output)
    # plot the distribution of the outputs
    import matplotlib.pyplot as plt
    x = np.linspace(-20, 20, 100)
    y = model.norm(torch.tensor(x, dtype=torch.float).to(device)).cpu().numpy()
    plt.figure(figsize=(10, 5))
    plt.plot(x, y, label=model_name)
    plt.title(f"Max: {output_normalized.max():.4f}, Min: {output_normalized.min():.4f}")
    plt.scatter(output.cpu().numpy().flatten(), output_normalized.cpu().numpy().flatten())
    plt.xlabel("Values before normalization", fontsize=14)
    plt.ylabel("Values after normalization", fontsize=14)
    plt.savefig(log_path / f"{model_name}_distribution.png")
    plt.close()

    return 0

def ig_analyis(model, input_data, log_dir, baseline=None, device="cuda"):
    from captum.attr import IntegratedGradients
    input_tensor = torch.tensor(input_data.values, dtype=torch.float)
    model.eval()
    ig = IntegratedGradients(model)
    if baseline is None:
        baseline = torch.zeros_like(input_tensor).to(device)
    elif baseline == "mean":
        baseline = input_tensor.mean(dim=0, keepdim=True).expand_as(input_tensor).to(device)
    attributions = ig.attribute(input_tensor.to(device), baselines=baseline)

    ig_values_mean = np.mean(attributions.cpu().numpy(), axis=0)
    ig_values_abs_mean = np.mean(np.abs(attributions.cpu().numpy()), axis=0)

    top_n = int(0.05 * len(ig_values_mean))
    top_indices = np.argsort(ig_values_abs_mean)[-top_n:][::-1]
    ig_pos = np.where(ig_values_mean > 0)[0]
    ig_neg = np.where(ig_values_mean < 0)[0]
    top_ig_pos = [idx for idx in top_indices if idx in ig_pos]
    top_ig_neg = [idx for idx in top_indices if idx in ig_neg]

    data_dir = pathlib.Path(__file__).parent.parent / "data"

    gene_info = pd.read_csv(data_dir / "Homo_sapiens.gene_info", sep="\t", index_col=0)
    gene_info.index = gene_info["GeneID"]

    selected_pos_features = [int(input_data.columns[idx]) for idx in top_ig_pos if int(input_data.columns[idx]) in gene_info["GeneID"].values]
    selected_neg_features = [int(input_data.columns[idx]) for idx in top_ig_neg if int(input_data.columns[idx]) in gene_info["GeneID"].values]
    pos_gene_names = gene_info.loc[selected_pos_features, "Symbol"].values
    neg_gene_names = gene_info.loc[selected_neg_features, "Symbol"].values

    with open(log_dir / "ig_positive_genes.txt", "w") as f:
        for gene in pos_gene_names:
            f.write(f"{gene}\n")

    with open(log_dir / "ig_negative_genes.txt", "w") as f:
        for gene in neg_gene_names:
            f.write(f"{gene}\n")

    return 0

def coutour_plot(model, input_tensor, target_binary, log_dir, device="cuda"):
    from sklearn.decomposition import PCA
    model.eval()
    embedding = model.norm(model.encoder(input_tensor.to(device))).cpu().detach().numpy()
    pca = PCA(n_components=2)
    reduced_embedding = pca.fit_transform(embedding)
    explained_variance_ratio = pca.explained_variance_ratio_
    print(f"Explained variance ratio of the first two principal components: {explained_variance_ratio}")
    s1 = np.std(reduced_embedding[:, 0])
    s2 = np.std(reduced_embedding[:, 1])
    x_range = np.linspace(-4 * s1, 4 * s1, 200)
    y_range = np.linspace(-4 * s2, 4 * s2, 200)
    X, Y = np.meshgrid(x_range, y_range)
    center_pca = (np.max(reduced_embedding, axis=0) + np.min(reduced_embedding, axis=0)) / 2
    grid_pca_coords = np.stack([X + center_pca[0], Y + center_pca[1]], axis=-1)
    grid_embeddings = pca.inverse_transform(grid_pca_coords.reshape(-1, 2)).reshape(X.shape[0], X.shape[1], -1)
    grid_embeddings_tensor = torch.tensor(grid_embeddings, dtype=torch.float32).to(device)
    grid_embeddings_tensor_flat = grid_embeddings_tensor.view(-1, model.num_encoding_params)
    prediction = torch.stack(model.quantum_net(grid_embeddings_tensor_flat.T, model.q_weights.to(device))).T.reshape(X.shape[0], X.shape[1], -1).float().to(device)
    prediction = model.fc(prediction).squeeze(-1)
    import matplotlib.pyplot as plt
    cs = plt.contourf(X + center_pca[0], Y + center_pca[1], prediction.cpu().detach().numpy(), levels=20, cmap="RdBu", alpha=0.7)
    plt.colorbar(cs, label="Predicted values")
    plt.scatter(reduced_embedding[:, 0], reduced_embedding[:, 1], c=target_binary, cmap="RdBu", edgecolor="k", alpha=0.7)
    plt.xlabel("PC1", fontsize=15)
    plt.ylabel("PC2", fontsize=15)
    plt.savefig(log_dir / "contour_plot.png")
    plt.close()
    return 0

def run_on_real_device(model, input_tensor, target_class, device="cuda"):
    from pytket import Circuit
    from pytket.pauli import QubitPauliString, Pauli
    from pytket.circuit import Qubit
    from pytket.utils.operators import QubitPauliOperator
    import qnexus as qnx

    model.eval()
    encoded = model.encoder(input_tensor.to(device))
    encoded_norm = model.norm(encoded)

    qnx.login()

    project = qnx.projects.get_or_create(name="GradualTanh-real-device")
    qnx.context.set_active_project(project)

    refs = []
    num_qubits = model.hyper_params["num_qubits"]
    num_encoding_layer = model.hyper_params["num_encoding_layer"]
    reps = model.hyper_params["reps"]
    for encode in encoded_norm:
        c = Circuit(num_qubits, num_qubits)
        q = [Qubit(i) for i in range(num_qubits)]

        combined_op = QubitPauliOperator()
        for i in range(num_qubits):
            combined_op[QubitPauliString({Qubit(i): Pauli.Z})] = 1.0

        param_num = 0
        for i in range(num_encoding_layer):
            for j in range(num_qubits):
                c.Rz(encode[param_num].item()/np.pi, j)
                c.Rx(model.q_weights[param_num].item()/np.pi, j)
                param_num += 1
            for j in range(num_qubits):
                c.CX(j, (j + 1) % num_qubits)
        for i in range(reps):
            for j in range(num_qubits):
                c.Rx(model.q_weights[param_num].item()/np.pi, j)
                param_num += 1
            for j in range(num_qubits):
                c.CX(j, (j + 1) % num_qubits)

        for i in range(num_qubits):
            c.Measure(i, i) 

        # compileが終わっていない場合
        import datetime
        jobname_suffix = datetime.datetime.now().strftime("%Y_%m_%d-%H-%M-%S-%f")
        ref = qnx.circuits.upload(circuit=c, project=project, name=f"Circuit-{jobname_suffix}")
        refs.append(ref)
    
    config = qnx.QuantinuumConfig(device_name="reimei")
    import datetime
    jobname_suffix = datetime.datetime.now().strftime("%Y_%m_%d-%H-%M-%S-%f")
    ref_compile_job = qnx.start_compile_job(
        programs=refs,
        backend_config=config,
        optimisation_level=2,
        name=f"compilation-job-{jobname_suffix}",
        project=project,
    )
    qnx.jobs.wait_for(ref_compile_job)
    ref_compiled_circuit = [qnx.jobs.results(ref_compile_job)[i].get_output() for i in range(len(encoded_norm))]

    ref_execute_job = qnx.start_execute_job(
        programs=ref_compiled_circuit,
        n_shots=[500 for _ in range(len(ref_compiled_circuit))],
        backend_config=config,
        name=f"execution-job-{jobname_suffix}",
        project=project,
    )
    qnx.jobs.wait_for(ref_execute_job)

    ref_result = qnx.jobs.results(ref_execute_job)
    import random
    for result in ref_result:
        backend_result = result.download_result()
        import inspect
        counts = backend_result.get_counts()

        current_pred = []
        total_shots = sum(counts.values())
        for i in range(num_qubits):
            exp_val_z = 0.0
            for bitstring, count in counts.items():
                if bitstring[i] == 0:
                    exp_val_z += count
                else:
                    exp_val_z -= count 
            
            exp_val_z /= total_shots
            current_pred.append(exp_val_z)
        test_pred.append(current_pred)

    from sklearn.metrics import roc_auc_score
    test_pred = torch.Tensor(test_pred)
    # print(f"Test predictions: {test_pred}")
    test_pred = model.fc(test_pred.to(device)).cpu().detach().numpy().flatten()
    print(f"Test predictions: {test_pred}")
    print(f"AUC: {roc_auc_score(target_class.detach().numpy(), test_pred)}")
    
    return 0