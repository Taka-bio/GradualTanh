import os
import sys
import pathlib
import pandas as pd
import numpy as np

def response_normalization(response):
    return np.tanh((response - response.mean()) / response.std())

def preprocess_data(drug_name):
    if drug_name == "Cetuximab" or drug_name == "Erlotinib":
        data_source = "PDX"
    elif drug_name == "Docetaxel" or drug_name == "Cisplatin" or drug_name == "Gemcitabine":
        data_source = "TCGA"
    else:
        raise ValueError(f"Unknown drug name: {drug_name}")
    file_path = pathlib.Path(__file__).parent
    expr_path = file_path / "exprs_homogenized" / f"GDSC_exprs.{drug_name}.eb_with.{data_source}_exprs.{drug_name}.tsv"
    response_path = file_path / "response" / f"GDSC_response.{drug_name}.tsv"
    expr = pd.read_csv(expr_path, sep="\t", index_col=0, decimal=',').T
    response = pd.read_csv(response_path, sep="\t", index_col=0)
    
    # Extract only the samples (index) that are common between expr and response
    response.index = response.index.astype(str)
    shared_samples = list(set(expr.index) & set(response.index))
    expr = expr.loc[shared_samples]
    response = response.loc[shared_samples]

    # Normalize logIC50
    IC50 = response_normalization(response["logIC50"]).values
    IC50_binary = (response["response"] == "R").astype(int).values
    
    # save preprocessed data
    preprocessed_data_dir = file_path / "preprocessed_data" / drug_name

    # Split train_data into train and test with a ratio of 9:1
    from sklearn.model_selection import train_test_split

    train_data, test_data, IC50_train, IC50_test, IC50_train_binary, IC50_test_binary = train_test_split(expr, IC50, IC50_binary, test_size=0.1, random_state=0)
    gene_mask = train_data.var() > 0.1
    train_data = train_data.loc[:, gene_mask]
    test_data = test_data.loc[:, gene_mask]
    train_data.to_csv(preprocessed_data_dir / "train_data.csv", index=False)
    test_data.to_csv(preprocessed_data_dir / "test_data.csv", index=False)
    np.save(preprocessed_data_dir / "train_IC50.npy", IC50_train)
    np.save(preprocessed_data_dir / "test_IC50.npy", IC50_test)
    np.save(preprocessed_data_dir / "train_IC50_binary.npy", IC50_train_binary)
    np.save(preprocessed_data_dir / "test_IC50_binary.npy", IC50_test_binary)

    return 0

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess data for DRP.")
    parser.add_argument("--drug_name", type=str, required=True, help="Drug name.")
    args = parser.parse_args()

    preprocess_data(args.drug_name)