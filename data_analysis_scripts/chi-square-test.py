import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn  as sns
from pathlib import Path
from scipy import stats

def load_data(data_dir:Path, models:list,):
    categories = {"matched": "_matched.csv", "misclassified": "_misclassified.csv", "missing": "_missing.csv"}
    dfs = []
    for model in models:
        model_data = pd.DataFrame(columns=list(categories.keys()), index=[f"yolov{model}"])
        for category, file_type in categories.items():
            df = pd.read_csv(data_dir / f"yolov{model}{file_type}")
            model_data[category] = df.shape[0]

        dfs.append(model_data)

    dfs = pd.concat(dfs)
    return dfs

def main():
    models = ["3", "5", "8"]
    data_dir = Path("data/matched_csv")
    data = load_data(data_dir, models)
    print(data.aslist)
    chi2, p, dof, expected = stats.chi2_contingency(data.values)
    print(f"Chi-squared: {chi2}, p-value: {p}, degrees of freedom: {dof}, expected: {expected}")


if __name__ == "__main__":
    main()