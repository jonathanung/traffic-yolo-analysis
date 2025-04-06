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

def three_way_chi_square(data:pd.DataFrame):
    yolov3_data = data.iloc[[0]]
    yolov5_data = data.iloc[[1]]
    yolov8_data = data.iloc[[2]]

    chi2, v3_v5p , _, _ = stats.chi2_contingency(pd.concat([yolov3_data,yolov5_data]))
    chi2, v3_v8p , _, _ = stats.chi2_contingency(pd.concat([yolov3_data,yolov8_data]))
    chi2, v5_v8p , _, _ = stats.chi2_contingency(pd.concat([yolov5_data,yolov8_data]))

    return [v3_v5p, v3_v8p, v5_v8p]

def main():
    models = ["3", "5", "8"]
    data_dir = Path("data/matched_csv")
    data = load_data(data_dir, models)
    _, p, _, _ = stats.chi2_contingency(data)

    pval = three_way_chi_square(data)
    print(f'''
    p-value for all: {p}
    p-value for v3 and v5: {pval[0]}
    p-value for v3 and v8: {pval[1]}
    p-value for v5 and v8: {pval[2]}
''')


if __name__ == "__main__":
    main()