import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matches_residuals import calculate_iou_DF, load_data
from pathlib import Path

def per_model_csvs(df, output_dir):
    for model_version, model_data in df.groupby("model"):
        try:
            output_file = output_dir / f"{model_version}_IoU.csv"
            model_data.to_csv(output_file, index=False)
            print(f"Saved IoU data for {model_version} to {output_file}")
        except Exception as e:
            print(f"Error writing {model_version} data to csv: {str(e)}")


def main():
    # create directory and list directory where results are stored
    output_dir = Path("./results/IoU_data")
    output_dir.mkdir(parents=True, exist_ok=True)
    matched_csv_dir = Path("./data/matched_csv")
    models = ["3", "5", "8"]


    # load model results and ground truth data
    IoU_data = load_data(matched_csv_dir,models,file_content="_matched.csv")
    gt_data = load_data(matched_csv_dir, models, file_content="_truth_matched.csv")

    #calculate IoU column
    IoU_data["IoU"] = calculate_iou_DF(IoU_data, gt_data)


    # export to csv
    per_model_csvs(IoU_data, output_dir)

if __name__ == "__main__":
    main()
