import os
import csv
import re


def convert_results_to_csv(yolo_path: str, yolo_models: list, dataset_names: list, output_dir: str):
    """
    Converts YOLO model results into CSV format for each model given path to YOLO results,
    list of YOLO models, and list of dataset names.
    """
    # define path to the yolo folders

    # per model create one csv
    for model in yolo_models:
        output_file_path = f"{output_dir}/{model}_output.csv"

        with open(output_file_path, 'w', newline='') as output_file:
            writer = csv.writer(output_file)
            for dataset in dataset_names:
                dataset_path = f"{yolo_path}{model}/{dataset}/labels/"

                for filename in os.listdir(dataset_path):
                    txt_path = f"{dataset_path}{filename}"

                    fileId = re.match(r".*--(\d{5})\.txt", filename).group(1)

                    with open(txt_path, 'r') as input_file:
                        rows = input_file.readlines()

                        # strip of \n character as csvwriter will handle new rows
                        # split each attribute into its on list within the row to replace the " " delimiter into a  ","
                        rows = [(f"{dataset} {fileId} " + row.strip()).split() for row in rows]

                        writer.writerows(rows)


def main():
    # create a new output directory
    os.makedirs("data/label2csv/", exist_ok=True)
    YOLO_PATH = f"data/results/"
    YOLO_MODELS = ['yolov3', 'yolov5', 'yolov8']
    DATASETS_NAMES = ['daySequence1', 'daySequence2', 'nightSequence1', 'nightSequence2']

    # call the txt_csv function to process files
    convert_results_to_csv(yolo_path=YOLO_PATH, yolo_models=YOLO_MODELS, dataset_names=DATASETS_NAMES,
                           output_dir="data/label2csv")


if __name__ == "__main__":
    main()
