import os
import csv

def txt_csv():
    # define path to the yolo folders
    yolo_path = f"data/pre_analysis/"

    yolo_models = ['yolov3', 'yolov5', 'yolov8']
    dataset_names = ['daySequence1', 'daySequence2', 'nightSequence1', 'nightSequence2']

    # per model create one csv
    for model in yolo_models:
        output_file_path = f"data/postprocessing/{model}_output.csv"

        with open(output_file_path, 'w', newline='') as output_file:
            writer = csv.writer(output_file)
            for dataset in dataset_names:
                dataset_path = f"{yolo_path}{model}/{dataset}/labels/"

                for filename in os.listdir(dataset_path):
                    txt_path = f"{dataset_path}{filename}"

                    with open(txt_path, 'r') as input_file:
                        rows = input_file.readlines()
                        # strip of \n character as csvwriter will handle new rows
                        # split each attribute into its on list within the row to replace the " " delimiter into a  ","
                        rows = [(f"{filename} " + row.strip()).split() for row in rows]


                        writer.writerows(rows)




def main():
    # create a new output directory
    os.makedirs("data/postprocessing/", exist_ok=True)

    # call the txt_csv function to process files
    txt_csv()


if __name__ == "__main__":
    main()
