import sys
import os
from pyspark.sql import SparkSession, Row
from pyspark.sql.types import StructType, StructField, StringType, LongType, BooleanType
from pathlib import Path

def main(sorted_csv_dir: str, matched_csv_dir: str, out_directory: str):
    """
    Uses Spark to count labels from ground truth, matched, missing, and misclassified files.
    Verifies if matched + missing count equals the total ground truth count for each model.
    Outputs a summary verification report to a text file AND a structured CSV file
    in the out_directory.
    """
    spark = SparkSession.builder.appName('Light Count Verification').getOrCreate()
    spark.sparkContext.setLogLevel('WARN')

    output_path = Path(out_directory)
    output_path.mkdir(parents=True, exist_ok=True) 
    output_txt_file_path = output_path / "verification_summary.txt"
    output_csv_file_path = output_path / "verification_counts.csv"
    results_summary_text = []
    results_for_csv = []

    ground_truth_file = Path(sorted_csv_dir) / "lisa_processed_label.csv"
    ground_truth_total_count = 0
    try:
        ground_truth_df = spark.read.csv(str(ground_truth_file), header=True, inferSchema=True)
        ground_truth_total_count = ground_truth_df.count()
        results_summary_text.append(f"Total Ground Truth Labels (from {ground_truth_file.name}): {ground_truth_total_count}\n")
        results_summary_text.append("="*40 + "\n")
    except Exception as e:
        print(f"ERROR: Could not read ground truth file {ground_truth_file}. Cannot perform verification.")
        print(f"Error details: {e}")
        results_summary_text.append(f"ERROR: Could not read ground truth file {ground_truth_file}. Cannot perform verification.\n")
        with open(output_txt_file_path, 'w') as f:
            f.writelines(results_summary_text)
        spark.stop()
        return

    # --- Process Models ---
    models = ["3", "5", "8"]
    for model_num in models:
        model_version_str = f"yolov{model_num}"
        results_summary_text.append(f"--- {model_version_str} Verification ---\n")

        matched_count = 0
        missing_count = 0
        misclassified_count = 0

        # Count Matched
        matched_file = Path(matched_csv_dir) / f"{model_version_str}_matched.csv"
        try:
            df = spark.read.csv(str(matched_file), header=True, inferSchema=True)
            matched_count = df.count()
        except Exception as e:
            print(f"  WARN: Could not read {matched_file}. Assuming count is 0. Error: {e}")
            results_summary_text.append(f"  WARN: Could not read {matched_file.name}. Count assumed 0.\n")

        # Count Missing
        missing_file = Path(matched_csv_dir) / f"{model_version_str}_missing.csv"
        try:
            df = spark.read.csv(str(missing_file), header=True, inferSchema=True)
            missing_count = df.count()
        except Exception as e:
            print(f"  WARN: Could not read {missing_file}. Assuming count is 0. Error: {e}")
            results_summary_text.append(f"  WARN: Could not read {missing_file.name}. Count assumed 0.\n")

        # Count Misclassified
        misclassified_file = Path(matched_csv_dir) / f"{model_version_str}_misclassified.csv"
        try:
            df = spark.read.csv(str(misclassified_file), header=True, inferSchema=True)
            misclassified_count = df.count()
        except Exception as e:
            print(f"  WARN: Could not read {misclassified_file}. Assuming count is 0. Error: {e}")
            results_summary_text.append(f"  WARN: Could not read {misclassified_file.name}. Count assumed 0.\n")

        # Perform Verification
        sum_matched_missing = matched_count + missing_count
        is_verified = (sum_matched_missing == ground_truth_total_count)
        verification_status = "PASSED" if is_verified else "FAILED"

        # Store results for text summary
        results_summary_text.append(f"  Matched Count:       {matched_count}\n")
        results_summary_text.append(f"  Missing Count:       {missing_count}\n")
        results_summary_text.append(f"  Sum (Matched+Missing): {sum_matched_missing}\n")
        results_summary_text.append(f"  Verification Status: {verification_status}\n")
        results_summary_text.append(f"  ----------------------------\n")
        results_summary_text.append(f"  Misclassified Count: {misclassified_count} (Informational)\n")
        results_summary_text.append("="*40 + "\n")

        # Store results for CSV
        results_for_csv.append(Row(
            model_version=model_version_str,
            matched=int(matched_count),
            missed=int(missing_count),
            total=int(sum_matched_missing),
            matches_truth=bool(is_verified),
            misclassified=int(misclassified_count)
        ))

    try:
        with open(output_txt_file_path, 'w') as f:
            f.writelines(results_summary_text)
    except Exception as e:
        print(f"ERROR: Failed to write output text file {output_txt_file_path}. Error: {e}")

    # --- Write CSV Output ---
    if results_for_csv:
        try:
            # Define schema explicitly for robustness
            schema = StructType([
                StructField("model_version", StringType(), False),
                StructField("matched", LongType(), False),
                StructField("missed", LongType(), False),
                StructField("total", LongType(), False),
                StructField("matches_truth", BooleanType(), False),
                StructField("misclassified", LongType(), False)
            ])
            # Create DataFrame from the collected Rows
            results_df = spark.createDataFrame(results_for_csv, schema=schema)

            # Write DataFrame to CSV
            results_df.coalesce(1).write.csv(str(output_csv_file_path), header=True, mode="overwrite")
        except Exception as e:
            print(f"ERROR: Failed to write output CSV file {output_csv_file_path}. Error: {e}")
    else:
        print("Skipping CSV output as no model results were processed.")


    spark.stop()
    print("Spark session stopped.")

if __name__=='__main__':
    sorted_csv_input_dir = "./data/sortedcsv"
    matched_csv_input_dir = "./data/matched_csv"
    output_dir_path = "./data/counts"

    main(sorted_csv_input_dir, matched_csv_input_dir, output_dir_path)