#! /bin/zsh

# make sure ./data/matched_csv exists
mkdir -p ./data/matched_csv

# make sure ./data/sortedcsv exists
mkdir -p ./data/sortedcsv

# make sure lisa_processed_label.csv doesn't exist in ./data/matched_csv
if [ -f ./data/matched_csv/lisa_processed_label.csv ]; then
    rm ./data/matched_csv/lisa_processed_label.csv
fi

# Copy lisa_processed_label.csv from ./data/sortedcsv to ./data/matched_csv
cp ./data/sortedcsv/lisa_processed_label.csv ./data/matched_csv/lisa_processed_label.csv

# rename lisa_processed_label.csv to all_real_labels.csv
mv ./data/matched_csv/lisa_processed_label.csv ./data/matched_csv/all_real_labels.csv

# add header to all_real_labels.csv
echo "dataset,img_id,class_id,x_center,y_center,width,height,confidence" > ./data/matched_csv/temp.csv
cat ./data/matched_csv/all_real_labels.csv >> ./data/matched_csv/temp.csv
mv ./data/matched_csv/temp.csv ./data/matched_csv/all_real_labels.csv
