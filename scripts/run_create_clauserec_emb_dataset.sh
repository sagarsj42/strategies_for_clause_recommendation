# Create a working directory to load the files and store the serialized dataset
mkdir -p working_dir && cd $_

# Transfer the two dataframe files saved by the run_prepare_clauserec_dataset.sh
cp ../path/to/clauserec-train.parquet .
cp ../path/to/clauserec-dev.parquet .

# Transfer & unzip the pretrained MLM model if used for this dataset version
cp ../path/to/pretrain-bert-contr-mlm.zip .
unzip -o -qq pretrain-bert-contr-mlm.zip

# Launch the script to store the dataset as serialized embeddings
python ../data/create_clauserec_embeddings_dataset.py

# Zip the prepared serialized dataset
zip -r clauserec-mlm.zip clauserec-mlm/
