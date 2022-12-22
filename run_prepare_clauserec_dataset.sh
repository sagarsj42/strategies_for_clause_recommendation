# Create the working directory
mkdir -p working_dir && cd $_

# Transfer the LEDGAR cleaned corpus to working directory
cp ../path/to/LEDGAR/LEDGAR_2016-2019_clean.jsonl .

# Run the script for preparing the subset for clause recommendation and other auxiliary files
python ./codes/clause_prediction_dataset_from_ledgar.py
