# Create a working directory to load the data and save outputs
mkdir -p working_dir && cd $_

# Transfer the LEDGAR cleaned corpus to working directory
cp ../path/to/LEDGAR/LEDGAR_2016-2019_clean.jsonl .

# Prepare the pretraining data
python ./codes/prepare_mlm_data.py

# Launch the pretraining script
python ./codes/pretrain_mlm.py
