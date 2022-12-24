# Create and change to working directory
mkdir -p /scratch/sagarsj42 && cd $_

# Transfer the map containin all the clauses per contract
cp ../path/to/contract-clauses.json .

# Transfer and unzip the further pretrained MLM model if being used for this dataset
# scp sagarsj42@ada:/share1/sagarsj42/pretrain-bert-contr-mlm.zip .
# unzip -o -qq pretrain-bert-contr-mlm.zip

# Launch the script for contract embedding serialization
# The script will save the serialied file of contract embeddings to the working directory
python ~/contract-generation/codes/contgen/create_contract_embeddings_dataset.py
