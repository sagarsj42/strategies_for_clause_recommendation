# Create the working directory
mkdir -p working_dir && cd $_

# Transfer the map from clause type to clauses
cp ../path/to/label-clauses.json .

# Transfer serialized embeddings dataset corresponding to a base encoder
# Unzip the contents to get the dataset folder
# Ensure to use the files corresponding to the same for ids file, clause type (label) embeddings, full contract embeddings
cp ../path/to/clauserec-lbbase.zip .
unzip -oq clauserec-lbbase.zip

# Transfer the ids file containing the train, dev and test splits
cp ../path/to/clauserec-lbbase-3split-ids.json .

# Transfer the serialized embeddings for clause types
cp ../path/to/label-embeddings-lbbase.pkl .

# Transfer the serialized embeddings for full contracts
cp ../path/to/contract-embeddings-lbbase.pkl .

# Transfer the serialized embeddings for clause type-wise clause embeddings from contracts
cp ../path/to/simclause-embeddings-lbbase.pkl .

# Transfer the trained worpiece tokenizer file
cp ../path/to/clauserec-tokenizer-wordpiece.json .

# Transfer the further pretrained MLM model, if using it
# cp ../path/to/pretrain-bert-contr-mlm.zip .
# unzip pretrain-bert-contr-mlm.zip

# Transfer the experiment folder
cp ../path/to/experiment-name.zip .
unzip -oq experiment-name.zip

# Launch the evalaution script
# The script will save a file names experiment-name.results.json in the working directory
python ../eval/metrics_calculator.py \
    --DATA_PATH legal-contracts-clauserec-mlm \ # serialized clause recommendation dataset
    --MODEL_PATH experiment-name \ # experiment folder
    --N_EPOCHS 20 \ # epochs the model is trained on
    --BATCH_SIZE 8 \ # batch size to be used for evaluation
    --RETRIEVE_K 6 # no. of similar contracts to be retrieved
