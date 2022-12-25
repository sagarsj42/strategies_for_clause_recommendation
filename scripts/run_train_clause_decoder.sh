# Create the working directory
mkdir -p working_dir && cd $_

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

# Transfer the pretrained MLM model, if using it
# cp ../path/to/pretrain-bert-contr-mlm.zip .
# unzip pretrain-bert-contr-mlm.zip

# Launch the training script, set the parameter nproc_per_node equal to the # GPUs being used
# The important parameters to be changed for trying out different experiments in the paper are commented
# The parameters to be changed for trying out different experiments are mentioned in the code file
python -m torch.distributed.launch --nproc_per_node=1 --use_env ../train/train_clause_decoder.py
