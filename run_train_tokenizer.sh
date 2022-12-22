# Create the working directory
mkdir -p working_dir && cd $working_dir

# Transfer serialized embeddings dataset, unzip the contents to get the dataset folder
cp ../path/to/clauserec-lbbase.zip .
unzip -oq clauserec-lbbase.zip

# Launch the script for training the tokenizer
# It saves the tokenizer in a file path pointed by TOKENIZER_PATH
python --use_env ./codes/contgen/train_tokenizer.py
