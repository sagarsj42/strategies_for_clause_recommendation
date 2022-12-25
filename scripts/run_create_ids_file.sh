# Create the working directory
mkdir -p working_dir && cd $_

# Transfer the map from clause type to clauses
cp ../path/to/label-clauses.json .

# Transfer serialized embeddings dataset corresponding to a base encoder
# Unzip the contents to get the dataset folder
cp ../path/to/clauserec-lbbase.zip .
unzip -oq clauserec-lbbase.zip

# Run the script to create an ids map corresponding to the serialized dataset used
# The output will be a .json file containing the train, dev and test ids
python ../data/create_split_3_ids.py
