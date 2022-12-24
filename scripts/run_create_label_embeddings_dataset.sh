# Create the working directory
mkdir -p working_dir && cd $_

# Transfer the map from clause type to clauses for serialization
cp ../path/to/label-clauses.json .

# Launch the serialization script, it will store the serializes clause type embeddings in the working directory
python ./codes/create_label_embeddings_dataset.py
