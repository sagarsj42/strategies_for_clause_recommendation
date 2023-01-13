# Investing Strategies for Clause Recommendation

Source code for the paper, "Investigating Strategies for Clause Recommendation" published at JURIX 2022  
You may find the publication [here](https://ebooks.iospress.nl/doi/10.3233/FAIA220450).

## Reproducing the results

You'll find the scripts to run inside the ./scripts folder, and code for different components in other folders. Follow this order to prepare the workable subset from the LEDGAR corpus (download the LEDGAR dataset from [here](https://drive.switch.ch/index.php/s/j9S0GRMAbGZKa1A).) We'll be making use of the cleaned corpus.

### Preparing the subset with contract mapping
`bash ./scripts/run_prepare_clauserec_dataset.sh`

### Further pretrain a BERT or BERT-based model on the dataset
`bash ./scripts/run_bert_pretrain.sh`

### Preparing the embedding-serialized datasets
```
bash ./scripts/run_create_clauserec_emb_dataset.sh
bash ./scripts/run_create_label_embeddings_dataset.sh
```

### Preparing for training: create id files with train/dev/test splits, train a tokenizer
```
bash ./scripts/run_create_ids_file.sh
bash ./scripts/run_train_tokenizer.sh
```

### Train for clause recommendation corresponding to a strategy
`bash ./scripts/run_train_clause_decoder.sh`

### Evaluate the best training checkpoint using metrics
`bash ./scripts/run_metrics_calculator.sh`

## Citation
If you're using this, please cite the work as: 
Joshi, Sagar, et al. "Investigating Strategies for Clause Recommendation." Legal Knowledge and Information Systems. IOS Press, 2022. 73-82.
