import os
os.chdir('working_dir')

import warnings

import torch

from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_MASKED_LM_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    PreTrainedTokenizer,
    TrainingArguments,
    AutoConfig,
    AutoTokenizer,
    AutoModelWithLMHead,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    LineByLineTextDataset,
    TextDataset,
    DataCollatorForLanguageModeling,
    DataCollatorForWholeWordMask,
    DataCollatorForPermutationLanguageModeling,
    PretrainedConfig,
    Trainer,
    set_seed
)

set_seed(123)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ModelDataArguments(object):
    def __init__(self, train_data_file, eval_data_file=None, line_by_line=False, 
                 mlm=False, mlm_probability=0.15, whole_word_mask=False, plm_probability=float(1.0/6), 
                 max_span_length=5, block_size=-1, overwrite_cache=False, model_type=None, model_config_name=None, 
                 tokenizer_name=None, model_name_or_path=None, model_cache_dir=None):
        
        if 'CONFIG_MAPPING' not in globals():
            raise ValueError('Could not find `CONFIG_MAPPING` imported! \
                             Make sure to import it from `transformers` module!')
            
        if (model_type is not None) and (model_type not in CONFIG_MAPPING.keys()):
            raise ValueError('Invalid `model_type`! Use one of the following: %s' \
                             % str(list(CONFIG_MAPPING.keys())))
                             
        if not any([model_type, model_config_name, model_name_or_path]):
            raise ValueError('You can\'t have all - `model_type`, `model_config_name`, \
                             `model_name_or_path` be `None`! You need to have at least one of them set!')
                             
        if not any([model_config_name, model_name_or_path]):
            warnings.formatwarning = lambda message,category,*args,**kwds: \
                             '%s: %s\n' % (category.__name__, message)
            warnings.warn('You are planning to train a model from scratch! 🙀')
                             
        if not any([tokenizer_name, model_name_or_path]):
            raise ValueError('Tokenizer cannot be trained from scratch here! \
                             You can train your own tokenizer separately and use path here to load it!')
                             
        self.train_data_file = train_data_file
        self.eval_data_file = eval_data_file
        self.line_by_line = line_by_line
        self.mlm = mlm
        self.mlm_probability = mlm_probability
        self.whole_word_mask = whole_word_mask
        self.plm_probability = plm_probability
        self.max_span_length = max_span_length
        self.block_size = block_size
        self.overwrite_cache = overwrite_cache
        
        self.model_type = model_type
        self.model_config_name = model_config_name
        self.tokenizer_name = tokenizer_name
        self.model_name_or_path = model_name_or_path
        self.model_cache_dir = model_cache_dir
        
        return
    

def get_model_config(args: ModelDataArguments):
    if args.model_config_name is None:
        model_config = AutoConfig.from_pretrained(args.model_config_name, 
                                                  cache_dir=args.model_cache_dir)
    elif args.model_name_or_path is not None:
        model_config = AutoConfig.from_pretrained(args.model_name_or_path, 
                                                  cache_dir=args.model_cache_dir)
    else:
        model_config = CONFIG_MAPPING[args.model_type]()
        
    if (model_config.model_type in ['bert', 'roberta', 'distilbert', 'camembert']) and (args.mlm is False):
        raise ValueError('BERT and RoBERTa-like models do not have LM heads but \
        masked LM heads. They must be run setting `mlm=True`')
        
    if model_config.model_type == 'xlnet':
        args.block_size = 512
        model_config.mem_len = 1024
        
    return model_config


def get_tokenizer(args: ModelDataArguments):
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, \
                                                  cache_dir=args.model_cache_dir)
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path,
                                                 cache_dir=args.model_cache_dir)
    if args.block_size <= 0:
        args.block_size = tokenizer.model_max_length
    else:
        args.block_size = min(args.block_size, tokenizer.model_max_length)
        
    return tokenizer


def get_model(args: ModelDataArguments, model_config):
    if ('MODEL_FOR_MASKED_LM_MAPPING' not in globals()) and ('MODEL_FOR_CAUSAL_LM_MAPPING' not in globals):
        raise ValueError('Could not find `MODEL_FOR_MASKED_LM_MAPPING` and \
                         `MODEL_FOR_CAUSAL_LM_MAPPING` imported! Make sure \
                         you import them from `transformers` module.')
    
    if args.model_name_or_path:
        if type(model_config) in MODEL_FOR_MASKED_LM_MAPPING.keys():
            return AutoModelForMaskedLM.from_pretrained(
                args.model_name_or_path,
                from_tf=bool('.ckpt' in args.model_name_or_path),
                config=model_config,
                cache_dir=args.model_cache_dir
            )
        elif type(model_config) in MODEL_FOR_CAUSAL_LM_MAPPING.keys():
            return AutoModelForCausalLM.from_pretrained(
                args.model_name_or_path,
                from_tf=bool('.ckpt' in args.model_name_or_path),
                config=model_config,
                cache_dir=args.model_cache_dir
            )
        else:
            raise ValueError('Invalid `model_name_or_path`! It should be in %s or %s' \
                             % (str(MODEL_FOR_MASKED_LM_MAPPING.keys()), \
                                str(MODEL_FOR_CAUSAL_LM_MAPPING.keys())))
    else:
        print('Training new model from scratch!')
        return AutoModelWithLMHead.from_config(model_config)
    

def get_dataset(args: ModelDataArguments, tokenizer: PreTrainedTokenizer, evaluate: bool=False):
    file_path = args.eval_data_file if evaluate else args.train_data_file
    
    if args.line_by_line:
        return LineByLineTextDataset(tokenizer=tokenizer, file_path=file_path, block_size=args.block_size)
    else:
        return TextDataset(tokenizer=tokenizer, file_path=file_path, 
                            block_size=args.block_size, overwrite_cache=args.overwrite_cache)
    
    
def get_collator(args: ModelDataArguments, model_config: PretrainedConfig, tokenizer: PreTrainedTokenizer):
    if model_config.model_type == 'xlnet':
        return DataCollatorForPermutationLanguageModeling(
            tokenizer=tokenizer,
            plm_probability=args.plm_probability,
            max_span_length=args.max_span_length
        )
    else:
        if args.mlm and args.whole_word_mask:
            return DataCollatorForWholeWordMask(
                tokenizer=tokenizer,
                mlm_probability=args.mlm_probability
            )
        else:
            return DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=args.mlm,
                mlm_probability=args.mlm_probability
            )
        

model_data_args = ModelDataArguments(
    train_data_file='ledgar-clause-sent-train.txt',
    eval_data_file='ledgar-clause-sent-eval.txt',
    line_by_line=True,
    mlm=True,
    whole_word_mask=True,
    mlm_probability=0.15,
    plm_probability=float(1.0/6),
    max_span_length=5,
    block_size=50,
    overwrite_cache=False,
    model_type='bert',
    model_config_name='bert-base-uncased',
    tokenizer_name='bert-base-uncased',
    model_name_or_path='bert-base-uncased',
    model_cache_dir=None
)

training_args = TrainingArguments(
    output_dir='pretrain-bert-contr-mlm',
    overwrite_output_dir=True,
    do_train=True,
    do_eval=True,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=128,
    evaluation_strategy='steps',
    logging_steps=700,
    eval_steps=10000,
    prediction_loss_only=True,
    learning_rate=5e-5,
    weight_decay=0,
    adam_epsilon=1e-8,
    max_grad_norm=1.0,
    num_train_epochs=2,
    save_steps=-1
)

print('Loading model configuration')

config = get_model_config(model_data_args)

print('Loading model tokenizer')

tokenizer = get_tokenizer(model_data_args)

print('Loading actual model')

model = get_model(model_data_args, config)
model.resize_token_embeddings(len(tokenizer))

print('Creating train dataset')

train_dataset = get_dataset(model_data_args, tokenizer=tokenizer, 
                            evaluate=False) if training_args.do_train else None

print('Creating eval dataset')

eval_dataset = get_dataset(model_data_args, tokenizer=tokenizer,
                          evaluate=True) if training_args.do_eval else None

data_collator = get_collator(model_data_args, config, tokenizer)

if (len(train_dataset) // training_args.per_device_train_batch_size // training_args.logging_steps \
    * training_args.num_train_epochs) > 1000:
    warnings.warn('Your logging steps will do a lot of printing! \
                  Consider increasing `logging_steps` t avoid overflowing the logs with a lot of prints.')
    
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset
)

if training_args.do_train:
    print('Start training')
    
    model_path = (model_data_args.model_name_or_path 
                  if model_data_args.model_name_or_path is not None and 
                      os.path.isdir(model_data_args.model_name_or_path) 
                  else None)
    
    trainer.train(model_path=model_path)
    
    trainer.save_model()
    
if trainer.is_world_process_zero():
    tokenizer.save_pretrained(training_args.output_dir)
