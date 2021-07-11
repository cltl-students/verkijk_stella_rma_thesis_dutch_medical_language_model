

import tensorflow
import tokenizers
import torch
import transformers
import json
from typing import Dict, List, Optional
from pathlib import Path
from torch.utils.data.dataset import Dataset
from tokenizers.processors import RobertaProcessing
from adapted_robbert_class import LineByLineTextDatasetRobbert
from tokenizers import ByteLevelBPETokenizer


from transformers import (
    RobertaConfig, 
    RobertaTokenizerFast,
    RobertaTokenizer,
    RobertaForMaskedLM,
    LineByLineTextDataset,
    TextDataset,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments
    )

def checkRequirements():
    print("Checking requirements...")
    assert torch.cuda.is_available()

def trainTokenizer(paths, outfile):
    """
    Trains a tokenizer
    Saves a vocabulary
    """
    
    print("Initializing tokenizer...")
    from tokenizers import ByteLevelBPETokenizer
    tokenizer = ByteLevelBPETokenizer()

    print("Training tokenizer...")
    tokenizer.train(files=paths, vocab_size=52000, min_frequency=2, special_tokens=[
        "<s>",
        "<pad>",
        "</s>",
        "<unk>",
        "<mask>",
    ])

    tokenizer.save_model(outfile)
    print("Saved vocab to disk.")

def setConfig():
    """
    Set model configurations
    """
    print("Setting RoBERTa configurations for model...")
    config = RobertaConfig(
        vocab_size=52000,
        max_position_embeddings=514,
        num_attention_heads=12,
        num_hidden_layers=12,
        type_vocab_size=1,
    )
    
    return(config)


def get_optimizer(model, args):
    """
    Initializing Adam optimizer with warm up and linear decay
    warm-up = taking a certain amount of steps to get to the peak learning rate
    decay = after the peak learning rate has been reached, start decreasing the learning rate
    :param model: initialized transformer model
    :param args: dict
    """
    
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.01,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    
    optimizer = transformers.AdamW(optimizer_grouped_parameters, lr = args['lr'], betas = args['betas'], weight_decay = args['weight_decay'], eps = args['eps']) 
    scheduler = transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps = args['num_warmup_steps'], num_training_steps = args['num_training_steps']) 
    
    return(optimizer, scheduler)

def setArguments(model, data_collator, dataset, eval_dataset, optimizer, scheduler, args):
    """
    Set training arguments and initialize trainer
    :param args: dictionary
    """

    print("Setting training arguments...")
    training_args = TrainingArguments(
        output_dir=args['output_dir'],
        overwrite_output_dir=args['overwrite_output_dir'],
        num_train_epochs=args['num_train_epochs'],
        per_device_train_batch_size=args['per_device_train_batch_size'],
        per_device_eval_batch_size=args['per_device_eval_batch_size'],
        do_eval = args['do_eval'], #evaluate on small dataset to calculate loss
        do_train = args['do_train'], 
        evaluation_strategy = args['evaluation_strategy'],
        eval_steps = args['eval_steps'], #do evaluation on small dataset each x amount of steps
        save_steps= args['save_steps'], #save a checkpoint of the model each x amount of steps
        save_total_limit= args['save_total_limit'], #save at most 100 checkpoints
        logging_steps = args['logging_steps'], #save loss and learning rate to log every x amount of steps
        logging_dir = args['logging_dir'],
        gradient_accumulation_steps = args['gradient_accumulation_steps'], 
        eval_accumulation_steps = args['eval_accumulation_steps'],
        fp16=args['fp16'],
        weight_decay = args['weight_decay']
    )
    
    print("initializing trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
        eval_dataset = eval_dataset,
        optimizers = (optimizer, scheduler)
    )

    return(trainer)

def train(trainer, path_to_model):
    print("Start training...")
    trainer.train()

    trainer.save_model(path_to_model)
    print("Saved model to disk.")
    
    
def main(args):
    """
    Train tokenizer, create vocab, set configuration, initialize model, set data collator, initialize optimizer and scheduler, load line by line dataset, define trainer, train model, save model
    :param args: dictionary
    """
    
    #define paths to train and eval data
    paths = [str(x) for x in Path(args['path_to_traindata_folder']).glob("*.txt")]
    paths_eval = [str(x) for x in Path(args['path_to_eval']).glob("*.txt")]
    
    if args['train_tokenizer'] == True:
        #train a tokenizer and create a vocab 
        trainTokenizer(paths, args['outfile_tokenizer'])
        tokenizer = RobertaTokenizer.from_pretrained(args['outfile_tokenizer'], max_length=512, padding=True, truncation=True)
    else:
        #initialize trained tokenizer
        tokenizer = RobertaTokenizer.from_pretrained(args['outfile_tokenizer'], max_length=512, padding=True, truncation=True)
    
    #initialize model and set model configurations
    config = setConfig()
    
    if args['start_from_checkpoint'] == False:
        model = RobertaForMaskedLM(config=config)
    else:
        model = RobertaForMaskedLM(args['path_to_checkpoint'])
    
    #define data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )
    
    #define optimizer and scheduler
    optimizer, scheduler = get_optimizer(model, args)
    
    #preprocess data
    print('prepare data...')
    dataset = LineByLineTextDatasetRobbert(tokenizer, paths)
    eval_dataset = LineByLineTextDatasetRobbert(tokenizer, paths_eval)

   
    #initialize trainer with training arguments
    trainer = setArguments(model, data_collator, dataset, eval_dataset, optimizer, scheduler, args)

    train(trainer, args['output_dir'])
    

args = json.loads(open('training_arguments.json').read())
main(args)
