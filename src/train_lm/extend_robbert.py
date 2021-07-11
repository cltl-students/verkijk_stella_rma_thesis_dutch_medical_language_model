

import transformers
import tensorflow
import tokenizers
import torch
import sys
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
    
    #Create new vocab
    print("Initializing tokenizer...")
    #Initialize tokenizer
    from tokenizers import ByteLevelBPETokenizer
    tokenizer = ByteLevelBPETokenizer()

    print("Training tokenizer...")
    #Train tokenizer
    tokenizer.train(files=paths, vocab_size=52000, min_frequency=2, special_tokens=[
        "<s>",
        "<pad>",
        "</s>",
        "<unk>",
        "<mask>",
    ])

    # Save files to disk
    tokenizer.save_model(outfile)
    print("Saved vocab to disk.")


def get_optimizer(model, phase, args):
    """
    Initializing Adam optimizer with warm up and linear decay
    warm-up = taking a certain amount of steps to get to the peak learning rate
    decay = after the peak learning rate has been reached, start decreasing the learning rate
    """
    
    if phase == 'freeze_layers':
        optimizer_grouped_parameters = [p for p in model.parameters() if p.requires_grad == True]
    else:
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
    
  


def setArguments(model, data_collator, dataset, eval_dataset, optimizer, scheduler):
    """
    Set training arguments and initialize trainer
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
        fp16=args['fp16']
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
    
def main(args, phase):
    
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
    
    
    if args['start_from_checkpoint'] == False:
        #load RobBERT
        model = RobertaForMaskedLM.from_pretrained("pdelobelle/robbert-v2-dutch-base")
        #resize vocab
        model.resize_token_embeddings(len(tokenizer))
    else:
        model = RobertaForMaskedLM.from_pretrained(args['path_to_checkpoint'])
    
    #define data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )
    
    
    if phase == 'freeze_layers':
        #freeze all layers exept embedding/lexical layer
        for p in model.parameters():
            p.requires_grad = False
        model.get_input_embeddings().weight.requires_grad = True
    
    
    #define optimizer and scheduler
    optimizer, scheduler = get_optimizer(model, phase, args)

    #load data through dataloader class
    print('prepare data...')
    dataset = LineByLineTextDatasetRobbert(tokenizer, paths)
    eval_dataset = LineByLineTextDatasetRobbert(tokenizer, paths_eval)
   
    #initialize trainer with training arguments
    trainer = setArguments(model, data_collator, dataset, eval_dataset, optimizer, scheduler)

    #start training
    train(trainer, args['output_dir'])
    

phase = sys.argv[1]
args = json.loads(open('training_arguments.json').read())
main(args, phase)


