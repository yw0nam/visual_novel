import pandas as pd
import argparse
from custom_dataset import *
from transformers import Trainer
from transformers import TrainingArguments
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.model_selection import train_test_split

def define_argparser():
    p = argparse.ArgumentParser()
    p.add_argument('--csv_path', required=True)
    p.add_argument('--gradient_accumulation_steps', type=int, default=2)
    p.add_argument('--valid_ratio', type=float, default=.2)
    p.add_argument('--batch_size_per_device', type=int, default=48)
    p.add_argument('--n_epochs', type=int, default=5)
    p.add_argument('--model', type=str, default='gpt2')
    p.add_argument('--warmup_ratio', type=float, default=.2)
    p.add_argument('--max_length', type=int, default=512)
    p.add_argument('--load_weight', default=None)

    config = p.parse_args()

    return config

def main(config):
    
    tokenizer = AutoTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-v2')
    model = AutoModelForSequenceClassification.from_pretrained('cl-tohoku/bert-base-japanese-v2', num_labels=2)

    csv = pd.read_csv(config.csv_path)
    train, val = train_test_split(csv, test_size=0.2, random_state=1004, stratify=csv['use'])
    train_dataset = MyDataset(train)
    val_dataset = MyDataset(val)
    
    print(
        '|train| =', len(train_dataset),
        '|valid| =', len(val_dataset),
    )

    total_batch_size = config.batch_size_per_device * torch.cuda.device_count()
    n_total_iterations = int(len(train_dataset) / total_batch_size * config.n_epochs)
    n_warmup_steps = int(n_total_iterations * config.warmup_ratio)
    print(
        '#total_iters =', n_total_iterations,
        '#warmup_iters =', n_warmup_steps,
    )
    
    training_args = TrainingArguments(
        output_dir='./model/checkpoints',
        num_train_epochs=config.n_epochs,
        per_device_train_batch_size=config.batch_size_per_device,
        per_device_eval_batch_size=config.batch_size_per_device,
        warmup_steps=n_warmup_steps,
#         weight_decay=0.01,
        fp16=True,
        evaluation_strategy='epoch',
        logging_steps=n_total_iterations // 100,
        save_strategy ='epoch',
#         save_steps=n_total_iterations // config.n_epochs,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        load_best_model_at_end=True
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=dataCollator(tokenizer,
                                  config.max_length,
                                  with_text=False),
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    trainer.train()
    
    trainer.model.save_pretrained(config.model_fn)
    torch.save({
        # 'model': trainer.model.state_dict(),
        'config': config,
        # 'vocab': None,
        # 'classes': None,
        # 'tokenizer': tokenizer,
    }, config.model_fn + 'train_config.json')


if __name__ == '__main__':
    config = define_argparser()
    main(config)