import pandas as pd
import re
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from joblib import dump, load
from sklearn import tree
import numpy as np
import itertools
import pickle

from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from datasets import load_dataset

from datasets import Dataset, DatasetDict

import argparse
from transformers import enable_full_determinism

from transformers import AutoTokenizer
from torch.utils.data import DataLoader

from transformers import RobertaForSequenceClassification, AutoModelForSequenceClassification, AutoModelForCausalLM
from transformers import TrainingArguments
from transformers import Trainer
from transformers import pipeline
import numpy as np
import logging
import argparse
import evaluate
import datasets
import random
import os
import shutil
import torch
import tqdm
from glob import glob
import wandb
import copy
import json
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer#, DataCollatorForCompletionOnlyLM
from trl import apply_chat_template


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

parser = argparse.ArgumentParser(description='Run training and evaluation with pre-generated LLM data.')

parser.add_argument('--seed', type=int, const=0, default=0, nargs='?',
                    help='Seed to be used for shuffling.')
parser.add_argument('--repeat', type=int, const=10, default=10, nargs='?',
                    help='How many times should the process be repeated.')

parser.add_argument('--batch_size', type=int, const=32, default=32, nargs='?',
                    help='Traing batch size.')
parser.add_argument('--epochs', type=int, const=50, default=50, nargs='?',
                    help='No. epochs for each training.')
parser.add_argument('--shots', type=int, const=2, default=2, nargs='?',
                    help='Number of shots to use.')
parser.add_argument('--size', type=int, const=10, default=10, nargs='?',
                    help='Number of labelled samples to use.')
parser.add_argument('--lr', type=float, const=1e-5, default=1e-5, nargs='?',
                    help='Learning rate to use.')

parser.add_argument('--batch_size_eval', type=int, const=256, default=256, nargs='?',
                    help='Eval batch size.')

parser.add_argument('--results_dir', type=str,
                    help='Directory where result csvs are saved to.')

parser.add_argument('--experiment_type', type=str,
                    help='Specify what kind of mode to use.')
parser.add_argument('--model', type=str,
                    help='Specify which model to use.')
parser.add_argument('--task', type=str,
                    help='Specify which dataset use to train.')
parser.add_argument('--language', type=str,
                    help='Specify which language to use.')
parser.add_argument('--eval_dataset', type=str,
                    help='Specify which dataset use to evaluate.')

args = parser.parse_args()

model_mapper = {
    'roberta': 'FacebookAI/xlm-roberta-base',
    'roberta-large': 'FacebookAI/xlm-roberta-large',
    'llama3': 'meta-llama/Meta-Llama-3.1-8B-Instruct',
    'llama3-large': 'meta-llama/Meta-Llama-3-70B-Instruct',
    'gemma': 'google/gemma-3-4b-it',
    'qwen': 'Qwen/Qwen2.5-7B-Instruct',
}

MODEL = args.model
MODEL_NAME = model_mapper[MODEL]
BATCH_SIZE = args.batch_size
NUM_EPOCHS = args.epochs
TASK = args.task
EXPERIMENT_TYPE = args.experiment_type
SIZE = args.size
SHOTS = args.shots
LANGUAGE = args.language
LR = args.lr

CLASS_MAPPER = {
    'sentiment': ['negative', 'positive'],
    'topic': ['science and technology', 'travel', 'politics', 'sports', 'health', 'entertainment', 'geography'],
    'intent': ['alarm query', 'audio volune down', 'calendar remove', 'cooking recipe', 'datetime convert', 'send email', 'play audiobook', 'movie recommendation', 'transport ticket', 'weather query'],
    'sarcasm': ['serious', 'sarcastic']
}

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

from transformers import AutoTokenizer

print(f'Running {EXPERIMENT_TYPE} with {MODEL}!')

def load_data(size):
    if HUMAN:
        df_train = pd.read_csv(os.path.join('data', TASK, f'{args.language}-train_human_parsed.csv'))
        df_train = df_train[['text', 'label']]
        df_train.dropna(inplace=True)
    else:
        df_train = pd.read_csv(os.path.join('data', TASK, f'{args.language}-train_full.csv'))

    num_classes = len(CLASS_MAPPER[TASK])
    to_choose = max(2, int(size/num_classes))
    targets = np.array(df_train.label)
    sub_dfs = []
    for cls in range(num_classes):
        inds = np.argwhere(targets == cls).reshape(-1)
        indices = torch.randperm(len(inds))
        inds = inds[indices]
        inds = inds[:to_choose]
        sub_dfs.append(df_train.iloc[inds])
    df_train = pd.concat(sub_dfs)
    print(df_train)
    
    # df_train, _ = train_test_split(df_train, train_size=SIZE, shuffle=True, stratify=df_train['label'])
    print(f'Training dataset size is: {df_train.shape}')
    df_test = pd.read_csv(os.path.join('data', TASK, f'{args.language}-test.csv'))
    print(f'Testing dataset size is: {df_test.shape}')
    return df_train.reset_index(), df_test.reset_index()

def prepare_ft_dataset(df_train, df_test):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_dataset = Dataset.from_pandas(df_train[['text', 'label']])
    test_dataset = Dataset.from_pandas(df_test[['text', 'label']])
    tokenized_train = tokenizer([str(text) for text in train_dataset["text"]], padding=True, return_tensors='pt', truncation=True, max_length=256, add_special_tokens=True)
    tokenized_test = tokenizer([str(text) for text in test_dataset["text"]], padding=True, return_tensors='pt', truncation=True, max_length=256, add_special_tokens=True)
    tokenized_train['label'] = train_dataset['label']
    tokenized_train['text'] = train_dataset['text']
    tokenized_test['label'] = test_dataset['label']
    tokenized_test['text'] = test_dataset['text']
    return Dataset.from_dict(tokenized_train).with_format("torch"), Dataset.from_dict(tokenized_test).with_format("torch")


def prepare_llm_dataset(df_train, df_test, task, shots=args.shots):
    messages = [
        {'role': 'system', 'content': 'You are a helpful assistant that will follow every instruction from the user. Provide only a short answer.'},
    ]
    num_classes = len(CLASS_MAPPER[task])

    texts = []
    labels = []
    print(shots)
    print(task)
    if shots > 0:
        print('I am running in here')
        targets = np.array(df_train.label)    
        for cls in range(num_classes):
            inds = np.argwhere(targets == cls).reshape(-1)
            indices = torch.randperm(len(inds))
            inds = inds[indices]
            # inds = inds[:SHOTS]
            inds = inds[:shots]
            print(df_train.text)
            for idx in inds:
                texts.append(df_train.text[int(idx)])
                labels.append(df_train.label[int(idx)])
        
    if task == 'sentiment':
        messages.append({'role': 'user', 'content': 'Determine sentiment of the sentence using following options: 1) negative; 2) positive. Use only these two options.'})
        if 'gemma' in MODEL:
            messages.append({'role': 'assistant', 'content': f'Ok, I will determine the sentiment of the Sentences you will give me using only the options provided!'},)
        for idx in torch.randperm(len(labels)):
            messages.append({'role': 'user', 'content': f'Sentence: {texts[idx]}'})
            messages.append({'role': 'assistant', 'content': CLASS_MAPPER[task][labels[idx]]})
    elif task == 'topic':
        messages.append({'role': 'user', 'content': 'Determine topic of the sentence using following options: 1) science and technology; 2) travel; 3) politics; 4) sports; 5) health; 6) entertainment; 7) geography. Use only these options.'})
        if 'gemma' in MODEL:
            messages.append({'role': 'assistant', 'content': f'Ok, I will determine the topic of the Sentences you will give me using only the options provided!'},)
        for idx in torch.randperm(len(labels)):
            messages.append({'role': 'user', 'content': f'Sentence: {texts[idx]}'})
            messages.append({'role': 'assistant', 'content': CLASS_MAPPER[task][labels[idx]]})
    elif task == 'intent':
        messages.append({'role': 'user', 'content': 'Determine topic of the sentence using following options: 1) alarm query; 2) audio volune down; 3) calendar remove; 4) cooking recipe; 5) datetime convert; 6) send email; 7) play audiobook; 8) movie recommendation; 9) transport ticket; 10) weather query. Use only these options.'})
        if 'gemma' in MODEL:
            messages.append({'role': 'assistant', 'content': f'Ok, I will determine the intent of the Sentences you will give me using only the options provided!'},)
        for idx in torch.randperm(len(labels)):
            messages.append({'role': 'user', 'content': f'Sentence: {texts[idx]}'})
            messages.append({'role': 'assistant', 'content': CLASS_MAPPER[task][labels[idx]]})
    elif task == 'sarcasm':
        messages.append({'role': 'user', 'content': 'Determine comment in the sentence is sarcastic using following options: 1) serious; 2) sarcastic. Use only these two options.'})
        if 'gemma' in MODEL:
            messages.append({'role': 'assistant', 'content': f'Ok, I will determine whether comment in the sentence is sarcastic using only the options provided!'},)
        for idx in torch.randperm(len(labels)):
            messages.append({'role': 'user', 'content': f'Sentence: {texts[idx]}'})
            messages.append({'role': 'assistant', 'content': CLASS_MAPPER[task][labels[idx]]})
        
    return messages


def prepare_instruction_tuning_dataset(df_train, df_test, task, tokenizer):
    messages = [
        {'role': 'system', 'content': 'You are a helpful assistant that will follow every instruction from the user. Provide only a short answer.'},
    ]
    if task == 'sentiment':
        messages.append({'role': 'user', 'content': 'Determine sentiment of the sentence using following options: 1) negative; 2) positive. Use only these two options.'})
        if 'gemma' in MODEL:
            messages.append({'role': 'assistant', 'content': f'Ok, I will determine the sentiment of the Sentences you will give me using only the options provided!'},)
    elif task == 'topic':
        messages.append({'role': 'user', 'content': 'Determine topic of the sentence using following options: 1) science and technology; 2) travel; 3) politics; 4) sports; 5) health; 6) entertainment; 7) geography. Use only these options.'})
        if 'gemma' in MODEL:
            messages.append({'role': 'assistant', 'content': f'Ok, I will determine the topic of the Sentences you will give me using only the options provided!'},)
    elif task == 'intent':
        messages.append({'role': 'user', 'content': 'Determine topic of the sentence using following options: 1) alarm query; 2) audio volune down; 3) calendar remove; 4) cooking recipe; 5) datetime convert; 6) send email; 7) play audiobook; 8) movie recommendation; 9) transport ticket; 10) weather query. Use only these options.'})
        if 'gemma' in MODEL:
            messages.append({'role': 'assistant', 'content': f'Ok, I will determine the intent of the Sentences you will give me using only the options provided!'},)
    elif task == 'sarcasm':
        messages.append({'role': 'user', 'content': 'Determine comment in the sentence is sarcastic using following options: 1) serious; 2) sarcastic. Use only these two options.'})
        if 'gemma' in MODEL:
            messages.append({'role': 'assistant', 'content': f'Ok, I will determine whether comment in the sentence is sarcastic using only the options provided!'},)

    dataset = {'prompt': [], 'completion': []}
    for idx, row in df_train.iterrows():
        temp_messages = copy.deepcopy(messages)
        temp_messages.append({'role': 'user', 'content': f'Sentence: {row["text"]}'})
        dataset['prompt'].append(temp_messages)
        dataset['completion'].append([{'role': 'assistant', 'content': CLASS_MAPPER[task][row['label']]}])
    return dataset
    

def finetuning(train, test, task):
    train_loader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=len(CLASS_MAPPER[task]), classifier_dropout= 0.2).to(device)
    
    optim = torch.optim.AdamW(model.parameters(), lr=LR)
    total_batches = len(train_loader)
    
    for epoch in range(NUM_EPOCHS):
        total_loss = 0

        model.train()
        for batch in train_loader:
            optim.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs[0]
            loss.backward()
            optim.step()

            total_loss += loss.item()
            wandb.log({'loss': loss.item()})
            
        if epoch % 5  == 0 and epoch != 0:
            logging.info("LOSS: " + str(total_loss/len(train_loader)))

    model.eval()
    test_loader = DataLoader(test, batch_size=256, shuffle=False)
    all_preds = []
    all_corrs = []
    
    total_batches = len(test_loader)
    with torch.no_grad():
        total_correct = 0
        for idx, batch in enumerate(test_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

            _, predicted = torch.max(outputs[1], 1)
            all_preds.extend(predicted)
            all_corrs.extend(labels)
            
    return all_preds, all_corrs

def llm_prompting_icl(messages, test, task):
    if 'large' in MODEL:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
        print('Using quantisation')
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map='auto', torch_dtype=torch.float16, oad_in_4bit=True,)
        pipe = pipeline('text-generation', model=model, tokenizer=tokenizer, use_cache=False)
    else:
        pipe = pipeline('text-generation', model=MODEL_NAME, model_kwargs={'torch_dtype': torch.bfloat16}, device=device, use_cache=False)

    all_preds = []
    all_corrs = []
    outputs = []
    prompts = []
    for idx, row in test.iterrows():
        temp_messages = copy.deepcopy(messages)
        temp_messages.append({'role': 'user', 'content': f'Sentence: {row["text"]}'})
        prompts.append(temp_messages)
        all_corrs.append(row['label'])
    
    with torch.no_grad():
        decoded = pipe(prompts, max_new_tokens=20, do_sample=False, num_beams=1, top_p=None, temperature=None, use_cache=False, pad_token_id=pipe.tokenizer.eos_token_id)
    for dec in decoded:
        decoded_text = dec[0]['generated_text'][-1]['content']
        outputs.append(decoded_text)

        found = False
        for idx, cls in enumerate(CLASS_MAPPER[task]):
            if cls in decoded_text.lower() or str(idx) in decoded_text.lower():
                all_preds.append(idx) 
                found = True
                break
        if not found:
            all_preds.append(-1)   
    return all_preds, all_corrs, outputs

def instruction_tuning(df_train, df_test):
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,                     
        lora_alpha=16,             
        lora_dropout=0.05,          
        target_modules=["k_proj", "v_proj", "down_proj"],
    )
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map='auto')
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()


    tuning_dataset = prepare_instruction_tuning_dataset(df_train, df_test, TASK, tokenizer)
    tuning_dataset = Dataset.from_dict(tuning_dataset)
    tuning_dataset = tuning_dataset.map(apply_chat_template, fn_kwargs={"tokenizer": tokenizer})

    training_args = TrainingArguments(
        output_dir=os.path.join('results', TASK, LANGUAGE, f'dataset_{SIZE}'),
        per_device_train_batch_size=BATCH_SIZE,
        learning_rate=3e-5,
        num_train_epochs=10,
        logging_strategy="no",
        save_strategy="no",
        max_steps=600,
        gradient_accumulation_steps=1,
        optim="paged_adamw_8bit",
        lr_scheduler_type="linear",
        warmup_steps=10,        
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=tuning_dataset,
    )

    trainer.train()
    model.eval()

    messages = prepare_llm_dataset(df_train, df_test, TASK)
    pipe = pipeline('text-generation', model=model, tokenizer=tokenizer, use_cache=False)
    test_loader = DataLoader(test, batch_size=1, shuffle=False)
    all_preds = []
    all_corrs = []
    outputs = []
    prompts = []
    for idx, row in test.iterrows():
        temp_messages = copy.deepcopy(messages)
        temp_messages.append({'role': 'user', 'content': f'Sentence: {row["text"]}'})
        prompts.append(temp_messages)
        all_corrs.append(row['label'])
    
    with torch.no_grad():
        decoded = pipe(prompts, max_new_tokens=20, do_sample=False, num_beams=1, top_p=None, temperature=None, use_cache=False, pad_token_id=pipe.tokenizer.eos_token_id)
        # print(decoded)
    for dec in decoded:
        decoded_text = dec[0]['generated_text'][-1]['content']
        # print(decoded_text)
        outputs.append(decoded_text)

        # all_corrs.append(row['label'])
        found = False
        for idx, cls in enumerate(CLASS_MAPPER[task]):
            if cls in decoded_text.lower() or str(idx) in decoded_text.lower():
                all_preds.append(idx) 
                found = True
                break
        if not found:
            all_preds.append(-1)

    del test_loader
    del messages
    del pipe
    
    messages = prepare_llm_dataset(df_train, df_test, TASK, shots=2)
    print(messages)
    pipe = pipeline('text-generation', model=model, tokenizer=tokenizer, use_cache=False)
    test_loader = DataLoader(test, batch_size=1, shuffle=False)
    all_preds_icl = []
    all_corrs_icl = []
    outputs_icl = []
    prompts_icl = []
    for idx, row in test.iterrows():
        temp_messages = copy.deepcopy(messages)
        temp_messages.append({'role': 'user', 'content': f'Sentence: {row["text"]}'})
        prompts_icl.append(temp_messages)
        all_corrs_icl.append(row['label'])
    
    with torch.no_grad():
        decoded = pipe(prompts, max_new_tokens=20, do_sample=False, num_beams=1, top_p=None, temperature=None, use_cache=False, pad_token_id=pipe.tokenizer.eos_token_id)
    for dec in decoded:
        decoded_text = dec[0]['generated_text'][-1]['content']
        outputs_icl.append(decoded_text)

        found = False
        for idx, cls in enumerate(CLASS_MAPPER[task]):
            if cls in decoded_text.lower() or str(idx) in decoded_text.lower():
                all_preds_icl.append(idx) 
                found = True
                break
        if not found:
            all_preds_icl.append(-1)
    
    del trainer
    del pipe
    del model
    del test_loader
    del messages
    torch.cuda.empty_cache()
    
    return all_preds, all_corrs, outputs, all_preds_icl, all_corrs_icl, outputs_icl


def evaluate_results_ft(predictions, golden):
    preds = [x.item() for x in predictions]
    corrs = [x.item() for x in golden]
    f1 = f1_score(corrs, preds, average='macro')
    acc = accuracy_score(corrs, preds)
    return f1, acc, preds, corrs


def evaluate_results_llm(predictions, golden):
    f1 = f1_score(golden, predictions, average='macro')
    acc = accuracy_score(golden, predictions)
    return f1, acc, predictions, golden

# wandb.init(
#     project=f'synth_size_change',
#     name=f'{EXPERIMENT_TYPE}-{TASK}-{MODEL}-{LANGUAGE}-{SIZE}-human_{HUMAN}',
#     config={'shots': SHOTS}
# )

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = not torch.backends.cudnn.deterministic

if TASK == 'sentiment':
    SIZES = [10, 20, 30, 40, 50, 75, 100, 125, 150, 175, 200, 250, 300, 350, 400]
        
elif TASK == 'topic':
    SIZES = [10, 20, 30, 40, 50, 75, 100, 125, 150, 175, 200, 250, 300, 350, 400, 500, 600, 800, 1000, 1200, 1400]

elif TASK == 'intent':
    SIZES = [10, 20, 30, 40, 50, 75, 100, 125, 150, 175, 200, 250, 300, 350, 400, 500, 600, 800, 1000, 1250, 1500, 1750, 2000]

elif TASK == 'sarcasm':
    SIZES = [10, 20, 30, 40, 50, 75, 100, 125, 150, 175, 200]


for SIZE in SIZES:
    results_path = os.path.join('results', TASK, LANGUAGE, f'dataset_{SIZE}')
    if EXPERIMENT_TYPE == 'instruction_tuning' and os.path.exists(os.path.join(results_path, f'partial-new-{MODEL}-{EXPERIMENT_TYPE}-{NUM_EPOCHS}-w_icl.json')):
        with open(os.path.join(results_path, f'partial-new-{MODEL}-{EXPERIMENT_TYPE}-{NUM_EPOCHS}-w_icl.json'), 'r') as file:
            it_results = json.load(file)
    else:
        it_results = {'f1': [], 'acc': [], 'predictions': [], 'golden': [], 'outputs': [], 'f1_icl': [], 'acc_icl': [], 'predictions_icl': [], 'golden_icl': [], 'outputs_icl': []}
    print(os.path.join(results_path, f'full-{MODEL}-{EXPERIMENT_TYPE}-{NUM_EPOCHS}-dynamic_batch_size.json'))
    if (EXPERIMENT_TYPE == 'finetuning' and os.path.exists(os.path.join(results_path, f'full-{MODEL}-{EXPERIMENT_TYPE}-{NUM_EPOCHS}-dynamic_batch_size.json'))) or (EXPERIMENT_TYPE == 'llm' and os.path.exists(os.path.join(results_path, f'{MODEL}-{EXPERIMENT_TYPE}-{SHOTS}.json'))) or (EXPERIMENT_TYPE == 'instruction_tuning' and os.path.exists(os.path.join(results_path, f'full-{MODEL}-{EXPERIMENT_TYPE}-{NUM_EPOCHS}-dynamic_batch_size-w_icl.json'))):
        print(f'Skipping because the setting already exists -- {MODEL}, {EXPERIMENT_TYPE}, {NUM_EPOCHS}, {SHOTS}, {SIZE}')
        continue
    if SIZE > 10 and SHOTS == 0 and EXPERIMENT_TYPE == 'llm':
            break
    print(f'Running size: {SIZE}')
    
    if EXPERIMENT_TYPE == 'instruction_tuning':
        BATCH_SIZE = 4
    else:
        if SIZE <= 20:
            BATCH_SIZE = 4
        elif SIZE <= 40:
            BATCH_SIZE = 6
        elif SIZE <= 100:
            BATCH_SIZE = 8
        elif SIZE <= 150:
            BATCH_SIZE = 12
        elif SIZE <= 250:
            BATCH_SIZE = 16
        elif SIZE <= 600:
            BATCH_SIZE = 32
        else:
            BATCH_SIZE = 64
    
    print(f'Running with batch sie of {BATCH_SIZE}')
    SEED = args.seed
    REPEATS = args.repeat
    random.seed(SEED)
    rep_eat_seeds = [random.randint(1, 100000) for _ in range(REPEATS)]
    print(f'Repeats {REPEATS}')
    accs = []
    f1s = []
    predictions_s = []
    golden_s = []
    outputs_s = []
    for rep_eat, seed in zip(range(REPEATS), rep_eat_seeds):
        if EXPERIMENT_TYPE == 'instruction_tuning' and rep_eat < len(it_results['f1']):
            continue
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        print(f'Running repeat {rep_eat}')
        
                
        if not os.path.exists(results_path):
            os.makedirs(results_path)
    
        train, test = load_data(SIZE)
    
        if EXPERIMENT_TYPE == 'finetuning':
            train, test = prepare_ft_dataset(train, test)
            predictions, golden =  finetuning(train, test, TASK)
            f1, acc, predictions, golden = evaluate_results_ft(predictions, golden)
            # wandb.log({'f1': f1, 'acc': acc, 'repeat': rep_eat, 'size': SIZE, 'epochs': NUM_EPOCHS, 'batch_size': BATCH_SIZE})
        elif EXPERIMENT_TYPE == 'llm':
            print(f'Using {SHOTS} shots')
            messages = prepare_llm_dataset(train, test, TASK)
            predictions, golden, outputs =  llm_prompting_icl(messages, test, TASK)
            f1, acc, predictions, golden = evaluate_results_llm(predictions, golden)
            # wandb.log({'f1': f1, 'acc': acc, 'repeat': rep_eat, 'size': SIZE, 'shots': SHOTS})
            outputs_s.append(outputs)
        elif EXPERIMENT_TYPE == 'instruction_tuning':
            print('Running instruction tuning!')
            predictions, golden, outputs, predictions_icl, golden_icl, outputs_icl =  instruction_tuning(train, test)
            f1, acc, predictions, golden = evaluate_results_llm(predictions, golden)
            f1_icl, acc_icl, predictions_icl, golden_icl = evaluate_results_llm(predictions_icl, golden_icl)
            # wandb.log({'f1': f1, 'acc': acc, 'repeat': rep_eat, 'size': SIZE, 'shots': SHOTS, 'epochs': NUM_EPOCHS, 'batch_size': BATCH_SIZE, 'f1_icl': f1_icl, 'acc_icl': acc_icl})
            it_results['f1'].append(f1)
            it_results['acc'].append(acc)
            it_results['predictions'].append(predictions)
            it_results['golden'].append(golden)
            it_results['outputs'].append(outputs)

            it_results['f1_icl'].append(f1_icl)
            it_results['acc_icl'].append(acc_icl)
            it_results['predictions_icl'].append(predictions_icl)
            it_results['golden_icl'].append(golden_icl)
            it_results['outputs_icl'].append(outputs_icl)
            outputs_s.append(outputs)
            with open(os.path.join(results_path, f'partial-new-{MODEL}-{EXPERIMENT_TYPE}-{NUM_EPOCHS}-w_icl.json'), 'w') as file:
                json.dump(it_results, file)
            
        f1s.append(f1)
        accs.append(acc)
        predictions_s.append(predictions)
        golden_s.append(golden)
        torch.cuda.empty_cache()
    if EXPERIMENT_TYPE == 'finetuning':
        with open(os.path.join(results_path, f'full-{MODEL}-{EXPERIMENT_TYPE}-{NUM_EPOCHS}-dynamic_batch_size.json'), 'w') as file:
            json.dump({'f1': f1s, 'acc': accs, 'predictions': predictions_s, 'golden': golden_s}, file)
    elif EXPERIMENT_TYPE == 'llm':
        with open(os.path.join(results_path, f'{MODEL}-{EXPERIMENT_TYPE}-{SHOTS}.json'), 'w') as file:
            json.dump({'f1': f1s, 'acc': accs, 'predictions': predictions_s, 'golden': golden_s, 'outputs': outputs_s}, file)
    elif EXPERIMENT_TYPE == 'instruction_tuning':
        with open(os.path.join(results_path, f'full-{MODEL}-{EXPERIMENT_TYPE}-{NUM_EPOCHS}-dynamic_batch_size-w_icl.json'), 'w') as file:
            json.dump(it_results, file)