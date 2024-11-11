import pandas as pd
import torch
from transformers import DebertaV2Tokenizer, DebertaV2Model, TrainingArguments, Trainer, DataCollatorWithPadding, AutoModelForSequenceClassification, BitsAndBytesConfig, AutoTokenizer
import datasets
import fns

RUN_NAME = 'DEB_LARGE_LR4e-5_EP20'
MODEL_NAME = 'microsoft/deberta-v3-large'

# Load model and tokenizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=39,
    ignore_mismatched_sizes=True,
)
model.gradient_checkpointing_enable()

# Get data
data_dir = './data/'
x_raw = pd.read_csv('./data/train_features_X4juyT6.csv', index_col='uid')
y_raw = pd.read_csv('./data/train_labels_JxtENGl.csv', index_col='uid')

# Use a collator to save space during training
collator = DataCollatorWithPadding(tokenizer)

# Make the inputs a single string with the LE and CME narratives
x = x_raw['NarrativeLE'] + '||' + x_raw['NarrativeCME']

# Run one-hot encoding function on labels
y = fns.one_hot_labels(y_raw)

# Convert data to a list of dictionaries
rowlist = []
for index, row in y.iterrows():
    d = {}
    d['labels'] = [x for x in row]
    d['text'] = x[index]
    rowlist.append(d)

# Convert to transformers dataset, go ahead and tokenize
ds = datasets.Dataset.from_list(rowlist)
def tokenize_function(dat):
    return tokenizer(dat['text'], padding=True, truncation=True, return_tensors='pt')
tokenized_ds = ds.map(tokenize_function, batched=True)

# Set up training arguments
train_args = TrainingArguments(
    output_dir='./models',
    eval_strategy='no',
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    num_train_epochs=20,
    learning_rate=4e-5,
    report_to='tensorboard',
    run_name = RUN_NAME,
    logging_steps=10,
    warmup_steps=250
)

# Create trainer, initialize
trainer = Trainer(
    model=model,
    args=train_args,
    train_dataset=tokenized_ds,
    data_collator=collator,
)
trainer.train()
model.save_pretrained(f'./models/{RUN_NAME}')

