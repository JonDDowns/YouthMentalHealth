import pandas as pd
import numpy as np
import torch
from transformers import DebertaV2Tokenizer, AutoModelForSequenceClassification
from sklearn.metrics import f1_score

# Load model and tokenizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')
tokenizer = DebertaV2Tokenizer.from_pretrained('./data/tokenizer', device_map=device)
model = AutoModelForSequenceClassification.from_pretrained(
    './models/DEB_LARGE_LR4e-5_EP20', 
    num_labels=39, 
    ignore_mismatched_sizes=True, 
    device_map=device
)
model.eval()

def get_logits(features):
    (len_x, _) = features.shape

    # Make inputs one large string, then tokenize
    x = features['NarrativeLE'] + '||' + features['NarrativeCME']
    tokens = tokenizer(x.to_list(), padding=True, return_tensors='pt')
    tokens = tokens.to(model.device)

    # Get model scores for the batch
    logits = model(**tokens).logits
    return logits

# Get data
x_raw = pd.read_csv('./data/train_features_X4juyT6.csv', index_col='uid')
y_raw = pd.read_csv('./data/train_labels_JxtENGl.csv', index_col='uid')

all_preds = []
num_per_batch = 10
num_batches = np.ceil(x_raw.shape[0] / num_per_batch)
for batch in np.array_split(x_raw, num_batches):
    with torch.no_grad():
        logits = get_logits(batch)
    all_preds.append(logits)

all_preds = torch.vstack(all_preds)
torch.save(all_preds, 'logits.pt')

