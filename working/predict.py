import pandas as pd
import numpy as np
import torch
from transformers import DebertaV2Tokenizer, AutoModelForSequenceClassification
from sklearn.metrics import f1_score

# Load model and tokenizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BINARY_THRESH = 0.04

# Load data and predictions
x_raw = pd.read_csv('./data/train_features_X4juyT6.csv', index_col='uid')
y_raw = pd.read_csv('./data/train_labels_JxtENGl.csv', index_col='uid')
logits = torch.load('./logits.pt')

def average_f1(predictions: pd.DataFrame, labels: pd.DataFrame):
    """Score a set of predictions using the competition metric. F1 is averaged
    across all target variables. For categorical variables, micro-averaged
    F1 score is used.

    Args:
        predictions (pd.DataFrame): Dataframe of predictions, with one column
            for each target variable. The index should be the uid.
        labels (pd.DataFrame): Dataframe of ground truth values, with one column
            for each target variable. The index should be the uid.
    """
    # Check that there are 23 target variables
    assert predictions.shape[1] == 23

    # Check that column order and row order are the same
    assert (predictions.columns == labels.columns).all()
    assert (predictions.index == labels.index).all()

    # All values should be integers
    assert (predictions.dtypes == int).all()

    CATEGORICAL_VARS = ["InjuryLocationType", "WeaponType1"]
    BINARY_VARS = np.setdiff1d(labels.columns, CATEGORICAL_VARS)

    # Calculate F1 score averaged across binary variables
    binary_f1 = f1_score(
        labels[BINARY_VARS],
        predictions[BINARY_VARS],
        average="macro",
    )
    f1s = [binary_f1]

    # Calculate F1 score for each categorical variable
    for cat_col in CATEGORICAL_VARS:
        f1s.append(f1_score(labels[cat_col], predictions[cat_col], average="micro"))

    return np.average(f1s, weights=[len(BINARY_VARS), 1, 1])

# Normalize logit predictions by row using softmax
logits = logits.cpu()
probs = torch.nn.functional.softmax(logits, dim=1)

# Get model scores for the batch
probs = probs.cpu()

# Binary variable predictions
binary_probs = probs[:, :21]
binary_preds = (binary_probs >= BINARY_THRESH).long().detach().numpy()

# Injury location predictions
injury_loc_probs = logits[:, 21:27]
_, injury_loc_inds = torch.max(injury_loc_probs, dim=1)
injury_loc_inds = (injury_loc_inds + 1).detach().numpy()

# Weapon type predictions
weapon_type_probs = logits[:, 27:]
_, weapon_type_inds = torch.max(weapon_type_probs, dim=1)
weapon_type_inds = (weapon_type_inds + 1).detach().numpy()

# Prepare output array
out = np.zeros((4000, 23))
out[:, :21] = binary_preds
out[:, 21] = injury_loc_inds
out[:, 22] = weapon_type_inds
out = out.astype(int)

# Convert to DF, use index/columns from the training inputs/outputs
out = pd.DataFrame(out)
out.columns = y_raw.columns
out.index = x_raw.index

# Write to file for Steven to review
out.to_csv('for_steven.csv')

# Print F1 scores
print(average_f1(out, y_raw))
