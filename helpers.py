# adapted from Seaborn version @ https://www.kaggle.com/agungor2/various-confusion-matrix-plots

import matplotlib.pyplot as plt
plt.style.use('ggplot')

from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sns
import numpy as np


def plot_cm(y_true, y_pred, figsize=(10, 8), colour_by_perc=False):
    cm = confusion_matrix(y_true, y_pred, labels=np.unique(y_true))
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%\n%d' % (p, c)
    if colour_by_perc:
        cm = pd.DataFrame(cm_perc, index=np.unique(y_true), columns=np.unique(y_true))
    else:
        cm = pd.DataFrame(cm, index=np.unique(y_true), columns=np.unique(y_true))
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, cmap='plasma', annot=annot, fmt='', ax=ax, square=True)
    plt.xticks(rotation=45)
    plt.title('Confusion Matrix')
    