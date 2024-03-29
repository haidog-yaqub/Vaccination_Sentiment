import torch
from torch import nn
from sklearn.metrics import confusion_matrix
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd


def cm_matrix(y_true, y_pred, cls):
    labels = [i for i in range(cls)]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm = np.array(cm, dtype=int).T
    return cm


def Evaluate(model, test_loader, device, loss_func, epoch, path, cls):
    model.eval()
    step = 0
    report_loss = 0
    predicts = []
    labels = []

    with torch.no_grad():
        for text, mask, label in tqdm(test_loader):
            text = text.to(device)
            mask = mask.to(device)
            label = label.to(device)

            pred = model(text, mask)
            loss = loss_func(pred, label)
            report_loss += loss.item()
            step += 1

            prediction = torch.argmax(F.softmax(pred, dim=1), dim=1)

            predicts += prediction.to('cpu').tolist()
            labels += label.to('cpu').tolist()

        print('Val Loss: {:.6f}'.format(report_loss / step))

    pd.DataFrame(np.array([predicts, labels]).T,
                 columns=['pred', 'label']).to_csv(path+'_'+str(epoch)+'.csv')
    model.train()

    return report_loss / step, cm_matrix(np.array(predicts), np.array(labels), cls)
