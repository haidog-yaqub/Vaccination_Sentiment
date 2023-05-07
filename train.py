from model.bert import BERT
from dataset.tweets import Tweets
from utils.utils import Evaluate

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import torchvision.transforms as transforms
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cls = 2
batch_size = 4
lr_rate = 1e-5
weight_decay = 1e-5
epochs = 12
report_step = 10
languages = ['en', 'zh-CN', 'de']
# languages = ['en']


if __name__ == "__main__":
    model = BERT(cls=cls)
    model = model.to(device)

    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_rate, weight_decay=weight_decay)

    val_data = Tweets(df='data/tweets.csv', label='sentiment', subset='val')
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    step = 0
    report_loss = 0.0
    evaluations = []

    model.train()

    for epoch in range(epochs):
        print("\nEpoch is " + str(epoch + 1))
        # train
        train_data = Tweets(df='data/tweets.csv', label='sentiment', subset='train',
                            trans_augment=[languages[epoch % len(languages)]])
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        for i, (text, mask, label) in enumerate(train_loader):
            optimizer.zero_grad()
            text = text.to(device)
            mask = mask.to(device)
            label = label.to(device)

            pred = model(text, mask)
            loss = loss_func(pred, label)
            report_loss += loss.item()
            loss.backward()
            optimizer.step()

            step += 1
            if (i + 1) % report_step == 0:
                print('Epoch: [{}][{}]    Batch: [{}][{}]    Loss: {:.6f}'.format(
                    epoch + 1, epochs, i + 1, len(train_loader), report_loss / report_step))
                # writer.add_scalar('TrainLoss', report_loss / report_step, step)
                report_loss = 0.0

        # evaluation
        eval_loss, cm = Evaluate(model, val_loader, device, loss_func, epoch, 'val/', cls=cls)
        evaluations.append(eval_loss)
        print('Acc: {:.2f} %'.format(np.trace(cm) / np.sum(cm) * 100))
        if cls == 2:
            precision = cm[1, 1] / (cm[1, 1]+cm[0, 1])
            recall = cm[1, 1] / (cm[1, 1]+cm[1, 0])
            print('F1-Score: {:.2f}'.format((2 * precision * recall) / (precision + recall)))
        for c in range(cls):
            print(cm[c, :])
            count = sum(cm[c, :])
            print(cm[c, :]/count)

        # save model
        torch.save(model.state_dict(), 'save_sentiment/'+str(epoch)+'.pt')

