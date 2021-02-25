'''
    To install dependencies, run both of the following commands, in sequence:

    pip install -r requirements.txt
    python -m spacy download en_core_web_lg
'''

import csv
import re
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import word_tokenize
import numpy as np
import pandas as pd
import random
from sklearn import metrics
import spacy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

nltk.download('punkt')

class NeuralNet(nn.Module):
    def __init__(self, input_dims):
        super(NeuralNet, self).__init__()
        self.nlp = spacy.load("en_core_web_lg")
        self.layer1 = nn.Linear(input_dims, 10)
        self.layer2 = nn.Linear(10, 3)

    def forward(self, x):
        x = self.process_strings(x)
        x = self.layer1(x)
        x = nn.LeakyReLU()(x)
        x = self.layer2(x)
        x = nn.Softmax(dim=3)(x)
        return x[0][0]
    
    def process_strings(self, x):
        return torch.FloatTensor([[list(map(
            lambda string: self.nlp(str(string)).vector, # extra typecast to str to avoid np.str_ problems
            x
        ))]])


def main():
    # read training set
    train = pd.read_csv('train.csv')
    X_train = train['Text']
    y_train = train['Verdict']
    train_set = list(zip(
        X_train, 
        map(
            lambda y: y+1,
            y_train
        )
    ))
    random.shuffle(train_set)
    print('Loaded trainset')

    # find proportions of examples with each label (will be used for weighting cross-entropy loss)
    train_set_label_counts = [0, 0, 0]
    for datum in train_set:
        train_set_label_counts[datum[1]] += 1
    class_weights = np.array([10000, 10000, 10000]) / train_set_label_counts
    class_weights = torch.FloatTensor(class_weights)

    # create and train model
    neural_net = NeuralNet(300)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(neural_net.parameters(), lr=0.001, weight_decay=0.00001)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=4, shuffle=True, num_workers=4)
    print('Created model')

    for epoch in range(3):
        print(f'Epoch {epoch}')
        for i, data in enumerate(train_loader, 0):
            x, labels = data
            optimizer.zero_grad()
            outputs = neural_net(x)
            loss = criterion(outputs, torch.LongTensor(labels))
            loss.backward()
            optimizer.step()

    print('Done training')

    # read test set and generate predictions
    test_set = pd.read_csv('test.csv')
    test_x = test_set['Text']
    test_outputs = neural_net(test_x)
    test_output_labels = list(map(
        lambda distrib: np.argmax(distrib.detach().numpy()) - 1,
        test_outputs
    ))
    test_set['Verdict'] = pd.Series(test_output_labels)
    test_set.drop(columns=['Text'], inplace=True)
    test_set.to_csv('A0184415E.csv', index=False)
    print('Done.')


# Allow the main class to be invoked if run as a file.
if __name__ == "__main__":
    main()
