#!/usr/bin/env python

import torch.nn as nn


class CharCNN(nn.Module):
    def __init__(self, features_in, features_out, sequence_length, p_dropout=0.0):
        """
        Character-level Convolutional Neural Network


        Based on architecture described in:

            Xiang Zhang, Junbo Zhao, Yann LeCun. Character-level Convolutional
            Networks for Text Classification. Advances in Neural Information
            Processing Systems 28 (NIPS 2015)
        """
        super().__init__()
        self.conv_layers = nn.ModuleList()
        self.lin_layers = nn.ModuleList()
        #sequence_length_out = 8

        self.conv_layers.append(
            nn.Sequential(nn.Conv1d(features_in, 32, kernel_size=7),
                          nn.ReLU(),
                          nn.MaxPool1d(kernel_size=3, stride=3)))

        #self.conv_layers.append(
        #    nn.Sequential(nn.Conv1d(256, 256, kernel_size=7),
        #                  nn.ReLU(),
        #                  nn.MaxPool1d(kernel_size=3, stride=3)))

        #for _ in range(3):
        #    self.conv_layers.append(
        #        nn.Sequential(nn.Conv1d(256, 256, kernel_size=3),
        #                      nn.ReLU()))

        #conv_to_lin_dim = 32 * sequence_length_out
        #assert conv_to_lin_dim >= 256

        self.conv_layers.append(
            nn.Sequential(nn.Conv1d(32, 32, kernel_size=3),
                          nn.ReLU(),
                          nn.AdaptiveMaxPool1d(8)))

        self.lin_layers.append(
            nn.Sequential(nn.Linear(32*8, 128),
                          nn.ReLU(),
                          nn.Dropout(p=p_dropout)))

        #self.lin_layers.append(
        #    nn.Sequential(nn.Linear(1024, 1024),
        #                  nn.ReLU(),
        #                  nn.Dropout(p=p_dropout)))

        self.lin_layers.append(nn.Linear(128, features_out))

    def forward(self, X):
        for layer in self.conv_layers:
            print(X.size())
            print(layer)
            X = layer(X)

        print(X.size())
        X = X.view(X.size(0), -1)
        for layer in self.lin_layers:
            print(X.size())
            print(layer)
            X = layer(X)

        return X

    def init(self):
        # initialize weights from a uniform distribution following 
        # "Understanding the difficulty of training deep feedforward 
        #  neural networks" - Glorot, X. & Bengio, Y. (2010)
        for param in self.parameters():
            nn.init.xavier_uniform_(param)
