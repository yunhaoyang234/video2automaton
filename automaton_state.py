import numpy as np
import random
import torch

class State:
    def __init__(self, idx, frame_num, label, props):
        # label is a string with characters T or F indicating True and False
        # e.g., 'TTF' or 'FTF'
        self.state_idx = idx
        self.frame_num = frame_num
        self.label = label
        self.label_list = self.build_labels(label, props)
        self.prob = 1

    def __repr__(self):
        return str(self.state_idx) + ' ' + str(self.label_list)

    def __str__(self):
        return self.__repr__()+' '+str(self.frame_num)+' '+ str(self.prob)

    def build_labels(self, label, props):
        labels = []
        for i in range(len(props)):
            if label[i] == 'T':
                labels.append(props[i])
        return labels

    def compute_prob(self, probabilities):
        # probabilities: 2d array [num propositions * num frames]
        probability = 1
        for i in range(len(self.label)):
            if self.label[i] == 'T':
                probability *= probabilities[i][self.frame_num]
            else:
                probability *= 1 - probabilities[i][self.frame_num]
        self.prob = round(probability,2)
