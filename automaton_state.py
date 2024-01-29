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
        if label == 'initial' or label == 'final':
            return [label]
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

def create_labels(num_props):
    label_list = []
    def add_labels(num_props, label, label_list):
        if len(label) == num_props:
            label_list.append(label)
            return
        add_labels(num_props, label + 'T', label_list)
        add_labels(num_props, label + 'F', label_list)

    add_labels(num_props, '', label_list)
    return label_list

def build_automaton(props, num_frames, probabilities):
    states = []
    labels = create_labels(len(props))

    init_state = State(0, -1, 'initial', props)
    states.append(init_state)
    idx = 1
    for f in range(num_frames):
        for lab in labels:
            state = State(0, f, lab, props)
            state.compute_prob(probabilities)
            if state.prob > 0:
                state.state_idx = idx
                idx += 1
                states.append(state)

    transitions = []
    prev_states = [init_state]

    for i in range(num_frames):
        cur_states = []
        for s in states:
            if s.frame_num == i:
                cur_states.append(s)
        for cs in cur_states:
            for ps in prev_states:
                transitions.append((ps.state_idx, cs.state_idx, cs.prob))
        prev_states = cur_states.copy()

    final_state = State(idx, num_frames, 'final', props)
    states.append(final_state)
    for ps in prev_states:
        transitions.append((ps.state_idx, idx, 1))
    transitions.append((idx,idx,1))

    return states, transitions
    