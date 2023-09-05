import stormpy
import numpy
import stormpy.examples
import stormpy.examples.files
from stormpy.storage import Expression

from video_to_frame import *
from automaton_state import *
import argparse
import pprint
import json

parser = argparse.ArgumentParser()
parser.add_argument('--propositions_seperate_by_comma', type=str, default='human,car')
parser.add_argument('--box_threshold', type=float, default=0.4)
parser.add_argument('--scale', type=int, default=1)
parser.add_argument('--second_per_frame', type=str, default=1)
parser.add_argument('--video_path', type=str, default='')
parser.add_argument('--LTLf_spec', type=str, default='') # example: "P=? [initial U (F face)]"

def main(args):
    props = args.propositions_seperate_by_comma.split(',')
    BOX_TRESHOLD = args.box_threshold
    scale=args.scale
    second_per_frame=args.second_per_frame
    states, transitions, accept_states = frame2automaton(args.video_path, props, scale, second_per_frame)
    
    spec = args.LTLf_spec
    transition_matrix = build_trans_matrix(transitions, len(states))
    state_labeling = build_label_func(states, props)
    markovian_states = stormpy.BitVector(len(states), list(range(len(states))))
    components = stormpy.SparseModelComponents(transition_matrix=transition_matrix, state_labeling=state_labeling,
                                               markovian_states=markovian_states)
    ma = stormpy.storage.SparseMA(components)
    print(ma)
    print(model_checking(ma, spec))


def build_trans_matrix(transitions, num_of_states):
    matrix = np.zeros((num_of_states,num_of_states))
    for t in transitions:
        matrix[t[0], t[1]] = t[2]
    trans_matrix = stormpy.build_sparse_matrix(matrix, list(range(len(states))))
    return trans_matrix

def build_label_func(states, props):
    state_labeling = stormpy.storage.StateLabeling(len(states))
    state_labeling.add_label('initial')
    for label in props:
        state_labeling.add_label(label)
    for state in states:
        for label in state.label_list:
            state_labeling.add_label_to_state(label, state.state_idx)
    return state_labeling

def model_checking(model, spec):
    path = stormpy.examples.files.prism_dtmc_die
    prism_program = stormpy.parse_prism_program(path)
    properties = stormpy.parse_properties(formula_str, prism_program)
    result = stormpy.model_checking(model, properties[0])
    filter = stormpy.create_filter_initial_states_symbolic(model)
    result.filter(filter)
    return result

