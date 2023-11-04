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
parser.add_argument('--LTLf_spec', type=str, default='') # example: P=? [F "face"]

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
    components.exit_rates = [0.0 for i in range(len(states))]
    ma = stormpy.storage.SparseMA(components)
    print(ma)
    print(model_checking(ma, spec))


def model_checking(model: stormpy.storage.SparseMA, formula_str: str) -> any:
    """Model checking.

    Args:
        model (stormpy.storage.SparseMA): Markov Automata.
        formula_str (str): Formula string.

    Returns:
    any: Result.
    """
    # Initialize Prism Program
    path = stormpy.examples.files.prism_dtmc_die  #  prism_mdp_maze
    prism_program = stormpy.parse_prism_program(path)

    # Define Properties
    properties = stormpy.parse_properties(formula_str, prism_program)

    # Get Result and Filter it
    result = stormpy.model_checking(model, properties[0])
    filter = stormpy.create_filter_initial_states_sparse(model)
    result.filter(filter)
    return result


def build_trans_matrix(transitions: list[tuple[int, int, float]], states: list[State]):
    """Build transition matrix.

    Args:
        transitions (list[tuple[int, int, float]]): List of transitions.
        states (list[State]): List of states.
    """
    matrix = np.zeros((len(states), len(states)))
    for t in transitions:
        matrix[int(t[0]), int(t[1])] = float(t[2])
    trans_matrix = stormpy.build_sparse_matrix(matrix, list(range(len(states))))
    return trans_matrix


def build_label_func(states: list[State], props: list[str]) -> stormpy.storage.StateLabeling:
    """Build label function.

    Args:
        states (list[State]): List of states.
        props (list[str]): List of propositions.


    Returns:
        stormpy.storage.StateLabeling: State labeling.
    """
    state_labeling = stormpy.storage.StateLabeling(len(states))

    for label in props:
        state_labeling.add_label(label)

    for state in states:
        if state.state_index == 0:
            state_labeling.add_label("init")
            state_labeling.add_label_to_state("init", state.state_index)
        else:
            for label in state.current_descriptive_label:
                state_labeling.add_label_to_state(label, state.state_index)

    return state_labeling
    
if __name__ == '__main__':
    main(parser.parse_args())
