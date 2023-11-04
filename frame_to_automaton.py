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

def main(args):
    props = args.propositions_seperate_by_comma.split(',')
    BOX_TRESHOLD = args.box_threshold
    scale=args.scale
    second_per_frame=args.second_per_frame
    states, transitions, accept_states = frame2automaton(args.video_path, props, scale, second_per_frame)
    print(states, transitions, accept_states)

def get_prop_traj(prop, video, annot=True):
    classes = [prop]
    traj = []
    for v in video:
        d = detect(v, classes)
        if len(d.class_id) > 0:
            traj.append(np.round(np.max(d.confidence), 2))
        else:
            traj.append(0)
        if annot:
            annotate(v, d, classes)
    return traj

def sigmoid(x, k=1, x0=0):
  return 1 / (1 + np.exp(-k * (x-x0) ))

def mapping(conf_arr, true_threshold=0.66, false_threshold=0.38):
    probs = []
    for conf in conf_arr:
        if conf >= true_threshold:
            probs.append(1)
        elif conf <= false_threshold:
            probs.append(0)
        else:
            probs.append(round(sigmoid(conf, k=50, x0=0.56), 2))
    return probs

def get_probabilities(props, video):
    confs = []
    for p in props:
        confs.append(get_prop_traj(p, video))
    print(confs)
    probabilities = []
    for c in confs:
        probabilities.append(mapping(c))
    return probabilities

def frame2automaton(video_path, props, scale=2, second_per_frame=2):
    video = read_video(video_path, scale=scale, second_per_frame=second_per_frame)
    num_frames = len(video)
    probabilities = get_probabilities(props, video)
    states, transitions, accept_states = build_automaton(props, num_frames, probabilities)
    return states, transitions, accept_states

if __name__ == '__main__':
    main(parser.parse_args())
