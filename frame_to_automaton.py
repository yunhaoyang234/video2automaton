from video_to_frame import *
from automaton_state import *

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

scale=1
second_per_frame=2
props = ['head']

# change the path to your own file directory
frame2automaton(cwd + 'data/harvard.mp4', props, scale, second_per_frame)

frame2automaton(cwd + 'data/stanford.mp4', props, scale, second_per_frame)

frame2automaton(cwd + 'data/mit.mp4', props, scale, second_per_frame)