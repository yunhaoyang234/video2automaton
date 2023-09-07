# video2automaton

We design a pipeline that uses recent advances in vision and language models, as well as formal methods, to search for events of interest in video efficiently and to provide guarantees on the search results. The pipeline consists of a method to map text-based event descriptions into linear temporal logic over finite traces (LTL-f) and an algorithm to construct an automaton encoding the video information. 

![pipeline](https://github.com/yunhaoyang234/video2automaton/blob/main/examples/pipeline.png)

## Setup
### For the updated package installation, please visit [here](https://github.com/UTAustin-SwarmLab/Neuro-Symbolic-Video-Frame-Search)

Python version >= 3.7.0\
For automaton construction, run\
`python setup.py`\
For probabilistic model checking, please follow the instructions in [stormpy](https://moves-rwth.github.io/stormpy/installation.html)

## Datasets
- `HMDB: a large human motion database` [download](https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/) with privacy annotations [download](https://htwang14.github.io/PA-HMDB51-website/index.html)
- `nuImages` with object annotations [download](https://www.nuscenes.org/nuimages)
- `Driving Control Dataset`

## Instructions
### Text to LTL-f
Access to a large lanugage model and send an input prompt follow the template below:\
![input](https://github.com/yunhaoyang234/video2automaton/blob/main/examples/prompt.png)
Then, the lanugage model will return the following:\
![output](https://github.com/yunhaoyang234/video2automaton/blob/main/examples/response.png)

Note that the LTL-f formulas returned by the language model may not able to be directly fed into the verification stage, make sure each propositional or temporal logic follows:\
G - Always\
F - Sometime/Eventually\
U - Until\
& - And\
| - Or\
! - Not\
[more rules](https://www.stormchecker.org/documentation/background/properties.html#propositional-expressions)

### Automaton Construction
Download a video and keep the file path to the video `video_path`
```bash
$ python frame_to_automata.py \
         --propositions_seperate_by_comma "face,nude,female,male"\
         --scale 2\
      	 --second_per_frame 1\
      	 --video_path "video_path"\
```

An illustration is presented in [Jupyter-Notebook](https://github.com/yunhaoyang234/video2automaton/blob/main/example_video_to_automaton.ipynb)

### Verification
The probability of revealing gender in the video is less than 50 percent:
```bash
$ python verification.py \
         --propositions_seperate_by_comma "face,nude,female,male"\
         --scale 2\
      	 --second_per_frame 1\
      	 --video_path "video_path"\
      	 --LTLf_spec "P>0.5 [ initial U (G  !male & !female) ]"\
```
