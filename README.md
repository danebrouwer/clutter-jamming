## Clutter Jamming
---
### Installation

1. Create a conda environment 
```
conda create --name clutter_env python=3.8
```
2. Activate conda environment 
```
conda activate clutter_env 
```
3. Clone the github repo and cd into the repo
```
git clone https://github.com/danebrouwer/clutter-jamming.git
cd clutter-jamming
```
4. Install the dependencies
```
pip install -r requirements.txt
```
5. Install the repo (run this command in the top level clutter-jamming directory)
```
pip install -e .
```

### Run
Let's walk through the two experiments that can be run. Before running either experiment, enter
```
cd clutter-jamming
```
#### Experiment 1
This experiment tests a given set of control strategies for a given number of pseudo-randomly generated scenes. You can run the experiment like this:
```
python experiment1.py
```
If you want to log information about the trials, use the --log argument when executing the script. If you would like to see the trials, use the --gui argument. These two arguments can be used at the same time.

#### Experiment 2
This experiment runs a test on the two primitive control strategies for a varying set of parameters (the straight line control strategy is also run as a comparison) . These parameters are easily changed in the code by modifying the lists. You can run this experiment like this:
```
python experiment2.py
```
As with experiment 1, the same command line arguments apply.

### Analyzing Results
Let's walk through how one generates plots from a log file. This differs from experiment to experiment, so we will go through both.

#### Experiment 1
For a given log file to investigate, navigate to analyze_results1.py and fill in all the TODO tags. This includes filling in the directory that contains the log file of interest as well as filling in the name of the actual log file. After doing those two things, enter the following to generate the plots.
```
python analyze_results1.py
```

#### Experiment 2
Analyzing results for experiment two takes slightly more time than experiment 1. As with experiment 1, find and complete the TODO tags but this time in analyze_results2.py. This includes the TODOS from experiment 1 plus filling in the parameters varied in the trial. Some plots are commented out, but feel free to uncomment them. To see the plots, enter
```
python analyze_results2.py
```
#### Contributors: Dane Brouwer, Marion Lepert, Joshua Citron
