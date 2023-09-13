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
git clone git@github.com:danebrouwer/clutter-jamming.git
cd clutter
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
Let's walk through the two experiments that can be run.
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

#### Contributors: Dane Brouwer, Marion Lepert, Joshua Citron
