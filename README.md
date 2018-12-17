# object-choice-prediction

## Requirements
Python 3

Required packages: pandas, numpy, cv2, matplotlib

## File Description

All files to build the graphs and run the models are in 966utils. The hand locations of most images are already tagged and rest in trial_data/hand_locations.csv (for the standing-only experiment), and sitting_hand_locations.csv (for Tao's portion of the sitting experiment), so it is not necessary to run tagger.py (which needs the hand-tracking submodule) to do tagging.


### To run the model with different parameters (ex: linear/polynomial) for the standing-only experiment:
edit and then run prediction_model.py. This file takes the hand locations and computes the probability distribution of the objects for each frame. This file outputs to model_results folder, with the file name dependant on the parameter specifications used. For the following files, reference the specific desired model_results file in this folder.

### To build the graph for standing-only:
Run data_visualizer.py

### To build the visualizations and get accuracy for sitting-experiment:
Run sitting_model_data_parser.py

### To determine how similar human observer and model predictions are:
Run evaluate_similarity.py
