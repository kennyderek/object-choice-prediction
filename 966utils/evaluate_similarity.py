
import os

import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
pd.set_option('display.max_rows', 20)
pd.set_option('display.max_columns', 10)

MODEL_TYPE = "poly"
DELTA = 15
WEIGHTS = [.5,.5]
results_file = ("./model_results/%s%i%s%s.csv" %(MODEL_TYPE, DELTA, str(WEIGHTS[0]), str(WEIGHTS[1])))

# now I want to break the data up into data per Clip
object_positions = pd.read_csv("./trial_data/object_positions.csv", sep=" ")
hand_locations = pd.read_csv("./trial_data/hand_locations.csv")
parsed_tao_data = pd.read_csv("./trial_data/parsed_tao_data.csv")
model_results = pd.read_csv(results_file)
clips = pd.DataFrame.groupby(parsed_tao_data, by="Clip")


# evaluate similarity of model results to the people's results
def evaluate_similarity():
    parsed_tao_data = pd.read_csv("./trial_data/parsed_tao_data.csv")
    # parsed_tao_data = parsed_tao_data[parsed_tao_data['Correct'] == True]
    clips = pd.DataFrame.groupby(parsed_tao_data, by="Clip")
    sim = []
    for clip in clips:
        name, data = clip[0], clip[1]
        if "Danny" not in name:
            clip_name = name.split("_")[0] + "_" + name.split("_")[1] + "/" + name.split("_")[2]
            model_trial = model_results[model_results['frame_name'].str.contains(clip_name + "/")]
            for i in range(len(data)):
                frame_num = data.iloc[i]['Frame']
                selection = data.iloc[i]['Selection']
                guess1, prob1 = list(model_trial[model_trial['frame_num'] == frame_num].iloc[0])[3:5]
                guess2, prob2 = list(model_trial[model_trial['frame_num'] == frame_num].iloc[0])[5:7]

                sim.append(selection == guess1)
                # sim.append((selection == guess1) or (selection == guess2))
    print("People and the model's 1st agree on %f percent of data" % (sum(sim) / len(sim)))
    # print("People and the model's 1st and 2nd choices agree on %f percent of data" %(sum(sim)/len(sim)))


evaluate_similarity()
