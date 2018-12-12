
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
results_file = ("./model_results/sitting%s%i%s%s.csv" %(MODEL_TYPE, DELTA, str(WEIGHTS[0]), str(WEIGHTS[1])))

total, correct = 0, 0
# now I want to break the data up into data per Clip
object_positions = pd.read_csv("./trial_data/object_positions.csv", sep=" ")
hand_locations = pd.read_csv("./trial_data/sitting_hand_locations.csv")
model_results = pd.read_csv(results_file)
correct_objects = pd.read_csv("./trial_data/touched_objects.txt", sep=" ")

frames15 = []
frames20 = []
frames25 = []
for clip_name in list(os.listdir("../reaching_images/Tao_Sitting")):
    clip_length = len(hand_locations[hand_locations['imgs'].str.contains(clip_name + "/")])
    correct_object = correct_objects[correct_objects['Clip'] == clip_name].iloc[0]['Label']
    model_trial = model_results[model_results['frame_name'].str.contains(clip_name + "/")]
    guess1, prob1 = list(model_trial[model_trial['frame_num'] == clip_length - 15].iloc[0])[3:5]
    guess2, prob2 = list(model_trial[model_trial['frame_num'] == clip_length - 20].iloc[0])[3:5]
    guess3, prob3 = list(model_trial[model_trial['frame_num'] == clip_length - 25].iloc[0])[3:5]
    frames15.append(guess1 == correct_object)
    frames20.append(guess2 == correct_object)
    frames25.append(guess3 == correct_object)

    hand_tracking_data = hand_locations[hand_locations['imgs'].str.contains(clip_name + "/")]
    hand_tracking_data.sort_values(['imgs'], ascending=[1], inplace=True)

    i = 0
    for delta in [40, 30, 20, 10]:
        i += 1
        model_image = cv2.imread("../reaching_images/" + hand_tracking_data.iloc[clip_length - delta]['imgs'])
        model_image = cv2.cvtColor(model_image, cv2.COLOR_BGR2RGB)
        all_guesses = {}
        model_guess_num = 0
        for click in range(delta-10, delta+1, 2):
            try:
                guess1, prob1 = list(model_trial[model_trial['frame_num'] == clip_length - click].iloc[0])[3:5]
                if guess1 in all_guesses:
                    all_guesses[guess1] += 1
                else:
                    all_guesses[guess1] = 1
                model_guess_num += 1
            except:
                pass
        model_heat_map_dist = list(all_guesses.items())
        for item in model_heat_map_dist:
            object_name, freq = item
            x, y = list(object_positions[object_positions["object_name"] == object_name].iloc[0])[1:3]
            if object_name == correct_object:
                cv2.circle(model_image, center=(x, y), radius=int(freq / model_guess_num * 10),
                           color=(0, 255, 0), thickness=3)
            else:
                cv2.circle(model_image, center=(x, y), radius=int(freq / model_guess_num * 10),
                           color=(255, 0, 0), thickness=3)

        plt.subplot(int("22"+str(i)))
        plt.imshow(model_image)
        plt.tick_params(axis='both', which='both', left=False, bottom=False, top=False, labelbottom=False, labelleft=False)
        plt.xlabel("%i frames remaining" %(delta))

    model_image = cv2.imread("../reaching_images/" + hand_tracking_data.iloc[clip_length-1]['imgs'])
    model_image = cv2.cvtColor(model_image, cv2.COLOR_BGR2RGB)
    for j in range(len(hand_tracking_data)):
        center = (hand_tracking_data.iloc[j]['x'], hand_tracking_data.iloc[j]['y'])
        cv2.circle(model_image, center=center, radius=1, thickness=2, color=(0,255,0))
    plt.subplot(224)
    plt.imshow(model_image)
    plt.xlabel("Path of hand centers")

    plt.suptitle("%s Heat Maps of Selected Objects" %(clip_name))
    plt.savefig("./sitting_model_results/" + clip_name)


print(sum(frames15)/len(frames15), "agreement at -15 frames") #.5
print(sum(frames20)/len(frames20), "agreement at -20 frames") #.4375
print(sum(frames25)/len(frames25), "agreement at -25 frames") #.25


