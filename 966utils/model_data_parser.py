
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


# most of the functionality of this function was moved to data_visualizer
def visualize_data():
    total, correct = 0, 0
    for clip in clips:
        name, data = clip[0], clip[1]
        if "Danny" not in name:# and "Shira" not in name:
            clip_name = name.split("_")[0] + "_" + name.split("_")[1] + "/" + name.split("_")[2]
            clip_length = len(hand_locations[hand_locations['imgs'].str.contains(clip_name + "/")])
            data['Frames_Remaining'] = clip_length - data['Frame']
            correct_responses = data[data['Correct']]
            incorrect_responses = data[~data['Correct']]

            frames_remaining_correct = np.array(list(correct_responses['Frames_Remaining']))
            frames_remaining_incorrect = np.array(list(incorrect_responses['Frames_Remaining']))

            all_frame_clicks = np.append(frames_remaining_correct, frames_remaining_incorrect)
            click_points = sorted(list(set(all_frame_clicks)))

            model_trial = model_results[model_results['frame_name'].str.contains(clip_name + "/")]

            # h.groupby(['l'])['f'].sum()/len(h.groupby(['l']))

            click_and_percentage = []
            for grp in pd.DataFrame.groupby(data, by=['Frames_Remaining']):
                grp_name, grp_data = grp
                click_and_percentage.append((grp_name, grp_data['Correct'].sum()/len(grp_data['Correct'])))
            click_and_percentage.sort(key=lambda x: x[0])

            model_and_percentage = []
            for click in click_points:
                try:
                    guess1, prob1 = list(model_trial[model_trial['frame_num'] == clip_length - click].iloc[0])[3:5]
                    guess2, prob2 = list(model_trial[model_trial['frame_num'] == clip_length - click].iloc[0])[5:7]
                except:
                    print(list(model_trial[model_trial['frame_num'] == clip_length - click]))
                    print(clip_length, click)
                    print(len(model_trial))
                    print(model_trial)

                if guess1 == data['Label'].iloc[0]:
                    model_and_percentage.append((click, prob1))
                elif guess2 == data['Label'].iloc[0]:
                    model_and_percentage.append((click, prob2))
                else:
                    model_and_percentage.append((click, 0))
            model_and_percentage.sort(key=lambda x: x[0])

            # average the data
            avg_click = np.average(all_frame_clicks)
            median_click = np.median(all_frame_clicks)
            guess1, prob1 = list(model_trial[model_trial['frame_num'] == clip_length - int(avg_click)].iloc[0])[3:5]
            total += 1
            if guess1 == data['Label'].iloc[0]:
                correct += 1

            # plt.plot(click_points, [i[1] for i in click_and_percentage], color='b')
            # plt.plot(click_points, [i[1] for i in model_and_percentage], color='c')
            # plt.show()

    print("Overall model correctness is ", correct / total)


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
                # guess2, prob2 = list(model_trial[model_trial['frame_num'] == frame_num].iloc[0])[5:7]

                sim.append(selection == guess1)
    print("People and the model's 1st agree on %f percent of data" % (sum(sim) / len(sim)))
    # print("People and the model's 1st and 2nd choices agree on %f percent of data" %(sum(sim)/len(sim)))


evaluate_similarity()


# polynomial fit, with directionalization, and delta 15, with weights [.5,.5], we get .548
# polynomial fit, with directionalization, and delta 20, with weights [.5,.5], we get .516
# polynomial fit, with directionalization, and delta 15, with weights [.3,.7], we get .48
# polynomial fit, with directionalization, and delta 15, with weights [.7,.3], we get .516
# linear fit, with directionalization, and delta 20, with weights [.5,.5], we get .32
# linear fit, with directionalization, and delta 15, with weights [.5,.5], we get .29