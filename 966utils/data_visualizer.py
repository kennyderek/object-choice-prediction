
import os

import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
figure(num=None, figsize=(12, 10), dpi=80, facecolor='w', edgecolor='k')
pd.set_option('display.max_rows', 20)
pd.set_option('display.max_columns', 10)

# to evaluate model results
MODEL_TYPE = "poly"
DELTA = 15
WEIGHTS = [.5,.5]
results_file = ("./model_results/%s%i%s%s.csv" %(MODEL_TYPE, DELTA, str(WEIGHTS[0]), str(WEIGHTS[1])))

SAVE_PLOTS = True
SHOW_PLOTS = False

all = []
main_dir = ["../tao_data_complete"]
for dir in main_dir:
    files = os.listdir(dir)
    for file in files:
        if "disqual" not in file:
            f = open(dir + "/" + file, "r",encoding='ISO-8859-1')
            lines = f.readlines()
            for line_idx in range(len(lines)):
                if "txt" in lines[line_idx]:
                    e = lines[line_idx].split(" ")
                    selection = e[len(e)-2]
                    if "B" in selection or "G" in selection or "W" in selection or "R" in selection:
                        all.append({'IP': e[0], "Frame": int(e[len(e)-5]), "Clip": e[6].split(".txt")[0], "Selection": e[len(e)-2]})
people = pd.DataFrame(all)
labels = pd.read_csv("./trial_data/touched_objects.txt", sep=" ")

res = pd.merge(people, labels)
res["Correct"] = res["Selection"] == res["Label"]

# filter for people who got less than 50% correct
groups = pd.DataFrame.groupby(res, by='IP')
bad_IPs = set([])
total, correct = 0, 0
for group in groups:
    name, data = group[0], group[1]
    if data["Correct"].sum()/len(data) < .5:
        print("Bad IP: ", name)
        bad_IPs.add(name)
res = res[~res["IP"].isin(bad_IPs)]


# now I want to break the data up into data per Clip
object_positions = pd.read_csv("./trial_data/object_positions.csv", sep=" ")
hand_locations = pd.read_csv("./trial_data/hand_locations.csv")
res.to_csv("./trial_data/parsed_tao_data.csv")
model_results = pd.read_csv(results_file)
clips = pd.DataFrame.groupby(res, by="Clip")
for clip in clips:
    name, data = clip[0], clip[1]
    if "Danny" not in name:
        # need to find the length of the frame
        clip_name = name.split("_")[0] + "_" + name.split("_")[1] + "/" + name.split("_")[2]
        clip_length = len(hand_locations[hand_locations['imgs'].str.contains(clip_name + "/")])
        data['Frames_Remaining'] = clip_length - data['Frame']
        correct_responses = data[data['Correct']]
        incorrect_responses = data[~data['Correct']]

        frames_remaining_correct = np.array(list(correct_responses['Frames_Remaining']))
        frames_remaining_incorrect = np.array(list(incorrect_responses['Frames_Remaining']))
        all_frame_clicks = np.append(frames_remaining_correct, frames_remaining_incorrect)
        click_points = sorted(list(set(all_frame_clicks)))

        plt.subplot(221)
        x = plt.hist([frames_remaining_correct, frames_remaining_incorrect],
                     color=['green', 'red'], label=['correct', 'incorrect'],
                     rwidth=.7, histtype='barstacked')
        plt.xticks([i for i in list(range(int(click_points[0])-int(click_points[0])%5, int(click_points[-1])+5, 5))])
        plt.ylabel('Number of Clicks')
        plt.xlabel('Frames Remaining')
        plt.title('MTurk Data for %s' % name)
        plt.legend()

        plt.subplot(222)
        hand_tracking_data = hand_locations[hand_locations['imgs'].str.contains(clip_name + "/")]
        hand_tracking_data.sort_values(['imgs'], ascending=[1], inplace=True)
        # get the image corresponding to the +1 standard dev. click
        all_frame_clicks = np.append(frames_remaining_correct, frames_remaining_incorrect)
        primary_frame_num = clip_length - int(all_frame_clicks.mean()) + int(all_frame_clicks.std())
        secondary_frame_num = clip_length - int(all_frame_clicks.mean())
        primary_img = cv2.imread("../reaching_images/" +
                                 hand_tracking_data.iloc[primary_frame_num]['imgs'])
        primary_img = cv2.cvtColor(primary_img, cv2.COLOR_BGR2RGB)

        heat_map_dist = list(data.groupby(['Selection']).size().items())
        for item in heat_map_dist:
            object_name, freq = item
            x, y = list(object_positions[object_positions["object_name"] == object_name].iloc[0])[1:3]
            if object_name == data['Label'].iloc[0]:
                cv2.circle(primary_img, center=(x, y), radius=int(freq/len(data)*10),
                           color=(0, 255, 0), thickness=3)
            else:
                cv2.circle(primary_img, center=(x, y), radius=int(freq/len(data)*10),
                           color=(255, 0, 0), thickness=3)
        plt.imshow(primary_img)
        # plt.imshow(secondary_img, alpha=.3)
        plt.title("Frame %i out of %i, %i remaining" %(primary_frame_num, clip_length, clip_length-primary_frame_num))
        plt.tick_params(axis='both', which='both', left=False, bottom=False, top=False, labelbottom=False, labelleft=False)
        plt.xlabel("Heat Map of MTurk Selections")

        plt.subplot(223)
        model_trial = model_results[model_results['frame_name'].str.contains(clip_name + "/")]

        # h.groupby(['l'])['f'].sum()/len(h.groupby(['l']))

        click_and_percentage = []
        for grp in pd.DataFrame.groupby(data, by=['Frames_Remaining']):
            grp_name, grp_data = grp
            click_and_percentage.append((grp_name, grp_data['Correct'].sum()/len(grp_data['Correct'])))
        click_and_percentage.sort(key=lambda x: x[0])

        model_and_percentage = []
        for click in click_points:
            guess1, prob1 = list(model_trial[model_trial['frame_num'] == clip_length - click].iloc[0])[3:5]
            guess2, prob2 = list(model_trial[model_trial['frame_num'] == clip_length - click].iloc[0])[5:7]

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

        plt.scatter(click_points, [i[1] for i in click_and_percentage], color='b', label="MTurk", marker='D')
        if MODEL_TYPE == "poly":
            graph_label = "Polynomial Model"
        else:
            graph_label = "Linear Model"
        plt.scatter(click_points, [i[1] for i in model_and_percentage], color='c', label=graph_label, marker='D')
        plt.ylim(-.1, 1.1)
        plt.xticks([i for i in list(range(int(click_points[0])-int(click_points[0])%5, int(click_points[-1])+5, 5))])
        plt.ylabel('Percent Correct')
        plt.xlabel('Frames Remaining')
        plt.legend()

        plt.subplot(224)
        model_image = cv2.imread("../reaching_images/" + hand_tracking_data.iloc[primary_frame_num]['imgs'])
        model_image = cv2.cvtColor(model_image, cv2.COLOR_BGR2RGB)
        all_guesses = {}
        model_guess_num = 0
        for click in click_points:
            guess1, prob1 = list(model_trial[model_trial['frame_num'] == clip_length - click].iloc[0])[3:5]
            if guess1 in all_guesses:
                all_guesses[guess1] += 1
            else:
                all_guesses[guess1] = 1
            model_guess_num += 1
        model_heat_map_dist = list(all_guesses.items())
        for item in model_heat_map_dist:
            object_name, freq = item
            x, y = list(object_positions[object_positions["object_name"] == object_name].iloc[0])[1:3]
            if object_name == data['Label'].iloc[0]:
                cv2.circle(model_image, center=(x, y), radius=int(freq/model_guess_num*10),
                           color=(0, 255, 0), thickness=3)
            else:
                cv2.circle(model_image, center=(x, y), radius=int(freq/model_guess_num*10),
                           color=(255, 0, 0), thickness=3)
        plt.imshow(model_image)
        # plt.title("Frame %i out of %i" %(primary_frame_num, clip_length))
        plt.tick_params(axis='both', which='both', left=False, bottom=False, top=False, labelbottom=False, labelleft=False)
        plt.xlabel("Heat Map of Model Selections")

        if SAVE_PLOTS:
            plt.savefig("./tao_data_graphs/" + name)
        if SHOW_PLOTS:
            plt.show()
        plt.clf()

print("Model finished with an overall accuracy of", correct/total)