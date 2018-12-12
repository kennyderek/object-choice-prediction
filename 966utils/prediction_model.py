import os

import pandas as pd
import numpy as np
import math
from copy import deepcopy
import matplotlib.pyplot as plt
import cv2

from scipy.odr import ODR, Model, Data

DELTA = 10
MODEL_TYPE = "poly"
WEIGHTS = [.5,.5]


def linear(B, x):
    return B[0] * x + B[1]


def poly(B, x):
    return B[0]*x**2 + B[1]*x + B[2]


def predict_path(clip, model_type=MODEL_TYPE):
    clip_data = hand_centers[hand_centers['imgs'].str.contains(clip + "/")]
    clip_data.sort_values(['imgs'], ascending=[1], inplace=True)

    object_locations = list(zip(list(objects['x_pos']), list(objects['y_pos'])))
    previous_update, latest_update = [(i, 1 / 16) for i in object_locations], [(i, 1 / 16) for i in object_locations]
    for current_frame in range(1, len(clip_data)):
        frame = clip_data.iloc[current_frame][0]
        upper_range = current_frame
        lower_range = max(0, upper_range - DELTA)
        segment_data = clip_data.iloc[lower_range:upper_range + 1]
        segment_data = segment_data[abs(segment_data['x'].mean()-segment_data['x']) < 2*segment_data['x'].std()]
        segment_data = segment_data[abs(segment_data['y'].mean()-segment_data['y']) < 2*segment_data['y'].std()]

        x = list(segment_data['x'])
        y = list(segment_data['y'])

        if len(x) == 0:
            write_line_to_storage(frame, current_frame, latest_update)
            continue

        mydata = Data(x, y)
        if model_type == 'linear':
            f = linear
            mod = Model(linear)
            myodr = ODR(mydata, mod, beta0=[0, 2])
        elif model_type == 'poly':
            f = poly
            mod = Model(poly)
            myodr = ODR(mydata, mod, beta0=[0, 0, 2])

        res = myodr.run()
        coeff = res.beta

        obj_and_metric = measure_objects(f, coeff)
        # we want to choose items only past the hand, in the direction we think the hand is going!
        if x[-1] >= x[0]:
            hand_movement_dir = "right"
        else:
            hand_movement_dir = "left"
        for i in range(len(obj_and_metric)):
            obj_loc = obj_and_metric[i][0]
            if ((obj_loc[0] < x[-1]) and hand_movement_dir == "left") or \
                    ((obj_loc[0] > x[-1]) and hand_movement_dir == "right"):
                pass
            else:
                obj_and_metric[i] = (obj_and_metric[i][0], np.inf)

        latest_update = softmax(obj_and_metric, return_sorted=True)

        latest_update = weigh_probabilities(previous_update, latest_update)
        previous_update = deepcopy(latest_update)

        # plot_line_probs_on_img(f, coeff, latest_update, root_folder, clip_data.iloc[current_frame]['imgs'])

        write_line_to_storage(frame, current_frame, latest_update)


def plot_line_probs_on_img(f, coeff, latest_update, root_folder, image_path):
    img = cv2.imread(root_folder + "/" + image_path)
    for x in range(0, img.shape[1]):
        y = f(coeff, x)
        cv2.circle(img, (int(x), int(y)), radius=2, color=(0,255,0))
    for obj in latest_update:
        cv2.circle(img, obj[0],radius=int(obj[1]*10), color=(0,255,0))
    plt.imshow(img)
    plt.show()


def weigh_probabilities(previous, current):
    p, c = sorted(previous, key= lambda x: (x[0][0], x[0][1])), sorted(current, key= lambda x: (x[0][0], x[0][1]))
    probs_p = np.array([prob[1] for prob in p])
    probs_c = np.array([prob[1] for prob in c])
    updated = np.array([WEIGHTS])@np.array([probs_p, probs_c])
    final = [(p[i][0], updated[0][i]) for i in range(16)]
    return sorted(final, key=lambda x: x[1], reverse=True)


def write_line_to_storage(frame_name, frame_num, results):
    output['frame_name'].append(frame_name)
    output['frame_num'].append(frame_num)
    output['obj1_name'].append(get_object_name(results[0][0]))
    output['obj1_prob'].append(results[0][1])
    output['obj2_name'].append(get_object_name(results[1][0]))
    output['obj2_prob'].append(results[1][1])


def get_object_name(coords):
    return objects[(objects['x_pos'] == coords[0]) & (objects['y_pos'] == coords[1])]['object_name'].iloc[0]


def measure_objects(f, coeff):
    obj_locations = list(zip(list(objects['x_pos']), list(objects['y_pos'])))
    obj_and_metric = []
    for obj in obj_locations:
        error = abs(obj[1] - f(coeff, obj[0]))
        obj_and_metric.append((obj, error))
    return obj_and_metric


def softmax(object_and_metric, return_sorted=False):
    total = 0
    try:
        for obj in object_and_metric:
            total += math.exp(-math.log(obj[1]+.00001))
        probs = [(i[0], math.exp(-math.log(i[1]+.00001))/total) for i in object_and_metric]
        total_probs = np.sum([i[1] for i in probs])
    except:
        print('log error on', object_and_metric)
        total_probs = 1
        probs = [(i, 1/16) for i in list(zip(list(objects['x_pos']), list(objects['y_pos'])))]

    if abs(1 - total_probs) > 0.001:
        print('Warning! Total probability over metric does not sum to 1. It is', total_probs)

    if return_sorted:
        return sorted(probs, key=lambda x: x[1], reverse=True)
    else:
        return probs


def run_model(root_folder, trial):
    for clip in os.listdir(root_folder + '/' + trial):
        if "_annotated" not in clip and "clip" in clip:
            predict_path(trial + '/' + clip)


if __name__ == '__main__':
    root_folder = "../reaching_images"
    hand_centers = pd.read_csv("./trial_data/sitting_hand_locations.csv", index_col=0)
    objects = pd.read_csv("./trial_data/object_positions.csv", sep=" ")
    output = {'frame_name': [], 'frame_num':[], 'obj1_name': [], 'obj1_prob':[], 'obj2_name':[], 'obj2_prob':[]}

    for trial in ['Tao_Sitting']:
        run_model(root_folder, trial)

    pd.DataFrame(output).to_csv("./model_results/sitting%s%i%s%s.csv" %(MODEL_TYPE,
                                                                 DELTA,
                                                                 str(WEIGHTS[0]),
                                                                 str(WEIGHTS[1])))



