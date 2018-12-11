

import os
import sys

import cv2
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

from handtracking.utils import label_map_util
import pandas as pd


detection_graph = tf.Graph()
sys.path.append("..")


MODEL_NAME = '../handtracking/hand_inference_graph'
# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT =  MODEL_NAME + '/frozen_inference_graph.pb'
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join(MODEL_NAME, 'hand_label_map.pbtxt')

NUM_CLASSES = 1
# load label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(
    label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


# Load a frozen infrerence graph into memory
def load_inference_graph():

    # load frozen tensorflow model into memory
    print("> ====== loading HAND frozen graph into memory")
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
        sess = tf.Session(graph=detection_graph)
    print(">  ====== Hand Inference graph loaded.")
    return detection_graph, sess


# draw the detected bounding boxes on the images
# You can modify this to also draw a label.
def box_hands(num_hands_detect, score_thresh, scores, boxes, im_width, im_height, image_np, draw=True):
    centers = []
    for i in range(num_hands_detect):
        if (scores[i] > score_thresh):
            (left, right, top, bottom) = (boxes[i][1] * im_width, boxes[i][3] * im_width,
                                          boxes[i][0] * im_height, boxes[i][2] * im_height)
            p1 = (int(left), int(top))
            p2 = (int(right), int(bottom))
            if draw:
                cv2.rectangle(image_np, p1, p2, (77, 255, 9), 3, 1)
            centers.append(((left+right)//2, (top+bottom)//2))
    return image_np, centers


# Actual detection .. generate scores and bounding boxes given an image
def detect_objects(image_np, detection_graph, sess):
    # Definite input and output Tensors for detection_graph
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    # Each box represents a part of the image where a particular object was detected.
    detection_boxes = detection_graph.get_tensor_by_name(
        'detection_boxes:0')
    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name(
        'detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name(
        'detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name(
        'num_detections:0')

    image_np_expanded = np.expand_dims(image_np, axis=0)

    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores,
            detection_classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})
    return np.squeeze(boxes), np.squeeze(scores)


def tag_hands(img_path):
    img = cv2.imread(img_path)
    RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    boxes, scores = detect_objects(RGB_img, detection_graph, sess)
    boxed_img, centers = box_hands(2, .1, scores, boxes, RGB_img.shape[1], RGB_img.shape[0], RGB_img, True)

    xs, ys = [], []
    def onclick(event):
        x, y = event.xdata, event.ydata
        xs.append(x)
        ys.append(y)
        plt.close(fig)

    if scores[0] < .15:
        fig, ax = plt.gcf(), plt.gca()
        ax.imshow(boxed_img)
        cid = fig.canvas.mpl_connect('button_press_event', onclick)
        plt.show()
        x, y = xs[-1], ys[-1]
        print(x, y)
    else:
        center = max(centers, key=lambda x: x[0])
        # they both use their left (so right-most) hand
        x, y = center[0], center[1]
    return int(x), int(y)


def run_tagger(root_folder, trial):
    t = 0
    all_clips = os.listdir(root_folder + "/" + trial)
    for clip in all_clips:
        t += 1
        print("Tagging clip %i out of %i" %(t, len(all_clips)))
        if "clip" in clip:
            img_dir = root_folder + "/" + trial + "/" + clip
            for img_path in os.listdir(img_dir):
                path_name = trial + "/" + clip + "/" + img_path
                x, y = tag_hands(root_folder + "/" + path_name)
                tagged_imgs.append(path_name)
                x_hand_locations.append(x)
                y_hand_locations.append(y)


if __name__ == '__main__':
    root_folder = "../reaching_images"
    detection_graph, sess = load_inference_graph()

    tagged_imgs = []
    x_hand_locations = []
    y_hand_locations = []

    for trial in ['Shira_Standing', 'Tao_Standing']:
        run_tagger(root_folder, trial)

    d = {"imgs" : tagged_imgs, "x": x_hand_locations, "y": y_hand_locations}
    df = pd.DataFrame(d)
    df.to_csv("./trial_data/hand_locations.csv")


