import tensorflow as tf
import numpy as np
import cv2
import colorsys
import random
from IPython.display import Image, display

# As tensorflow lite doesn't support tf.size used in tf.meshgrid, 
# we reimplemented a simple meshgrid function that use basic tf function.
def _meshgrid(n_a, n_b):

    return [
        tf.reshape(tf.tile(tf.range(n_a), [n_b]), (n_b, n_a)),
        tf.reshape(tf.repeat(tf.range(n_b), n_a), (n_b, n_a))
    ]


def yolo_head(preds, anchors, classes): #preds中有3个pred
    # pred: (batch_size, grid, grid, anchors, (x, y, w, h, obj, ...classes))
    outputs = {}
    
    for i in range(3):
        pred = preds[i]
        grid_size = tf.shape(pred)[1:3]
        box_xy, box_wh, objectness, class_probs = tf.split(
            pred, (2, 2, 1, classes), axis=-1)

        box_xy = tf.sigmoid(box_xy)
        objectness = tf.sigmoid(objectness)
        class_probs = tf.sigmoid(class_probs)
        #pred_box = tf.concat((box_xy, box_wh), axis=-1)  # original xywh for loss

        # !!! grid[x][y] == (y, x)
        grid = _meshgrid(grid_size[1],grid_size[0])
        grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)  # [gx, gy, 1, 2]

        box_xy = (box_xy + tf.cast(grid, tf.float32)) / \
            tf.cast(grid_size, tf.float32)
        box_wh = tf.exp(box_wh) * anchors[[6-i*3,7-i*3,8-i*3]]

        box_x1y1 = box_xy - box_wh / 2
        box_x2y2 = box_xy + box_wh / 2
        bbox = tf.concat([box_x1y1, box_x2y2], axis=-1)

        outputs['output'+str(i)] = (objectness, bbox, class_probs)

    return (outputs['output0'],outputs['output1'],outputs['output2'])


def read_classes(classes_path):
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def read_anchors(anchors_path):
    with open(anchors_path) as f:
        anchors = f.readline()
        anchors = [float(x) for x in anchors.split()]
        anchors = np.array(anchors).reshape(-1, 2)
    return anchors

def generate_colors(class_names):
    hsv_tuples = [(x / len(class_names), 1., 1.) for x in range(len(class_names))]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
    random.seed(10201)  # Fixed seed for consistent colors across runs.
    random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
    random.seed(None)  # Reset seed to default.
    return colors


def preprocess_image(img_path, model_image_size):
    img_raw = tf.image.decode_image(open('./images/'+img_path, 'rb').read(), channels=3)
    img = tf.expand_dims(img_raw, 0)
    img = tf.image.resize(img, model_image_size)/255.
    return img_raw, img

def draw_outputs(img, out_scores, out_boxes, out_classes, colors, class_names):

    wh = np.flip(img.shape[0:2])
    for i,c in list(enumerate(out_classes)):
        x1y1 = tuple((np.array(out_boxes[i][0:2]) * wh).astype(np.int32))
        x2y2 = tuple((np.array(out_boxes[i][2:4]) * wh).astype(np.int32))
        x1y1_lable = tuple((np.array(out_boxes[i][0:2]) * wh + [0,-15]).astype(np.int32))
        x2y2_lable = tuple((np.array(out_boxes[i][0:2]) * wh + [(len(class_names[int(out_classes[i])])+6)*12,0]).astype(np.int32))
        img = cv2.rectangle(img, x1y1, x2y2, colors[c], 2)
        img = cv2.rectangle(img, x1y1_lable, x2y2_lable, colors[c], -1)
        img = cv2.putText(img, '{} {:.2f}'.format(
            class_names[int(out_classes[i])], out_scores[i]),
            x1y1, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), 1)
        print('{} {:.2f}'.format(class_names[int(out_classes[i])], out_scores[i]),
              x1y1, x2y2)
    return img

