# sudo -E env PATH=${PATH} /home/panjacob/PycharmProjects/handmouse/venv/bin/python detect.py
# xhost +
import os
import time
import cv2
import numpy as np
import tensorflow as tf
from object_detection.builders import model_builder
from object_detection.utils import config_util
from object_detection.utils import label_map_util
from pynput.mouse import Button, Controller
from utilis import create_path
from utilis import paths, files
from screeninfo import get_monitors


@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections


def detect_from_img(img):
    image_np = np.array(img)
    image_np_expanded = np.expand_dims(image_np, 0)
    input_tensor = tf.convert_to_tensor(image_np_expanded, dtype=tf.float32)
    return detect_fn(input_tensor)


def parse_detections(detections_raw):
    num_detections = int(detections_raw.pop('num_detections'))
    scores = detections_raw['detection_scores']
    score_count = np.count_nonzero(scores.numpy() > MIN_SCORE)
    if num_detections > MAX_DETECTIONS:
        num_detections = MAX_DETECTIONS
    if num_detections > score_count:
        num_detections = score_count

    detections_raw = {key: value[0, :num_detections].numpy() for key, value in detections_raw.items()}
    detections_raw['num_detections'] = num_detections
    detections_raw['detection_classes'] = detections_raw['detection_classes'].astype(np.int64)
    detections_raw['detection_boxes'] *= IMAGE_SIZE
    detections_raw['detection_boxes'] = np.rint(detections_raw['detection_boxes']).astype(int)

    return list(
        zip(detections_raw['detection_scores'], detections_raw['detection_boxes'], detections_raw['detection_classes']))


def control_mouse(detections, img, CLICKTIME):
    for score, box, detclass in detections:
        ymin, xmin, ymax, xmax = box
        middle = ((xmin + xmax) / 2 / 320, (ymin + ymax) / 2 / 320)
        # cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 1)

        if detclass == 1 and time.time() - CLICKTIME > 0.5:
            img = cv2.circle(img, (int(middle[0] * 320), int(middle[1] * 320)), radius=10, color=(0, 0, 255),
                             thickness=-1)
            MOUSE.press(Button.left)
            CLICKTIME = time.time()
        elif detclass == 0:
            img = cv2.circle(img, (int(middle[0] * 320), int(middle[1] * 320)), radius=10, color=(0, 255, 255),
                             thickness=-1)
            MOUSE.release(Button.left)
        x = middle[0] * SCREEN_RESOLUTION[0] + (middle[0] - 0.5) * MOUSE_MOVE_SPEEDUP[0]
        y = middle[1] * SCREEN_RESOLUTION[1] + (middle[1] - 0.5) * MOUSE_MOVE_SPEEDUP[1]
        MOUSE.position = (x, y)

        return img


def init_detection_model():
    configs = config_util.get_configs_from_pipeline_file(files['PIPELINE_CONFIG'])
    detection_model = model_builder.build(model_config=configs['model'], is_training=False)
    ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
    ckpt.restore(os.path.join(paths['CHECKPOINT_PATH'], 'ckpt-5')).expect_partial()
    category_index = label_map_util.create_category_index_from_labelmap(files['LABELMAP'])
    return detection_model


def init_video(path):
    vidcap = cv2.VideoCapture(os.path.join(paths['TRACK_PATH'], '1.mp4'))
    vidcap.set(cv2.CAP_PROP_POS_FRAMES, 800)
    success, img = vidcap.read()
    return vidcap, success


def display(img):
    cv2.imshow('hehe', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        pass


def get_filename(file):
    return file.split('.')[0]


def convert(size, box):
    dr = 1. / size

    c = box[0]
    d = box[1]
    a = box[2]
    b = box[3]

    x = (a + b) / 2.0
    y = (c + d) / 2.0
    w = b - a
    h = d - c

    x = x * dr
    w = w * dr
    y = y * dr
    h = h * dr
    return (x, y, w, h)


def save_detections(detections, file, destination_path_true):
    filename = get_filename(file)
    yolo_file = ""
    for detection in detections:
        acc, box, class_number = detection
        x, y, w, h = convert(320, box)
        value = f"{class_number} {x} {y} {w} {h}\n"
        yolo_file += value
    f = open(os.path.join(destination_path_true, f"{filename}.txt"), "w")
    f.write(yolo_file)
    f.close()


def preprocess_frame(frame):
    img = cv2.resize(frame, (IMAGE_SIZE, IMAGE_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return cv2.flip(img, 1)


if __name__ == "__main__":
    MAX_DETECTIONS = 1
    IMAGE_SIZE = 320
    MIN_SCORE = 0.7
    monitor = get_monitors()[0]
    SCREEN_RESOLUTION = (monitor.width, monitor.height)
    MOUSE_MOVE_RATIO = 1.5
    MOUSE_MOVE_SPEEDUP = (SCREEN_RESOLUTION[0] * MOUSE_MOVE_RATIO, SCREEN_RESOLUTION[1] * MOUSE_MOVE_RATIO)
    CLICKTIME = time.time()
    MOUSE = Controller()
    detection_model = init_detection_model()
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if frame is None:
            print('frame not found')
            continue  # sometimes I have to wait for camera
        img = preprocess_frame(frame)

        detections_raw = detect_from_img(img)
        detections = parse_detections(detections_raw)
        if len(detections):
            img = control_mouse(detections, img, CLICKTIME)

        display(img)

cap.release()
cv2.destroyAllWindows()
