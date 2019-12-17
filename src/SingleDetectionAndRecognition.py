import os
import cv2
import numpy as np
from models import CustomModel

PATH = './Data/test-images'


def single_recognition(images):
    """
    Loads the classifier CNN model and weights and invokes the predictor method.
    Sends back the list of labels for the given list of input images
    :param images: Input images list
    :return: List of labels
    """
    images_to_predict = []
    for image in images:
        image = cv2.resize(image, (32, 32)).reshape((32, 32, 1))
        images_to_predict.append(image)
    vals = CustomModel.predict("custom_model.json", "weights.h5",
                               np.array(images_to_predict))
    labels = []
    for i, val in enumerate(vals):
        max_val = np.amax(val)
        label = np.argmax(val)
        if max_val >= 0.5 and label != 10:
            labels.append(label)
            # cv2.imshow("", images[i])
            # cv2.waitKey()
        else:
            labels.append(None)
    return labels


def detect_and_classify(orig_image):
    """
    Detect and classify the given image
    Detects using MSER and send the rois to number classifier
    Sends the output to non-maximal suppression algo and finally
    outputs the image with necessary annotations
    :param orig_image: np array of input image
    :return: image to be saved/displayed
    """
    images_to_predict = []
    if len(orig_image.shape) > 2:
        image = cv2.cvtColor(orig_image.copy(), cv2.COLOR_BGR2GRAY)
    else:
        image = orig_image.copy()
    image_mser = cv2.MSER_create(_max_variation = 0.225)
    rois, _ = image_mser.detectRegions(image)
    bounding_boxes = []
    for roi in rois:
        x2, y2 = np.amax(roi, axis = 0)
        x1, y1 = np.amin(roi, axis = 0)
        if (x2 - x1 <= 0.5 * image.shape[1] or y2 - y1 <= 0.5 * image.shape[0]) \
                and (x2 - x1 >= 0.05 * image.shape[1] and y2 - y1 >= 0.05 * image.shape[0]) \
                and (x2 - x1 <= (y2 - y1) * 2):

            img = image[y1: y2, x1: x2]
            if img.size > 0:
                images_to_predict.append(img)
                bounding_boxes.append((x1, y1, x2, y2))
    if len(images_to_predict) == 0:
        return orig_image
    labels = single_recognition(images_to_predict)
    input_to_nms = []
    for index, label in enumerate(labels):
        if label is None:
            continue
        x1, y1, x2, y2 = bounding_boxes[index]
        input_to_nms.append((x1, y1, x2, y2, label))
    output_from_nms = nms(np.array(input_to_nms), 0.05)
    for x1, y1, x2, y2, label in output_from_nms:
        orig_image = cv2.rectangle(orig_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        orig_image = cv2.putText(orig_image, str(label), org = (x1, y2 + 3), fontFace = cv2.FONT_HERSHEY_SIMPLEX, color = (0, 0, 255), thickness = 2, fontScale = 2)
    return orig_image


def nms(roi, overlap_threshold):
    """
    Perform non-maximal suppression of a set of regions of interest
    :param roi: List of bounding boxes. Each element is a tuple of (X1, Y1, X2, Y2, Class)
    :param overlap_threshold: The overlap threshold  between the bounding boxes
    :return:
    """
    if len(roi) == 0:
        return []

    roi = roi.astype(np.float32)
    final_roi_indices = []

    x1 = roi[:, 0]
    y1 = roi[:, 1]
    x2 = roi[:, 2]
    y2 = roi[:, 3]
    area_rois = (x2 - x1 + 1) * (y2 - y1 + 1)
    sorted_ids_list = np.argsort(y2)

    while len(sorted_ids_list) > 0:
        last = len(sorted_ids_list) - 1
        i = sorted_ids_list[last]
        final_roi_indices.append(i)
        mod_x1 = np.maximum(x1[i], x1[sorted_ids_list[:last]])
        mod_y1 = np.maximum(y1[i], y1[sorted_ids_list[:last]])
        mod_x2 = np.minimum(x2[i], x2[sorted_ids_list[:last]])
        mod_y2 = np.minimum(y2[i], y2[sorted_ids_list[:last]])

        width = np.maximum(0, mod_x2 - mod_x1 + 1)
        height = np.maximum(0, mod_y2 - mod_y1 + 1)
        overlap = (width * height) / area_rois[sorted_ids_list[:last]]

        sorted_ids_list = np.delete(sorted_ids_list, np.concatenate(([last], np.where(overlap > overlap_threshold)[0])))
    return roi[final_roi_indices].astype(np.int)


# for image_name in ["1.jpg", "2.jpg", "3.jpg", "4.jpg", "5.jpg", "6.jpg", "7.jpg"]:
#     image = cv2.imread(os.path.join(PATH, image_name))
#     out_img = detect_and_classify(image)
#     cv2.imwrite(os.path.join(PATH, "OUT_{}".format(image_name)), out_img)
