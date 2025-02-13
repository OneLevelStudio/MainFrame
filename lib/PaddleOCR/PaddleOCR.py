import os
import cv2
import math
import numpy as np

# ====================================================================================================

class POCR_Detection:
    def __init__(self, onnx_path, session=None):
        self.session = session
        if self.session is None:
            assert onnx_path is not None
            assert os.path.exists(onnx_path)
            from onnxruntime import InferenceSession
            self.session = InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        self.inputs = self.session.get_inputs()[0]
        self.min_size = 3
        self.max_size = 960
        self.box_thresh = 0.8
        self.mask_thresh = 0.8
        self.mean = np.array([123.675, 116.28, 103.53])  # imagenet mean
        self.mean = self.mean.reshape(1, -1).astype('float64')
        self.std = np.array([58.395, 57.12, 57.375])  # imagenet std
        self.std = 1 / self.std.reshape(1, -1).astype('float64')

    def filter_polygon(self, points, shape):
        width = shape[1]
        height = shape[0]
        filtered_points = []
        for point in points:
            if type(point) is list:
                point = np.array(point)
            point = self.clockwise_order(point)
            point = self.clip(point, height, width)
            w = int(np.linalg.norm(point[0] - point[1]))
            h = int(np.linalg.norm(point[0] - point[3]))
            if w <= 3 or h <= 3:
                continue
            filtered_points.append(point)
        return np.array(filtered_points)

    def boxes_from_bitmap(self, output, mask, dest_width, dest_height):
        mask = (mask * 255).astype(np.uint8)
        height, width = mask.shape
        outs = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if len(outs) == 2:
            contours = outs[0]
        else:
            contours = outs[1]
        boxes = []
        scores = []
        for index in range(len(contours)):
            contour = contours[index]
            points, min_side = self.get_min_boxes(contour)
            if min_side < self.min_size:
                continue
            points = np.array(points)
            score = self.box_score(output, contour)
            if self.box_thresh > score:
                continue
            box, min_side = self.get_min_boxes(points)
            if min_side < self.min_size + 2:
                continue
            box = np.array(box)
            box[:, 0] = np.clip(np.round(box[:, 0] / width * dest_width), 0, dest_width)
            box[:, 1] = np.clip(np.round(box[:, 1] / height * dest_height), 0, dest_height)
            boxes.append(box.astype("int32"))
            scores.append(score)
        return np.array(boxes, dtype="int32"), scores

    @staticmethod
    def get_min_boxes(contour):
        bounding_box = cv2.minAreaRect(contour)
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])
        if points[1][1] > points[0][1]:
            index_1 = 0
            index_4 = 1
        else:
            index_1 = 1
            index_4 = 0
        if points[3][1] > points[2][1]:
            index_2 = 2
            index_3 = 3
        else:
            index_2 = 3
            index_3 = 2
        box = [points[index_1], points[index_2], points[index_3], points[index_4]]
        return box, min(bounding_box[1])

    @staticmethod
    def box_score(bitmap, contour):
        h, w = bitmap.shape[:2]
        contour = contour.copy()
        contour = np.reshape(contour, (-1, 2))
        x1 = np.clip(np.min(contour[:, 0]), 0, w - 1)
        y1 = np.clip(np.min(contour[:, 1]), 0, h - 1)
        x2 = np.clip(np.max(contour[:, 0]), 0, w - 1)
        y2 = np.clip(np.max(contour[:, 1]), 0, h - 1)
        mask = np.zeros((y2 - y1 + 1, x2 - x1 + 1), dtype=np.uint8)
        contour[:, 0] = contour[:, 0] - x1
        contour[:, 1] = contour[:, 1] - y1
        contour = contour.reshape((1, -1, 2)).astype("int32")
        cv2.fillPoly(mask, contour, color=(1, 1))
        return cv2.mean(bitmap[y1:y2 + 1, x1:x2 + 1], mask)[0]

    @staticmethod
    def clockwise_order(point):
        poly = np.zeros((4, 2), dtype="float32")
        s = point.sum(axis=1)
        poly[0] = point[np.argmin(s)]
        poly[2] = point[np.argmax(s)]
        tmp = np.delete(point, (np.argmin(s), np.argmax(s)), axis=0)
        diff = np.diff(np.array(tmp), axis=1)
        poly[1] = tmp[np.argmin(diff)]
        poly[3] = tmp[np.argmax(diff)]
        return poly

    @staticmethod
    def clip(points, h, w):
        for i in range(points.shape[0]):
            points[i, 0] = int(min(max(points[i, 0], 0), w - 1))
            points[i, 1] = int(min(max(points[i, 1], 0), h - 1))
        return points

    def resize(self, image):
        h, w = image.shape[:2]
        # limit the max side
        if max(h, w) > self.max_size:
            if h > w:
                ratio = float(self.max_size) / h
            else:
                ratio = float(self.max_size) / w
        else:
            ratio = 1.
        resize_h = max(int(round(int(h * ratio) / 32) * 32), 32)
        resize_w = max(int(round(int(w * ratio) / 32) * 32), 32)
        return cv2.resize(image, (resize_w, resize_h))

    @staticmethod
    def zero_pad(image):
        h, w, c = image.shape
        pad = np.zeros((max(32, h), max(32, w), c), np.uint8)
        pad[:h, :w, :] = image
        return pad

    def __call__(self, x):
        h, w = x.shape[:2]
        if sum([h, w]) < 64:
            x = self.zero_pad(x)
        x = self.resize(x)
        x = x.astype('float32')
        cv2.subtract(x, self.mean, x)  # inplace
        cv2.multiply(x, self.std, x)  # inplace
        x = x.transpose((2, 0, 1))
        x = np.expand_dims(x, axis=0)
        outputs = self.session.run(None, {self.inputs.name: x})[0]
        outputs = outputs[0, 0, :, :]
        boxes, scores = self.boxes_from_bitmap(outputs, outputs > self.mask_thresh, w, h)
        return self.filter_polygon(boxes, (h, w))

class POCR_Classification:
    def __init__(self, onnx_path, session=None):
        self.session = session
        if self.session is None:
            assert onnx_path is not None
            assert os.path.exists(onnx_path)
            from onnxruntime import InferenceSession
            self.session = InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        self.inputs = self.session.get_inputs()[0]
        self.threshold = 0.98
        self.labels = ['0', '180']

    @staticmethod
    def resize(image):
        input_c = 3
        input_h = 48
        input_w = 192
        h = image.shape[0]
        w = image.shape[1]
        ratio = w / float(h)
        if math.ceil(input_h * ratio) > input_w:
            resized_w = input_w
        else:
            resized_w = int(math.ceil(input_h * ratio))
        resized_image = cv2.resize(image, (resized_w, input_h))
        if input_c == 1:
            resized_image = resized_image[np.newaxis, :]
        resized_image = resized_image.transpose((2, 0, 1))
        resized_image = resized_image.astype('float32')
        resized_image = resized_image / 255.0
        resized_image -= 0.5
        resized_image /= 0.5
        padded_image = np.zeros((input_c, input_h, input_w), dtype=np.float32)
        padded_image[:, :, 0:resized_w] = resized_image
        return padded_image

    def __call__(self, images):
        num_images = len(images)
        results = [['', 0.0]] * num_images
        indices = np.argsort(np.array([x.shape[1] / x.shape[0] for x in images]))
        batch_size = 6
        for i in range(0, num_images, batch_size):
            norm_images = []
            for j in range(i, min(num_images, i + batch_size)):
                norm_img = self.resize(images[indices[j]])
                norm_img = norm_img[np.newaxis, :]
                norm_images.append(norm_img)
            norm_images = np.concatenate(norm_images)
            outputs = self.session.run(None, {self.inputs.name: norm_images})[0]
            outputs = [(self.labels[idx], outputs[i, idx]) for i, idx in enumerate(outputs.argmax(axis=1))]
            for j in range(len(outputs)):
                label, score = outputs[j]
                results[indices[i + j]] = [label, score]
                if '180' in label and score > self.threshold:
                    images[indices[i + j]] = cv2.rotate(images[indices[i + j]], 1)
        return images, results

class POCR_Recognition:
    def __init__(self, onnx_path, session=None):
        self.session = session
        if self.session is None:
            assert onnx_path is not None
            assert os.path.exists(onnx_path)
            from onnxruntime import InferenceSession
            self.session = InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        self.inputs = self.session.get_inputs()[0]
        self.input_shape = [3, 48, 320]
        self.ctc_decoder = POCR_CTCDecoder()

    def resize(self, image, max_wh_ratio):
        input_h, input_w = self.input_shape[1], self.input_shape[2]
        assert self.input_shape[0] == image.shape[2]
        input_w = int((input_h * max_wh_ratio))
        w = self.inputs.shape[3:][0]
        if isinstance(w, str):
            pass
        elif w is not None and w > 0:
            input_w = w
        h, w = image.shape[:2]
        ratio = w / float(h)
        if math.ceil(input_h * ratio) > input_w:
            resized_w = input_w
        else:
            resized_w = int(math.ceil(input_h * ratio))
        resized_image = cv2.resize(image, (resized_w, input_h))
        resized_image = resized_image.transpose((2, 0, 1))
        resized_image = resized_image.astype('float32')
        resized_image = resized_image / 255.0
        resized_image -= 0.5
        resized_image /= 0.5
        padded_image = np.zeros((self.input_shape[0], input_h, input_w), dtype=np.float32)
        padded_image[:, :, 0:resized_w] = resized_image
        return padded_image

    def __call__(self, images):
        batch_size = 6
        num_images = len(images)
        results = [['', 0.0]] * num_images
        confidences = [['', 0.0]] * num_images
        indices = np.argsort(np.array([x.shape[1] / x.shape[0] for x in images]))
        for index in range(0, num_images, batch_size):
            input_h, input_w = self.input_shape[1], self.input_shape[2]
            max_wh_ratio = input_w / input_h
            norm_images = []
            for i in range(index, min(num_images, index + batch_size)):
                h, w = images[indices[i]].shape[0:2]
                max_wh_ratio = max(max_wh_ratio, w * 1.0 / h)
            for i in range(index, min(num_images, index + batch_size)):
                norm_image = self.resize(images[indices[i]], max_wh_ratio)
                norm_image = norm_image[np.newaxis, :]
                norm_images.append(norm_image)
            norm_images = np.concatenate(norm_images)
            outputs = self.session.run(None, {self.inputs.name: norm_images})
            result, confidence = self.ctc_decoder(outputs[0])
            for i in range(len(result)):
                results[indices[index + i]] = result[i]
                confidences[indices[index + i]] = confidence[i]
        return results, confidences

class POCR_CTCDecoder(object):
    def __init__(self):
        self.character = ['blank', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '<', '=', '>', '?',
                          '@', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O',
                          'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '[', '\\', ']', '^', '_',
                          '`', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p',
                          'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '{', '|', '}', '~', '!', '"', '#',
                          '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', ' ', ' ']

    def __call__(self, outputs):
        if isinstance(outputs, tuple) or isinstance(outputs, list):
            outputs = outputs[-1]
        indices = outputs.argmax(axis=2)
        return self.decode(indices, outputs)

    def decode(self, indices, outputs):
        results = []
        confidences = []
        ignored_tokens = [0]  # for ctc blank
        for i in range(len(indices)):
            selection = np.ones(len(indices[i]), dtype=bool)
            selection[1:] = indices[i][1:] != indices[i][:-1]
            for ignored_token in ignored_tokens:
                selection &= indices[i] != ignored_token
            result = []
            confidence = []
            for j in range(len(indices[i][selection])):
                result.append(self.character[indices[i][selection][j]])
                confidence.append(outputs[i][selection][j][indices[i][selection][j]])
            results.append(''.join(result))
            confidences.append(confidence)
        return results, confidences

def sort_polygon(points):
    points.sort(key=lambda x: (x[0][1], x[0][0]))
    for i in range(len(points) - 1):
        for j in range(i, -1, -1):
            if abs(points[j + 1][0][1] - points[j][0][1]) < 10 and (points[j + 1][0][0] < points[j][0][0]):
                temp = points[j]
                points[j] = points[j + 1]
                points[j + 1] = temp
            else:
                break
    return points

def crop_image(image, points):
    assert len(points) == 4, "shape of points must be 4*2"
    crop_width = int(max(np.linalg.norm(points[0] - points[1]),
                         np.linalg.norm(points[2] - points[3])))
    crop_height = int(max(np.linalg.norm(points[0] - points[3]),
                          np.linalg.norm(points[1] - points[2])))
    pts_std = np.float32([[0, 0],
                             [crop_width, 0],
                             [crop_width, crop_height],
                             [0, crop_height]])
    matrix = cv2.getPerspectiveTransform(points, pts_std)
    image = cv2.warpPerspective(image,
                                matrix, (crop_width, crop_height),
                                borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_CUBIC)
    height, width = image.shape[0:2]
    if height * 1.0 / width >= 1.5:
        image = np.rot90(image, k=3)
    return image

def x1y1wh_to_x1y1x2y2_and_padding(bbox, padding):
    x, y, w, h = bbox
    x_min, y_min = x - padding, y - padding
    x_max, y_max = x + w + padding, y + h + padding
    return ((x_min, y_min), (x_max, y_max))

# ====================================================================================================

pocr_det = POCR_Detection('lib/PaddleOCR/model/det.onnx')
# pocr_rec = POCR_Recognition('lib/PaddleOCR/model/rec.onnx')
# pocr_cls = POCR_Classification('lib/PaddleOCR/model/cls.onnx')

def Process_PaddleOCR(img_path, padding=1, debug_dot=False):

    if debug_dot == True:
        print(".", end="")

    img = cv2.imread(img_path)
    obboxs = list(pocr_det(img)) # sort_polygon(list(pocr_det(img)))
    # cropped_images, _angles = pocr_cls([crop_image(img, x) for x in obboxs])
    # ocr_txts, _confidences = pocr_rec(cropped_images)
    ocr_bboxes = [x1y1wh_to_x1y1x2y2_and_padding(bb, padding=padding) for bb in [cv2.boundingRect(obb) for obb in obboxs]]
    
    # # Visualize
    # for i, ((x1, y1), (x2, y2)) in enumerate(ocr_bboxes):
    #     cv2.polylines(img, [np.array(obboxs[i], dtype=np.int32)], True, (255, 0, 0), 2)
    #     cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    #     # cv2.putText(img, ocr_txts[i], (int(x1), int(y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1, cv2.LINE_AA)
    # cv2.imwrite('output.jpg', img)

    return ocr_bboxes