from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from scipy import misc
import tensorflow as tf
import numpy as np
import facenet
import align.detect_face
# import dlib
from skimage import transform, draw


def load_mtcnn():
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.25)
        sess = tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)
            return (pnet, rnet, onet)


# LEFT_EYE_INDICES = [36, 37, 38, 39, 40, 41]
# RIGHT_EYE_INDICES = [42, 43, 44, 45, 46, 47]

# def rect_to_tuple(rect):
#     left = rect.left()
#     right = rect.right()
#     top = rect.top()
#     bottom = rect.bottom()
#     return left, top, right, bottom


# def extract_eye(shape, eye_indices):
#     points = map(lambda i: shape.part(i), eye_indices)
#     return list(points)

# def extract_eye_center(shape, eye_indices):
#     points = extract_eye(shape, eye_indices)
#     xs = map(lambda p: p.x, points)
#     ys = map(lambda p: p.y, points)
#     return sum(xs) // 6, sum(ys) // 6

# def extract_left_eye_center(shape):
#     return extract_eye_center(shape, LEFT_EYE_INDICES)


# def extract_right_eye_center(shape):
#     return extract_eye_center(shape, RIGHT_EYE_INDICES)


def angle_between_2_points(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    tan = (y2 - y1) / (x2 - x1)
    return np.degrees(np.arctan(tan))


# def calc_intesect_area(rect1, rect2):
#     if (rect1[0] >= rect2[2] or rect2[0] >= rect1[2]):
#         return 0

#     if (rect1[1] >= rect2[3] or rect2[1] >= rect1[3]):
#         return 0

#     l = max(rect1[0], rect2[0])
#     r = min(rect1[2], rect2[2])
#     t = max(rect1[1], rect2[1])
#     b = min(rect1[3], rect2[3])

#     return (r - l) * (b - t)


# def get_align_angle(img, face_rect):
#     detector = dlib.get_frontal_face_detector()
#     predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

#     dets = detector(img)

#     max_area = 0
#     target_index = -1
#     for i, det in enumerate(dets):
#         area = calc_intesect_area(face_rect, rect_to_tuple(det))
#         if (area > max_area):
#             max_area = area
#             target_index = i

#     if (target_index == -1):
#         return 0

#     shape = predictor(img, dets[target_index])
#     left_eye = extract_left_eye_center(shape)
#     right_eye = extract_right_eye_center(shape)
#     return angle_between_2_points(left_eye, right_eye)


def draw_rect(img, rect, color):
    left = int(rect[0])
    right = int(rect[2])
    top = int(rect[1])
    bottom = int(rect[3])

    r1, c1 = draw.line(left, top, right, top)
    r2, c2 = draw.line(left, top, left, bottom)
    r3, c3 = draw.line(right, bottom, right, top)
    r4, c4 = draw.line(right, bottom, left, bottom)
    draw.set_color(img, [c1, r1], color)
    draw.set_color(img, [c2, r2], color)
    draw.set_color(img, [c3, r3], color)
    draw.set_color(img, [c4, r4], color)


def detect(mtcnn, image_path, dst_dir, image_size=500):
    print('detect', image_path)
    pnet = mtcnn[0]
    rnet = mtcnn[1]
    onet = mtcnn[2]

    minsize = 20
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor
    try:
        img = misc.imread(image_path)
    except (IOError, ValueError, IndexError) as e:
        errorMessage = '{}: {}'.format(image_path, e)
        print(errorMessage)
        return {'count': -1}
    else:
        if img.ndim < 2:
            print('Unable to detect "%s"' % image_path)
            return {'count': 0}

        if img.ndim == 2:
            img = facenet.to_rgb(img)
        img = img[:, :, 0:3]

        bounding_boxes, _ = align.detect_face.detect_face(
            img, minsize, pnet, rnet, onet, threshold, factor)

        nrof_faces = bounding_boxes.shape[0]
        if nrof_faces > 0:
            det = bounding_boxes[:, 0:4]
            det_arr = []
            if nrof_faces > 1:
                for i in range(nrof_faces):
                    det_arr.append(np.squeeze(det[i]))
            else:
                det_arr.append(np.squeeze(det))

            for i, det in enumerate(det_arr):
                det = np.squeeze(det)
                left = int(det[0])
                right = int(det[2])
                top = int(det[1])
                bottom = int(det[3])
                # print('l,t,r,b: ', left, top, right, bottom)
                draw_rect(img, det, [255, 0, 0])

            rects = det_arr
            if (image_size > 0):
                mwh = max(img.shape[0], img.shape[1])
                ratio = image_size / mwh
                h = int(img.shape[0] * ratio)
                w = int(img.shape[1] * ratio)
                img = misc.imresize(img, (h, w), interp='bilinear')
                for i, det in enumerate(det_arr):
                    det = np.squeeze(det)
                    left = det[0] * ratio
                    right = det[2] * ratio
                    top = det[1] * ratio
                    bottom = det[3] * ratio
                    rects[i] = [left, top, right, bottom]

            if not (dst_dir is None):
                if not os.path.isdir(dst_dir):
                    os.makedirs(dst_dir)
                (_, filename) = os.path.split(image_path)
                # (filename, extname) = os.path.splitext(filename)
                output_filename = os.path.join(dst_dir, filename)
                misc.imsave(output_filename, img)

            return {'count': len(det_arr), 'rects': det_arr, 'size': {'width': w, 'height': h}}
        else:
            print('Unable to detect "%s"' % image_path)
            return {'count': 0}


def detect_align(mtcnn,
                 image_path,
                 save_dir=None,
                 detect_multiple_faces=False,
                 image_size=160,
                 whiten=True,
                 rect=None,
                 size=None):
    # print(image_path)
    pnet = mtcnn[0]
    rnet = mtcnn[1]
    onet = mtcnn[2]

    minsize = 20
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor
    try:
        img = misc.imread(image_path)
    except (IOError, ValueError, IndexError) as e:
        errorMessage = '{}: {}'.format(image_path, e)
        print(errorMessage)
        return None
    else:
        if img.ndim < 2:
            print('Unable to align "%s"' % image_path)
            return None

        if img.ndim == 2:
            img = facenet.to_rgb(img)
        img = img[:, :, 0:3]

        bounding_boxes, landmarks = align.detect_face.detect_face(
            img, minsize, pnet, rnet, onet, threshold, factor)
        print('landmarks', landmarks)
        nrof_faces = bounding_boxes.shape[0]
        if nrof_faces > 0:
            det = bounding_boxes[:, 0:4]
            det_arr = []
            img_size = np.asarray(img.shape)[0:2]
            if isinstance(size, dict):
                size['width'] = int(img_size[1])
                size['height'] = int(img_size[0])

            if nrof_faces > 1:
                if detect_multiple_faces:
                    for i in range(nrof_faces):
                        det_arr.append(np.squeeze(det[i]))
                else:
                    # bounding_box_size = (det[:, 2] - det[:, 0]) * (
                    #     det[:, 3] - det[:, 1])
                    # img_center = img_size / 2
                    # offsets = np.vstack(
                    #     [(det[:, 0] + det[:, 2]) / 2 - img_center[1],
                    #      (det[:, 1] + det[:, 3]) / 2 - img_center[0]])
                    # offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
                    # index = np.argmax(
                    #     bounding_box_size - offset_dist_squared * 2.0
                    # )  # some extra weight on the centering
                    # det_arr.append(det[index, :])

                    index = np.argmax(bounding_boxes[:, 4])
                    det_arr.append(det[index, :])
                    landmark = landmarks[:, index]
            else:
                det_arr.append(np.squeeze(det))
                landmark = np.squeeze(landmarks)

            print('landmark', landmark)
            det = np.squeeze(det_arr[0])
            left = det[0]
            right = det[2]
            top = det[1]
            bottom = det[3]
            if isinstance(rect, dict):
                rect['left'] = int(left)
                rect['right'] = int(right)
                rect['top'] = int(top)
                rect['bottom'] = int(bottom)

            dw = right - left
            dh = bottom - top

            target_edge = min(img_size[0], img_size[1])

            margin_left_right = (target_edge - dw) / 2
            margin_top_bottom = (target_edge - dh) / 2

            target_left = left - margin_left_right
            target_right = right + margin_left_right
            target_top = top - margin_top_bottom
            target_bottom = bottom + margin_top_bottom

            if (target_left < 0):
                target_left = 0
                target_right = target_edge
            elif (target_right > img_size[1]):
                target_left = img_size[1] - target_edge
                target_right = img_size[1]
                left = left - (img_size[1] - target_edge)
                right = right - (img_size[1] - target_edge)
            else:
                left = margin_left_right
                right = target_edge - margin_left_right

            if (target_top < 0):
                target_top = 0
                target_bottom = target_edge
            elif (target_bottom > img_size[0]):
                target_top = img_size[0] - target_edge
                target_bottom = img_size[0]
                top = top - (img_size[0] - target_edge)
                bottom = bottom - (img_size[0] - target_edge)
            else:
                top = margin_top_bottom
                bottom = target_edge - margin_top_bottom

            target_left = int(target_left)
            target_top = int(target_top)
            target_right = int(target_right)
            target_bottom = int(target_bottom)

            left = int(left)
            right = int(right)
            top = int(top)
            bottom = int(bottom)

            dw = right - left
            dh = bottom - top
            margin = int(32 * max(dw, dh) / 128)
            cropped = img[target_top:target_bottom, target_left:
                          target_right, :]

            bb = np.zeros(4, dtype=np.int32)
            bb[0] = np.maximum(left - margin / 2, 0)
            bb[1] = np.maximum(top - margin / 2, 0)
            bb[2] = np.minimum(right + margin / 2, target_edge)
            bb[3] = np.minimum(bottom + margin / 2, target_edge)

            cropped = cropped[bb[1]:bb[3], bb[0]:bb[2], :]
            left = 0
            top = 0
            right = bb[2] - bb[0]
            bottom = bb[3] - bb[1]
            # angle = angle_between_2_points((landmark[0], landmark[5]),(landmark[1], landmark[6]))
            # if (angle > 0 or angle < 0):
            #     print("angle: ", angle)
            #     cen = ((left + right) / 2, (top + bottom) / 2)
            #     cropped = transform.rotate(cropped, angle, center=cen)

            scaled = misc.imresize(
                cropped, (image_size, image_size), interp='bilinear')

            if not (save_dir is None):
                if not os.path.isdir(save_dir):
                    os.makedirs(save_dir)
                (_, filename) = os.path.split(image_path)
                # (filename, extname) = os.path.splitext(filename)
                output_filename = os.path.join(save_dir, filename)
                misc.imsave(output_filename, scaled)

            if whiten:
                return prewhiten(scaled)
            else:
                return scaled
        else:
            print('Unable to align "%s"' % image_path)
            return None

def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0 / np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1 / std_adj)
    return y


if __name__ == '__main__':
    mtcnn = load_mtcnn()
    # detect(mtcnn,
    #        '../test_data/test_img/detect_test/test.jpg',
    #        '../test_data/test_img/detect_test/detect', 0)
    # detect_align(mtcnn, '../test_data/test_img/origin/langdingfan/langdingfan_0001.jpg', '../test_data/test_img/align')
    rect = {'left': 0, 'top': 0, 'right': 0, 'bottom': 0}
    size = {'width':0, 'height':0}
    detect_align(
        mtcnn, 'D:/workspace/byte/FaceDetect/uploads/1513058426124-458502.jpg',
        'D:/workspace/byte/FaceDetect/uploads/align', False, 160, True, rect,size)
    print('size, rect', size, rect)
