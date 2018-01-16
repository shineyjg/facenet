from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import math
import random
import tensorflow as tf
import numpy as np
from img_loader import load_imgs
from detect_align import load_mtcnn
from embedding import load_pretrain_model, calc_distance, calc_embeddings
import lfw
import facenet
from sklearn import metrics
from scipy.optimize import brentq
from scipy import interpolate


def calc_accuracy(imgs, embeddings, threshold, show_fail=False):
    accurency = 0
    img_count = len(embeddings)
    for i in range(img_count):
        for j in range(i + 1, img_count):
            dist = calc_distance([embeddings[i]], [embeddings[j]])[0]
            res = dist < threshold
            if (imgs[i]['name'] == imgs[j]['name'] and res):
                accurency += 1
            elif (not (imgs[i]['name'] == imgs[j]['name']) and not res):
                accurency += 1
            else:
                if (show_fail):
                    print('fail: ', imgs[i]['img_file'], imgs[j]['img_file'],
                          res, dist)

    return accurency * 2 / (img_count * (img_count - 1))


def find_best_threshold(imgs, embeddings, thresholds):
    accuracy = np.zeros_like(thresholds)
    for i, threshold in enumerate(thresholds):
        accuracy[i] = calc_accuracy(imgs, embeddings, threshold)
        # print('test threshold: ', threshold, ', accuracy: ', accuracy[i])

    best_threshold_index = np.argmax(accuracy)
    return [thresholds[best_threshold_index], accuracy[best_threshold_index]]


def threshold(img_dir):
    t1 = time.time()
    mtcnn = load_mtcnn()
    t2 = time.time()
    print('mtcnn loaded, time: %.2f' % (t2 - t1))

    imgs = load_imgs(mtcnn, img_dir)
    random.shuffle(imgs)
    t1 = time.time()
    print('images loaded, count: %d, time: %.2f' %(len(imgs), t1 - t2))

    batch_count = 10
    batch_size = math.floor(len(imgs) / 10)

    thresholds = np.arange(0.3, 1.2, 0.01)
    threshold_accuracies = []
    with tf.Graph().as_default():
        with tf.Session() as sess:
            t1 = time.time()
            model = load_pretrain_model()
            t2 = time.time()
            print('pretrained model loaded, time: %.2f' % (t2 - t1))
            embeddings = calc_embeddings(model, sess, imgs)
            for i in range(batch_count - 1):
                t1 = time.time()
                ta = find_best_threshold(
                    imgs[batch_size * i:batch_size * (i + 1)],
                    embeddings[batch_size * i:batch_size * (i + 1)],
                    thresholds)
                t2 = time.time()
                threshold_accuracies.append(ta)
                print('batch: %d, threshold %f, accuracy: %f, time: %.2f' % (i, ta[0], ta[1], t2 - t1))

            ta = np.mean(threshold_accuracies, 0)
            threshold = ta[0]
            accuracy = ta[1]
            print('mean threshold: %f, mean accuracy: %f' % (threshold,
                                                             accuracy))

            #test
            accuracy = calc_accuracy(imgs[batch_size * 9:],
                                     embeddings[batch_size * 9:], threshold)

            print('test accuracy: ', accuracy)


def threshold2(img_dir):
    t1 = time.time()
    mtcnn = load_mtcnn()
    t2 = time.time()
    print('mtcnn loaded, time: %.2f' % (t2 - t1))

    imgs = load_imgs(mtcnn, img_dir, False)
    t1 = time.time()
    print('images loaded, count: %d, time: %.2f' % (len(imgs), t1 - t2))

    with tf.Graph().as_default():
        with tf.Session() as sess:
            t1 = time.time()
            model = load_pretrain_model()
            t2 = time.time()
            print('pretrained model loaded, time: %.2f' % (t2 - t1))
            embeddings = calc_embeddings(model, sess, imgs)
            diff = np.subtract(embeddings[0::2], embeddings[1::2])
            dist = np.sum(np.square(diff), 1)
            print('dist same person', dist)

            em1 = [embeddings[0], embeddings[1]]
            em2 = [embeddings[2], embeddings[3]]
            diff = np.subtract(em1, em2)
            dist = np.sum(np.square(diff), 1)
            print('dist diff person', dist)

def threshold3(img_dir,
               pairs_file,
               batch_size=100,
               image_size=160,
               lfw_nrof_folds=10,
               ext = 'jpg'):
    pairs = lfw.read_pairs(pairs_file)

    # Get the paths for the corresponding images
    paths, actual_issame = lfw.get_paths(img_dir, pairs, ext)

    with tf.Graph().as_default():
        with tf.Session() as sess:
            facenet.load_model('./models/facenet/20170512-110547')

            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name(
                "input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name(
                "embeddings:0")
            phase_train_placeholder = tf.get_default_graph(
            ).get_tensor_by_name("phase_train:0")

            embedding_size = embeddings.get_shape()[1]

            nrof_images = len(paths)
            nrof_batches = int(math.ceil(1.0 * nrof_images / batch_size))
            emb_array = np.zeros((nrof_images, embedding_size))
            for i in range(nrof_batches):
                start_index = i * batch_size
                end_index = min((i + 1) * batch_size, nrof_images)
                paths_batch = paths[start_index:end_index]
                images = facenet.load_data(paths_batch, False, False,
                                           image_size)
                feed_dict = {
                    images_placeholder: images,
                    phase_train_placeholder: False
                }
                emb_array[start_index:end_index, :] = sess.run(
                    embeddings, feed_dict=feed_dict)

            # print('emb_array size: ', emb_array.shape, ', same length: ', len(actual_issame))

            tpr, fpr, accuracy, val, val_std, far = lfw.evaluate(
                emb_array, actual_issame, nrof_folds=lfw_nrof_folds)

            print('Accuracy: %1.3f+-%1.3f' % (np.mean(accuracy),
                                              np.std(accuracy)))
            print('Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val, val_std,
                                                                 far))

            auc = metrics.auc(fpr, tpr)
            print('Area Under Curve (AUC): %1.3f' % auc)
            eer = brentq(lambda x: 1. - x - interpolate.interp1d(fpr, tpr)(x),
                         0., 1.)
            print('Equal Error Rate (EER): %1.3f' % eer)


if __name__ == '__main__':
    # threshold('../test_img_flw/imgs')
    # threshold2('../test_data/test_img/raw')
    threshold3('../test_data/face_test_img_flw/imgs',
               '../test_data/face_test_img_flw/pairs.txt', 100, 160, 10)
    threshold3('../test_data/face_test_img_flw/imgs_aligned',
               '../test_data/face_test_img_flw/pairs.txt', 100, 160, 10)
    # threshold3('./datasets/lfw/lfw_mtcnnpy_160', './data/pairs.txt', 100, 160,
    #            10, 'png')
