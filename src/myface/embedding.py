from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import facenet
from detect_align import load_mtcnn
from img_loader import *

def load_pretrain_model():
    facenet.load_model('./models/facenet/20170512-110547')
    images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
    embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
    phase_train_placeholder = tf.get_default_graph().get_tensor_by_name(
        "phase_train:0")
    return (images_placeholder, embeddings, phase_train_placeholder)

def calc_embeddings(model, sess, imgs, image_size=160):
    images_placeholder = model[0]
    embeddings = model[1]
    phase_train_placeholder = model[2]
    nrof_images = len(imgs)
    images = np.zeros((nrof_images, image_size, image_size, 3))
    for i in range(nrof_images):
        images[i, :, :, :] = imgs[i]
    feed_dict = {images_placeholder: images, phase_train_placeholder: False}
    emb_array = sess.run(embeddings, feed_dict=feed_dict)
    return emb_array


def calc_distance(embeddings1, embeddings2):
    diff = np.subtract(embeddings1, embeddings2)
    return np.sum(np.square(diff), 1)
    

if __name__ == '__main__':
    mtcnn = load_mtcnn()
    img1 = load_img(mtcnn, '../face_test_img/bbh/1.jpg')
    img2 = load_img(mtcnn, '../face_test_img/bbh/2.jpg')

    with tf.Graph().as_default():
        with tf.Session() as sess:
            model = load_pretrain_model()
            embeddings = calc_embeddings(model, sess, [img1, img2])
            dist = calc_distance([embeddings[0]], [embeddings[1]])
            print('distance', dist[0])