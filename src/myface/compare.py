# -*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import time
import json
import redis
import tensorflow as tf
from detect_align import load_mtcnn, detect_align
from embedding import load_pretrain_model, calc_distance, calc_embeddings

def main():
    REDIS_EXPIRE = 10

    mtcnn = load_mtcnn()
    with tf.Graph().as_default():
        with tf.Session() as sess:
            model = load_pretrain_model()

            r = redis.Redis('localhost', 6379, password='Byte20171116@')

            while (True):
                print('wait to compare face!')
                filename = r.brpop('face_compare')
                orig_filename = bytes.decode(filename[1])
                print('origfilename', orig_filename)
                filename = orig_filename.split('#')

                filename1 = filename[0]
                filename2 = filename[1]
                threshold = float(filename[2])

                # print('split', filename, filename1, filename2)

                imgpath1 = os.path.join('../uploads', filename1)
                imgpath2 = os.path.join('../uploads', filename2)
                # print('imgpath', imgpath1, imgpath2)

                start = time.time()
                rect1= {'left': 0, 'top': 0, 'right': 0, 'bottom': 0}
                size1 = {'width':0, 'height':0}
                img1 = detect_align(mtcnn, imgpath1, '../uploads/align', False,
                                    160, True, rect1, size1)
                rect2 = {'left': 0, 'top': 0, 'right': 0, 'bottom': 0}
                size2 = {'width':0, 'height':0}
                img2 = detect_align(mtcnn, imgpath2, '../uploads/align', False, 160, True, rect2, size2)
                if (img1 is None or img2 is None):
                    ret = {'status': -4, 'msg': 'detect face error'}
                else:
                    embeddings = calc_embeddings(model, sess, [img1, img2])
                    dist = calc_distance([embeddings[0]], [embeddings[1]])[0]
                    judge_time = time.time() - start
                    # print('judge time:　', judge_time)
                    if (dist <= threshold):
                        msg = "同一人可能性高"
                    # elif dist <= 1.1:
                    #     msg = "同一人可能性一般"
                    # elif dist <= 1.23:
                    #     msg = "不同人可能性一般"
                    else:
                        msg = "不同人可能性高"
                    ret = {
                        'status': 0,
                        'msg': msg,
                        'distance': dist.item(),
                        'time': judge_time,
                        'size1': size1,
                        'rect1': rect1,
                        'size2': size2,
                        'rect2': rect2
                    }

                r.expire(orig_filename, REDIS_EXPIRE)
                print(ret)
                r.lpush(orig_filename, json.dumps(ret))


if __name__ == '__main__':
    sys.path.append('../')
    main()
