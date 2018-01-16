from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from detect_align import load_mtcnn, detect_align


def load_imgs(mtcnn, src_dir, return_dict = True):
    imgs = []
    dirs = os.listdir(src_dir)
    for person in dirs:
        person_dir = os.path.join(src_dir, person)
        if (not os.path.isdir(person_dir)):
            continue

        files = os.listdir(person_dir)
        for file in files:
            img_file = os.path.join(person_dir, file)
            print('load_imgs() img_file: ', img_file)
            if return_dict:
                imgs.append({
                    'name': person,
                    'img_file': img_file,
                    'img': detect_align(mtcnn, img_file)
                })
            else:
                imgs.append(detect_align(mtcnn, img_file,'../test_data/test_img/align'))
    return imgs


def load_img(mtcnn, img_file):
    return {
        'name': os.path.split(os.path.split(img_file)[0])[1],
        'img_file': img_file,
        'img': detect_align(mtcnn, img_file)
    }


if __name__ == '__main__':
    mtcnn = load_mtcnn()
    imgs = load_imgs(mtcnn, '../test_data/test_img/raw', False)
    # print(imgs)
