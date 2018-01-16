import os
import random
import math
from detect_align import load_mtcnn, detect_align
from scipy import misc


def modify_filename(src_dir):
    dirs = os.listdir(src_dir)
    for person in dirs:
        person_dir = os.path.join(src_dir, person)
        if (not os.path.isdir(person_dir)):
            continue

        files = os.listdir(person_dir)
        index = 0
        for file in files:
            index += 1
            _, extname = os.path.splitext(file)
            img_file = os.path.join(person_dir, file)
            new_file = "%s_%04d%s" % (person, index, extname)
            new_file = os.path.join(person_dir, new_file)
            os.rename(img_file, new_file)


def resize(img_path, dst_path):
    try:
        img = misc.imread(img_path)
    except (IOError, ValueError, IndexError) as e:
        errorMessage = '{}: {}'.format(img_path, e)
        print(errorMessage)
        return None
    else:
        s = 250 / img.shape[1]
        height = int(img.shape[0] * s)
        scaled = misc.imresize(img, (height,250), interp='bilinear')
        misc.imsave(dst_path, scaled)


def resize_all(src_dir, dst_dir):
    if (not os.path.isdir(dst_dir)):
        os.makedirs(dst_dir)

    dirs = os.listdir(src_dir)
    for person in dirs:
        person_dir = os.path.join(src_dir, person)
        if (not os.path.isdir(person_dir)):
            continue
        dst_person_dir = os.path.join(dst_dir, person)
        if (not os.path.isdir(dst_person_dir)):
            os.mkdir(dst_person_dir)

        files = os.listdir(person_dir)
        for file in files:
            img_file = os.path.join(person_dir, file)
            resize(img_file, os.path.join(dst_person_dir, file))

def to_lfw(src_dir, dst_dir):
    if (not os.path.isdir(dst_dir)):
        os.makedirs(dst_dir)

    mtcnn = load_mtcnn()
    dirs = os.listdir(src_dir)
    for person in dirs:
        person_dir = os.path.join(src_dir, person)
        if (not os.path.isdir(person_dir)):
            continue
        dst_person_dir = os.path.join(dst_dir, person)
        if (not os.path.isdir(dst_person_dir)):
            os.mkdir(dst_person_dir)

        files = os.listdir(person_dir)
        for file in files:
            img_file = os.path.join(person_dir, file)
            detect_align(mtcnn, img_file, dst_person_dir)


def load_imgs(src_dir):
    imgs = []
    dirs = os.listdir(src_dir)
    for person in dirs:
        person_dir = os.path.join(src_dir, person)
        if (not os.path.isdir(person_dir)):
            continue
        person_imgs = []
        files = os.listdir(person_dir)
        for file in files:
            (filename, _) = os.path.splitext(file)
            num = int(filename.split('_')[-1])
            person_imgs.append(num)
        imgs.append({'name': person, 'imgs': person_imgs})

    return imgs

def gen_all_same_pair(imgs):
    pairs = []
    for _, person_img in enumerate(imgs):
        count = len(person_img['imgs'])
        for i in range(count):
            for j in range(i+1, count):
                pairs.append({
                    'name':
                    person_img['name'],
                    'imgs': [person_img['imgs'][i], person_img['imgs'][j]]
                })
    random.shuffle(pairs)
    return pairs


def gen_all_diff_pair(imgs):
    pairs = []
    person_count = len(imgs)
    for i in range(person_count):
        for j in range(i+1, person_count):
            p1 = imgs[i]
            p2 = imgs[j]
            l1 = len(p1['imgs'])
            l2 = len(p2['imgs'])
            for k in range(l1):
                for l in range(l2):
                    pairs.append([{
                        'name': p1['name'],
                        'img': p1['imgs'][k]
                    }, {
                        'name': p2['name'],
                        'img': p2['imgs'][l]
                    }])
    random.shuffle(pairs)
    return pairs


def gen_pairs(src_dir, dst_dir, batchs=10, batch_size=100):
    imgs = load_imgs(src_dir)
    all_sames = gen_all_same_pair(imgs)
    all_diffs = gen_all_diff_pair(imgs)

    print('person count: ', len(imgs))
    print('same pairs: ', len(all_sames))
    print('diffirent pairs: ', len(all_diffs))

    count = batchs * batch_size
    if (len(all_sames) < count or len(all_diffs) < count):
        print('Not enough pairs, generate pairs.txt failed!')
        return

    output_pairs(all_sames[:count], all_diffs[:count], dst_dir, batchs, batch_size)


def output_pairs(sames, diffs, dst_dir, batchs, batch_size):
    fp = open(os.path.join(dst_dir, "pairs.txt"), "w")
    fp.write("%d\t%d\n" %(batchs, batch_size))
    for b in range(batchs):
        for i in range(batch_size):
            s = sames[batch_size*b + i]
            fp.write("%s\t%d\t%d\n" % (s['name'], s['imgs'][0], s['imgs'][1]))
        for i in range(batch_size):
            s = diffs[batch_size * b + i]
            fp.write("%s\t%d\t%s\t%d\n" % (s[0]['name'], s[0]['img'], s[1]['name'], s[1]['img']))
    fp.close()


if __name__ == '__main__':
    # modify_filename('../face_test_img')
    to_lfw('./datasets/lfw/raw', './datasets/lfw/align')
    # to_lfw('../test_data/face_test_img', '../test_data/face_test_img_flw/imgs')
    # gen_pairs('../face_test_img_flw/imgs', '../face_test_img_flw',10,100)
    # modify_filename('../test_img')
    # to_lfw('../test_data/test_img/origin', '../test_data/test_img/origin_flw')
    # gen_pairs('../test_data/test_img/origin', '../test_data/test_img/origin', 2, 7)
    # resize_all('../test_img', '../test_img_resize')
