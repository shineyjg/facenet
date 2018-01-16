# -*- coding:utf-8 -*-

import os
import sys
import time
import json
import redis
from detect_align import load_mtcnn, detect

def main():
    REDIS_EXPIRE = 10

    mtcnn = load_mtcnn()

    r = redis.Redis('localhost', 6379, password='Byte20171116@')

    while (True):
        print('wait to detect face!')
        filename = r.brpop('face_detect')
        filename = bytes.decode(filename[1])
        # print('filename', filename)

        imgpath = os.path.join('../uploads', filename)
        # print('imgpath', imgpath)

        start = time.time()
        faces = detect(mtcnn, imgpath, '../uploads/detect')
        print('detect result: ', faces)
        count = faces['count']
        if count < 0:
            ret = {'status': -4, 'msg': 'detect face error'}
        else:
            ret = {
                'status': 0,
                'msg': 'success',
                'faces': count,
                'time': time.time() - start,
                'detectUrl':'/detect/'+filename,
                'rects':[]
            }
            if (count > 0):
                ret['size'] = faces['size']
                for _,v in enumerate(faces['rects']):
                    ret['rects'].append({
                        'left': int(v[0]),
                        'top': int(v[1]),
                        'right': int(v[2]),
                        'bottom': int(v[3])
                    })
        r.expire(filename, REDIS_EXPIRE)
        print(ret)
        r.lpush(filename, json.dumps(ret))


if __name__ == '__main__':
    sys.path.append('../')
    main()
