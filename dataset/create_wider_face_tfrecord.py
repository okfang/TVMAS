from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib2
import tensorflow as tf
import numpy
import cv2
import os
import hashlib
import dataset_util
import functools

train_image_dir = './WIDER_train/images'
eval_image_dir = './WIDER_val/images'
train_annot_path = './wider_face_split/wider_face_train_bbx_gt.txt'
eval_annot_path = './wider_face_split/wider_face_val_bbx_gt.txt'
train_output_path = './tfrecord_data/train/face_train.record'
eval_output_path = './tfrecord_data/eval/wider_face/face_val.record'
train_image_num = 12880
eval_image_num = 3326


def create_tf_example(f,image_path=None):
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    filename = f.readline().rstrip()
    filepath = os.path.join(image_path, filename)
    image_raw = cv2.imread(filepath)

    encoded_image_data = open(filepath, 'rb').read()
    key = hashlib.sha256(encoded_image_data).hexdigest()

    height, width, channel = image_raw.shape

    face_num = int(f.readline().rstrip())
    valid_face_num = 0

    for i in range(face_num):
        annot = f.readline().rstrip().split()
        # WIDER FACE DATASET CONTAINS SOME ANNOTATIONS WHAT EXCEEDS THE IMAGE BOUNDARY
        if (float(annot[2]) > 25.0):
            if (float(annot[3]) > 30.0):
                xmins.append(max(0.005, (float(annot[0]) / width)))
                ymins.append(max(0.005, (float(annot[1]) / height)))
                xmaxs.append(min(0.995, ((float(annot[0]) + float(annot[2])) / width)))
                ymaxs.append(min(0.995, ((float(annot[1]) + float(annot[3])) / height)))
                classes_text.append('face'.encode('utf8'))
                classes.append(1)
                print(xmins[-1], ymins[-1], xmaxs[-1], ymaxs[-1], classes_text[-1], classes[-1])
                valid_face_num += 1;

    print("Face Number is %d" % face_num)
    print("Valid face number is %d" % valid_face_num)

    feature_dict = {
        'image/height': dataset_util.int64_feature(int(height)),
        'image/width': dataset_util.int64_feature(int(width)),
        'image/filename': dataset_util.bytes_feature(filename.encode('utf8')),
        # 'image/source_id': dataset_util.bytes_feature(filename.encode('utf8')),
        # 'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded_image_data),
        'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }
    tf_example = tf.train.Example(features=tf.train.Features(feature=feature_dict))

    return valid_face_num, tf_example


def create_wider_face_tf_record(annot_path, output_path, image_num,image_path,num_shards):
    with contextlib2.ExitStack() as tf_record_close_stack:
        tf_record_output_filenames = [
            '{}-{:05d}-of-{:05d}'.format(output_path, idx, num_shards)
            for idx in range(num_shards)
        ]
        output_tfrecords = [
            tf_record_close_stack.enter_context(tf.python_io.TFRecordWriter(file_name))
            for file_name in tf_record_output_filenames
        ]

        # WIDER FACE DATASET ANNOTATED 12880 IMAGES
        f = open(annot_path)
        valid_image_num = 0
        invalid_image_num = 0
        target_create_tf_example = functools.partial(create_tf_example,image_path = image_path)
        for image_idx in range(image_num):
            print("image idx is %d" % image_idx)
            valid_face_number, tf_example = target_create_tf_example(f)
            shards_idx = image_idx % num_shards
            if (valid_face_number != 0):
                output_tfrecords[shards_idx].write(tf_example.SerializeToString())
                valid_image_num += 1
            else:
                invalid_image_num += 1
                print("Pass!")
        tf.logging.info('Finished writing, skipped %d annotations.')
        print("Valid image number is %d" % valid_image_num)
        print("Invalid image number is %d" % invalid_image_num)


if __name__ == '__main__':
    create_wider_face_tf_record(train_annot_path,train_output_path,train_image_num,train_image_dir,num_shards=100)
    create_wider_face_tf_record(eval_annot_path,eval_output_path,eval_image_num,eval_image_dir,num_shards=20)
