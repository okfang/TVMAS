import PIL.Image
import hashlib
import os

import io

import contextlib2
import tensorflow as tf

import dataset_util

image_dir = './ccpd_dataset/ccpd_base'
train_output_path = './tfrecord_data/train/ccpd_train.record'
eval_output_path = './tfrecord_data/eval/ccpd/ccpd_val.record'

def create_tf_example(filename):
    coordinates = filename.rsplit('/', 1)[-1].rsplit('.', 1)[0].split('-')[2]
    leftUp, rightDown = [[int(eel) for eel in el.split('&')] for el in coordinates.split('_')]
    xmin,ymin = leftUp
    xmax,ymax = rightDown

    with tf.gfile.GFile(filename, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = PIL.Image.open(encoded_jpg_io)
    height = image.height
    width = image.width
    key = hashlib.sha256(encoded_jpg).hexdigest()

    ymins = [float(ymin)/height]
    xmins = [float(xmin)/width]
    ymaxs = [float(ymax)/height]
    xmaxs = [float(xmax)/width]

    labels_text = ['vehicle plate'.encode('utf8')]
    labels = [2]

    # print("---------image size:",image.size)
    # print("---------xmin:{}, ymin:{}, xmax:{}, ymax:{}".format(xmin,ymin,xmax,ymax))
    # print("---------width:{}, height:{}".format(width,height))

    feature_dict = {
        'image/height': dataset_util.int64_feature(int(height)),
        'image/width': dataset_util.int64_feature(int(width)),
        'image/filename': dataset_util.bytes_feature(filename.encode('utf8')),
        # 'image/source_id': dataset_util.bytes_feature(filename.encode('utf8')),
        # 'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(labels_text),
        'image/object/class/label': dataset_util.int64_list_feature(labels),
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    return example

def create_ccpd_tf_record(image_dir,num_shards):
    """create ccpd tf-record  train 10k  eval 10k"""

    with contextlib2.ExitStack() as tf_record_close_stack:
        trian_tf_record_output_filenames = [
            '{}-{:05d}-of-{:05d}'.format(train_output_path, idx, num_shards)
            for idx in range(num_shards)
        ]
        train_output_tfrecords = [
            tf_record_close_stack.enter_context(tf.python_io.TFRecordWriter(file_name))
            for file_name in trian_tf_record_output_filenames
        ]
        val_tf_record_output_filenames = [
            '{}-{:05d}-of-{:05d}'.format(eval_output_path, idx, num_shards)
            for idx in range(num_shards)
        ]
        val_output_tfrecords = [
            tf_record_close_stack.enter_context(tf.python_io.TFRecordWriter(file_name))
            for file_name in val_tf_record_output_filenames
        ]
        images = os.listdir(image_dir)
        for index,image_name in enumerate(images):
            # create train record
            if index < 100000:
                if index % 1000 == 0:
                    tf.logging.info('On image %d of %d', index, len(images))
                filename = os.path.join(image_dir,image_name)
                tf_example = create_tf_example(filename)
                shards_idx = index%num_shards
                train_output_tfrecords[shards_idx].write(tf_example.SerializeToString())
            # create val record
            else:
                if index % 1000 == 0:
                    tf.logging.info('On image %d of %d', index, len(images))
                filename = os.path.join(image_dir,image_name)
                tf_example = create_tf_example(filename)
                shards_idx = index%num_shards
                val_output_tfrecords[shards_idx].write(tf_example.SerializeToString())
            tf.logging.info('Finished writing')

if __name__ == '__main__':
    create_ccpd_tf_record(image_dir,num_shards=500)
