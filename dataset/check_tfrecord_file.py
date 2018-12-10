import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as patches


filename = "/data/kingdom/fangdx/TVMAS/dataset/tfrecord_data/train/face_train.record-00000-of-00100"

def get_example_from_tfrecord(filename_queue):
    reader = tf.TFRecordReader()
    _,serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'image/encoded':
                                               tf.FixedLenFeature((), tf.string, default_value=''),
                                           'image/height':
                                               tf.FixedLenFeature((), tf.int64, default_value=1),
                                           'image/width':
                                               tf.FixedLenFeature((), tf.int64, default_value=1),
                                           # Object boxes and classes.
                                           'image/object/bbox/xmin':
                                               tf.VarLenFeature(tf.float32),
                                           'image/object/bbox/xmax':
                                               tf.VarLenFeature(tf.float32),
                                           'image/object/bbox/ymin':
                                               tf.VarLenFeature(tf.float32),
                                           'image/object/bbox/ymax':
                                               tf.VarLenFeature(tf.float32),
                                           'image/object/class/label':
                                               tf.VarLenFeature(tf.int64),
                                           'image/object/class/text':
                                               tf.VarLenFeature(tf.string)
                                       })
    image = tf.image.decode_jpeg(features["image/encoded"])
    xmin = tf.sparse_tensor_to_dense(features["image/object/bbox/xmin"])
    ymin = tf.sparse_tensor_to_dense(features["image/object/bbox/ymin"])
    xmax = tf.sparse_tensor_to_dense(features["image/object/bbox/xmax"])
    ymax = tf.sparse_tensor_to_dense(features["image/object/bbox/ymax"])
    boxes = tf.stack([xmin,ymin,xmax,ymax],axis=1)
    width = features['image/width']
    height = features['image/height']

    return image,boxes,width,height




if __name__ == '__main__':
    filename_queue = tf.train.string_input_producer([filename])
    image,boxes,width,height = get_example_from_tfrecord(filename_queue)
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        for i in range(2):
            _image,_boxes, _width, _height = sess.run(
                [image,boxes,width,height])
            print(type(_image),type(_boxes),type(_width),type(_height))
            coord.request_stop()
            coord.join(threads)

            plt.imshow(_image)
            ax = plt.gca()
            for box in _boxes:
                _xmin, _xmax = box[0] * _width, box[2] * _width
                _ymin, _ymax = box[1] * _height, box[3] * _height

                rect = patches.Rectangle([_xmin, _ymin], _xmax - _xmin, _ymax - _ymin,linewidth=1,edgecolor='r',facecolor='none')
                ax.add_patch(rect)
            plt.show()