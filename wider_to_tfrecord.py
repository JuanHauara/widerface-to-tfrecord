import tensorflow as tf
import numpy
import cv2
import os
import hashlib
import argparse

from utils import dataset_util


parser = argparse.ArgumentParser(description = 'Convert official WIDER FACE dataset (http://shuoyang1213.me/WIDERFACE) to tfrecord.')
parser.add_argument('--dataset_path', type = str, help = "WIDER FACE dataset path with 'WIDER_train', 'WIDER_val', 'WIDER_test' and 'wider_face_split' dirs.")
parser.add_argument('--limit_train', type = int, default = 0, help = 'Limit the amount of images in the train dataset.')
parser.add_argument('--limit_validation', type = int, default = 0, help = 'Limit the amount of images in the validation dataset.')
parser.add_argument('--limit_test', type = int, default = 0, help = 'Limit the amount of images in the test dataset.')
args = parser.parse_args()


def parse_test_example(f, images_path):
    height = None # Image height
    width = None # Image width
    filename = None # Filename of the image. Empty if image is not from file
    encoded_image_data = None # Encoded image bytes
    image_format = b'jpeg' # b'jpeg' or b'png'

    filename = f.readline().rstrip()
    if not filename:
        raise IOError()

    filepath = os.path.join(images_path, filename)

    image_raw = cv2.imread(filepath)

    encoded_image_data = open(filepath, "rb").read()
    key = hashlib.sha256(encoded_image_data).hexdigest()

    height, width, channel = image_raw.shape

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(int(height)),
        'image/width': dataset_util.int64_feature(int(width)),
        'image/filename': dataset_util.bytes_feature(filename.encode('utf-8')),
        'image/source_id': dataset_util.bytes_feature(filename.encode('utf-8')),
        'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded_image_data),
        'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
        }))

    return tf_example


def parse_example(f, images_path):
    height = None # Image height
    width = None # Image width
    filename = None # Filename of the image. Empty if image is not from file
    encoded_image_data = None # Encoded image bytes
    image_format = b'jpeg' # b'jpeg' or b'png'

    xmins = [] # List of normalized left x coordinates in bounding box (1 per box)
    xmaxs = [] # List of normalized right x coordinates in bounding box (1 per box)
    ymins = [] # List of normalized top y coordinates in bounding box (1 per box)
    ymaxs = [] # List of normalized bottom y coordinates in bounding box (1 per box)
    classes_text = [] # List of string class name of bounding box (1 per box)
    classes = [] # List of integer class id of bounding box (1 per box)
    poses = []
    truncated = []
    difficult_obj = []

    filename = f.readline().rstrip()
    if not filename:
        raise IOError()

    filepath = os.path.join(images_path, filename)

    image_raw = cv2.imread(filepath)

    encoded_image_data = open(filepath, "rb").read()
    key = hashlib.sha256(encoded_image_data).hexdigest()

    height, width, channel = image_raw.shape

    face_num = int(f.readline().rstrip())

    if face_num > 0:
        for i in range(face_num):
            annot = f.readline().rstrip().split()
            if not annot:
                raise Exception()

            # WIDER FACE DATASET CONTAINS SOME ANNOTATIONS WHAT EXCEEDS THE IMAGE BOUNDARY
            if(float(annot[2]) > 25.0):
                if(float(annot[3]) > 30.0):
                    xmins.append( max(0.005, (float(annot[0]) / width) ) )
                    ymins.append( max(0.005, (float(annot[1]) / height) ) )
                    xmaxs.append( min(0.995, ((float(annot[0]) + float(annot[2])) / width) ) )
                    ymaxs.append( min(0.995, ((float(annot[1]) + float(annot[3])) / height) ) )
                    classes_text.append(b'face')
                    classes.append(1)
                    poses.append("front".encode('utf8'))
                    truncated.append(int(0))

        tf_example = tf.train.Example(features=tf.train.Features(feature={
            'image/height': dataset_util.int64_feature(int(height)),
            'image/width': dataset_util.int64_feature(int(width)),
            'image/filename': dataset_util.bytes_feature(filename.encode('utf-8')),
            'image/source_id': dataset_util.bytes_feature(filename.encode('utf-8')),
            'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
            'image/encoded': dataset_util.bytes_feature(encoded_image_data),
            'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
            'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
            'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
            'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
            'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
            'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
            'image/object/class/label': dataset_util.int64_list_feature(classes),
            'image/object/difficult': dataset_util.int64_list_feature(int(0)),
            'image/object/truncated': dataset_util.int64_list_feature(truncated),
            'image/object/view': dataset_util.bytes_list_feature(poses),
            }))

        return tf_example
    else:
        _ = f.readline().rstrip()
        print("\n> ERROR in: filepath = {} | face_num = {} | This image won't be added to the dataset".format(filepath, face_num))
        return None


def run(images_path, description_file, output_path, no_bbox, limit = 0):
    f = open(description_file)
    writer = tf.python_io.TFRecordWriter(output_path)

    if limit > 0:
        print('>> Dataset limited to {} images.'.format(limit))
    print(">> Processing {}".format(images_path), end = '')
    cont = 0
    while True:
        try:
            if no_bbox:
                tf_example = parse_test_example(f, images_path)
            else:
                tf_example = parse_example(f, images_path)

                if tf_example != None:
                    writer.write(tf_example.SerializeToString())
                    cont += 1
                    print('.', end = '')

                if limit > 0 and cont >= limit:
                    break
        except IOError:
            break
        except Exception:
            print('Exception: {}'.format(Exception))
            raise

    writer.close()

    print("\n>> Done!\n>> Correctly created tfrecord for {} images.\n".format(cont))


def main(unused_argv):
    ground_truth_path = os.path.join(os.getcwd(), args.dataset_path, 'wider_face_split')

    # Training
    description_file = os.path.join(os.getcwd(), args.dataset_path, 'wider_face_split', 'wider_face_train_bbx_gt.txt')
    wider_path = os.path.join(os.getcwd(), args.dataset_path, 'WIDER_train') 
    images_path  = os.path.join(os.getcwd(), args.dataset_path, 'WIDER_train', 'images') 

    if os.path.exists(ground_truth_path) and os.path.exists(description_file) and os.path.exists(wider_path) and os.path.exists(images_path):
        output_path = os.path.join(os.getcwd(), 'tfrecord-output', "train.tfrecord")
        run(images_path, description_file, output_path, False, args.limit_train)
    else:
        print('>> Train dataset skipped.')

    # Validation
    description_file = os.path.join(os.getcwd(), args.dataset_path, 'wider_face_split', 'wider_face_val_bbx_gt.txt')
    wider_path = os.path.join(os.getcwd(), args.dataset_path, 'WIDER_val') 
    images_path  = os.path.join(os.getcwd(), args.dataset_path, 'WIDER_val', 'images') 

    if os.path.exists(ground_truth_path) and os.path.exists(description_file) and os.path.exists(wider_path) and os.path.exists(images_path):
        output_path = os.path.join(os.getcwd(), 'tfrecord-output', "validation.tfrecord")
        run(images_path, description_file, output_path, False, args.limit_validation)
    else:
        print('>> Validation dataset skipped.')

    # Testing. This set does not contain bounding boxes, so the tfrecord will contain images only
    description_file = os.path.join(os.getcwd(), args.dataset_path, 'wider_face_split', 'wider_face_test_filelist.txt')
    wider_path = os.path.join(os.getcwd(), args.dataset_path, 'WIDER_test') 
    images_path  = os.path.join(os.getcwd(), args.dataset_path, 'WIDER_tes', 'images') 

    if os.path.exists(ground_truth_path) and os.path.exists(description_file) and os.path.exists(wider_path) and os.path.exists(images_path):
        output_path = os.path.join(os.getcwd(), 'tfrecord-output', "test.tfrecord")
        run(images_path, description_file, output_path, True, args.limit_test)
    else:
        print('>> Test dataset skipped.')


if __name__ == '__main__':
    tf.app.run()
