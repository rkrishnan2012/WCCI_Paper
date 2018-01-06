# The code in this file will be used to train our Inception-V4 based network.
# 
# Note: The inception_v4 architecture diagram can be found here: http://yeephycho.github.io/blog_img/Inception_v4_hires.jpg
#
# Goals:
#   * Train a Inception_V4 network using the Food-101 dataset - incomplete
#
import tensorflow as tf
from tensorflow.python.platform import tf_logging as logging
import inception_preprocessing
from inception_v4 import inception_v4
from inception_utils import inception_arg_scope
from dataset_utils import read_label_file
import os
import time
slim = tf.contrib.slim
get_or_create_global_step = tf.train.get_or_create_global_step

# =============== CONFIGURATION ===============
DATASET_DIR = '../Dataset/food-101/images'

VAL_LOG_DIR = './validation_log'

LOG_DIR = './log'

IMAGE_SIZE = 299

NUM_CLASSES = 101

TFRECORD_FILE_PATTERN = 'foods_%s_*.tfrecord'

IMAGES_PER_GPU = 8

GPU_COUNT = 1

BATCH_SIZE = IMAGES_PER_GPU * GPU_COUNT

VALIDATION_STEPS = 50

VALIDATION_INTERVAL_SECS = 30


#============== DATASET LOADING ======================
def get_split(split_name, dataset_dir, file_pattern=TFRECORD_FILE_PATTERN, file_pattern_for_counting='foods'):
    '''
    Obtains the split - training or validation - to create a Dataset class for feeding the examples into a queue later on. This function will
    set up the decoder and dataset information all into one Dataset class so that you can avoid the brute work later on.
    Your file_pattern is very important in locating the files later. 
    INPUTS:
    - split_name(str): 'train' or 'validation'. Used to get the correct data split of tfrecord files
    - dataset_dir(str): the dataset directory where the tfrecord files are located
    - file_pattern(str): the file name structure of the tfrecord files in order to get the correct data
    - file_pattern_for_counting(str): the string name to identify your tfrecord files for counting
    OUTPUTS:
    - dataset (Dataset): A Dataset class object where we can read its various components for easier batch creation later.
    '''

    # First check whether the split_name is train or validation
    if split_name not in ['train', 'validation']:
        raise ValueError('The split_name %s is not recognized. Please input either train or validation as the split_name' % (split_name))

    # Create the full path for a general file_pattern to locate the tfrecord_files
    file_pattern_path = os.path.join(dataset_dir, file_pattern % (split_name))

    # Count the total number of examples in all of these shard
    num_samples = 0
    file_pattern_for_counting = file_pattern_for_counting + '_' + split_name
    tfrecords_to_count = [os.path.join(dataset_dir, file) for file in os.listdir(dataset_dir) if file.startswith(file_pattern_for_counting)]
    for tfrecord_file in tfrecords_to_count:
        for record in tf.python_io.tf_record_iterator(tfrecord_file):
            num_samples += 1

    # Create a reader, which must be a TFRecord reader in this case
    reader = tf.TFRecordReader

    # Create the keys_to_features dictionary for the decoder
    keys_to_features = {
      'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
      'image/format': tf.FixedLenFeature((), tf.string, default_value='jpg'),
      'image/class/label': tf.FixedLenFeature(
          [], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
    }

    # Create the items_to_handlers dictionary for the decoder.
    items_to_handlers = {
    'image': slim.tfexample_decoder.Image(),
    'label': slim.tfexample_decoder.Tensor('image/class/label'),
    }

    # Create the items_to_descriptions dictionary for the decoder.
    items_to_descriptions = {
        'image': 'A 3-channel RGB coloured flower image that is either tulips, sunflowers, roses, dandelion, or daisy.',
        'label': 'A label that is as such -- 0:daisy, 1:dandelion, 2:roses, 3:sunflowers, 4:tulips'
    }

    #Start to create the decoder
    decoder = slim.tfexample_decoder.TFExampleDecoder(keys_to_features, items_to_handlers)

    #Create the labels_to_name file
    labels_to_name_dict = read_label_file(dataset_dir)

    #Actually create the dataset
    dataset = slim.dataset.Dataset(
        data_sources = file_pattern_path,
        decoder = decoder,
        reader = reader,
        num_readers = 4,
        num_samples = num_samples,
        num_classes = NUM_CLASSES,
        labels_to_name = labels_to_name_dict,
        items_to_descriptions=items_to_descriptions)

    return dataset


def load_batch(dataset, batch_size, height=IMAGE_SIZE, width=IMAGE_SIZE, is_training=True):
    '''
    Loads a batch for training.
    INPUTS:
    - dataset(Dataset): a Dataset class object that is created from the get_split function
    - batch_size(int): determines how big of a batch to train
    - height(int): the height of the image to resize to during preprocessing
    - width(int): the width of the image to resize to during preprocessing
    - is_training(bool): to determine whether to perform a training or evaluation preprocessing
    OUTPUTS:
    - images(Tensor): a Tensor of the shape (batch_size, height, width, channels) that contain one batch of images
    - labels(Tensor): the batch's labels with the shape (batch_size,) (requires one_hot_encoding).
    '''
    #First create the data_provider object
    data_provider = slim.dataset_data_provider.DatasetDataProvider(
        dataset,
        common_queue_capacity = 50 * batch_size,
        common_queue_min = 3 * batch_size)

    #Obtain the raw image using the get method
    raw_image, label = data_provider.get(['image', 'label'])

    #Perform the correct preprocessing for this image depending if it is training or evaluating
    image = inception_preprocessing.preprocess_image(raw_image, height, width, is_training)

    #As for the raw images, we just do a simple reshape to batch it up
    raw_image = tf.expand_dims(raw_image, 0)
    raw_image = tf.image.resize_nearest_neighbor(raw_image, [height, width])
    raw_image = tf.squeeze(raw_image)

    #Batch up the image by enqueing the tensors internally in a FIFO queue and dequeueing many elements with tf.train.batch.
    images, raw_images, labels = tf.train.batch(
        [image, raw_image, label],
        batch_size = batch_size,
        num_threads = 4,
        capacity = 4 * batch_size,
        allow_smaller_final_batch = True)

    return images, raw_images, labels

def run():
    if not os.path.exists(VAL_LOG_DIR):
        os.mkdir(VAL_LOG_DIR)    

    #======================= TRAINING PROCESS =========================
    with tf.Graph().as_default():

        tf.logging.set_verbosity(tf.logging.INFO)

        tf_global_step = slim.get_or_create_global_step()
        
        dataset = get_split('validation', DATASET_DIR, file_pattern=TFRECORD_FILE_PATTERN)
        val_images, raw_val_images, val_labels = load_batch(dataset, batch_size=BATCH_SIZE)

        labels_to_name_dict = read_label_file(DATASET_DIR)

        # Create the training model and the validation model (which doesn't have dropout)
        with slim.arg_scope(inception_arg_scope()):
            logits, end_points = inception_v4(val_images, num_classes = dataset.num_classes, is_training=True)

        variables_to_restore = slim.get_variables_to_restore()

        predictions = tf.argmax(end_points['Predictions'], 1)
        
        # Define the metrics:
        names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
            'Accuracy': slim.metrics.streaming_accuracy(predictions, val_labels),
        })

        # Print the summaries to screen.
        for name, value in names_to_values.items():
            summary_name = '%s_Validation' % name
            op = tf.summary.scalar(summary_name, value, collections=[])
            op = tf.Print(op, [value], summary_name)
            tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)

        checkpoint_path = tf.train.latest_checkpoint(LOG_DIR)
        tf.logging.info('Evaluating %s' % checkpoint_path)

        tf.contrib.training.evaluate_repeatedly(
            checkpoint_dir=LOG_DIR,
            master='',
            hooks=[
                tf.contrib.training.StopAfterNEvalsHook(VALIDATION_STEPS),
                tf.contrib.training.SummaryAtEndHook(VAL_LOG_DIR),
            ],
            eval_ops = [names_to_updates['Accuracy']]
        )

if __name__ == '__main__':
    run()