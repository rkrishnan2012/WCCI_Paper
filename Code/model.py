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

LOG_DIR = './log'

CHECKPOINT_FILE = './inception_v4.ckpt'

IMAGE_SIZE = 299

NUM_CLASSES = 101

TFRECORD_FILE_PATTERN = 'foods_%s_*.tfrecord'

STEPS_PER_EPOCH = 100

NUM_EPOCHS = 300

IMAGES_PER_GPU = 8

GPU_COUNT = 1

BATCH_SIZE = IMAGES_PER_GPU * GPU_COUNT

LEARNING_RATE = 0.002

MOMENTUM = 0.9

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
        common_queue_capacity = 24 + 3 * batch_size,
        common_queue_min = 24)

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
    #Create the log directory here. Must be done here otherwise import will activate this unneededly.
    if not os.path.exists(LOG_DIR):
        os.mkdir(LOG_DIR)

    #======================= TRAINING PROCESS =========================
    with tf.Graph().as_default() as graph:
        tf.logging.set_verbosity(tf.logging.INFO) #Set the verbosity to INFO level

        # First create the dataset and load one batch
        dataset = get_split('train', DATASET_DIR, file_pattern=TFRECORD_FILE_PATTERN)
        train_images, _, train_labels = load_batch(dataset, batch_size=BATCH_SIZE)

        # Perform one-hot-encoding of the labels (Try one-hot-encoding within the load_batch function!)
        one_hot_labels = slim.one_hot_encoding(train_labels, dataset.num_classes)

        # Create the training model and the validation model (which doesn't have dropout)
        with slim.arg_scope(inception_arg_scope()):
            logits, end_points = inception_v4(train_images, num_classes = dataset.num_classes, is_training = True)

        # Define the scopes that you want to exclude for restoration
        exclude = ['InceptionV4/Logits', 'InceptionV4/AuxLogits']
        variables_to_restore = slim.get_variables_to_restore(exclude = exclude)

        # Performs the equivalent to tf.nn.sparse_softmax_cross_entropy_with_logits but enhanced with checks
        loss = tf.losses.softmax_cross_entropy(onehot_labels = one_hot_labels, logits = logits)
        total_loss = tf.losses.get_total_loss()    # obtain the regularization losses as well

        # Now we can define the optimizer that takes on the learning rate
        optimizer = tf.train.MomentumOptimizer(learning_rate=LEARNING_RATE, momentum=MOMENTUM)

        # Create the train_op.
        train_op = slim.learning.create_train_op(total_loss, optimizer)

        # State the metrics that you want to predict. We get a predictions that is not one_hot_encoded.
        predictions = tf.argmax(end_points['Predictions'], 1)
        probabilities = end_points['Predictions']

        accuracy, accuracy_update = tf.contrib.metrics.streaming_accuracy(predictions, train_labels)
        
        metrics_op = tf.group(accuracy_update, probabilities)

        #Now finally create all the summaries you need to monitor and group them into one summary op.
        tf.summary.scalar('losses/Total_Loss', total_loss)
        tf.summary.scalar('accuracy', accuracy)
        tf.summary.scalar('learning_rate', LEARNING_RATE)
        my_summary_op = tf.summary.merge_all()

        #Now we need to create a training step function that runs both the train_op, metrics_op and updates the global_step concurrently.
        def train_step(sess, train_op, global_step):
            '''
            Simply runs a session for the three arguments provided and gives a logging on the time elapsed for each global step
            '''
            #Check the time for each sess run
            start_time = time.time()
            total_loss, global_step_count, _ = sess.run([train_op, global_step, metrics_op])
            time_elapsed = time.time() - start_time

            #Run the logging to print some results
            logging.info('global step %s: loss: %.4f (%.2f sec/step)', global_step_count, total_loss, time_elapsed)

            return total_loss, global_step_count

        #Now we create a saver function that actually restores the variables from a checkpoint file in a sess
        saver = tf.train.Saver(variables_to_restore)
        def restore_fn(sess):
            return saver.restore(sess, checkpoint_file)

        #Define your supervisor for running a managed session. Do not run the summary_op automatically or else it will consume too much memory
        sv = tf.train.Supervisor(logdir = LOG_DIR, summary_op = None, init_fn = restore_fn)

        #Run the managed session
        with sv.managed_session() as sess:
            for step in range(STEPS_PER_EPOCH * NUM_EPOCHS):
                #At the start of every epoch, show the vital information:
                if step % STEPS_PER_EPOCH == 0:
                    logging.info('Epoch %s/%s', step/STEPS_PER_EPOCH + 1, NUM_EPOCHS)
                    accuracy_value = sess.run([accuracy])
                    logging.info('Current Learning Rate: %s', LEARNING_RATE)
                    logging.info('Current Streaming Accuracy: %s', accuracy_value)

                    # optionally, print your logits and predictions for a sanity check that things are going fine.
                    logits_value, probabilities_value, predictions_value, labels_value = sess.run([logits, probabilities, predictions, train_labels])
                    print('logits: \n', logits_value)
                    print('Probabilities: \n', probabilities_value)
                    print('predictions: \n', predictions_value)
                    print('Labels:\n:', labels_value)

                #Log the summaries every 10 step.
                if step % 10 == 0:
                    loss, _ = train_step(sess, train_op, sv.global_step)
                    summaries = sess.run(my_summary_op)
                    sv.summary_computed(sess, summaries)
                    
                #If not, simply run the training step
                else:
                    loss, _ = train_step(sess, train_op, sv.global_step)

            # We log the final training loss and accuracy
            logging.info('Final Loss: %s', loss)
            logging.info('Final Accuracy: %s', sess.run(accuracy))

            # Once all the training has been done, save the log files and checkpoint model
            logging.info('Finished training! Saving model to disk now.')
            sv.saver.save(sess, sv.save_path, global_step = sv.global_step)


if __name__ == '__main__':
    run()