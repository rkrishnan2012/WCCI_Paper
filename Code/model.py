"""
The code in this file will be used to train our Inception-Resnet-V2 based network.

Note: The inception_resnet_v2 architecture diagram can be found here:
https://1.bp.blogspot.com/-O7AznVGY9js/V8cV_wKKsMI/AAAAAAAABKQ/maO7n2w3dT4Pkcmk7wgGqiSX5FUW2sfZgCLcB/s1600/image00.png
"""
import functools
import itertools
import os
import six

import tensorflow as tf

import dataset_utils
import utils
from inception_resnet_v2 import inception_resnet_v2

tf.logging.set_verbosity(tf.logging.INFO)

# =============== CONFIGURATION ===============
DATASET_DIR = '../Dataset/food-101/images'

LOG_DIR = './rmsprop'

IMAGE_SIZE = 299

NUM_CLASSES = 101

TFRECORD_FILE_PATTERN = 'foods_%s_*.tfrecord'

FILE_PATTERN_FOR_COUNTING = 'foods'

IMAGES_PER_GPU = 8

GPU_COUNT = 2

BATCH_SIZE = IMAGES_PER_GPU * GPU_COUNT

LEARNING_RATE = 0.045

DECAY = 0.9

VALIDATION_STEPS = 50

STEPS_PER_EPOCH = 101000 / BATCH_SIZE

VARIABLE_STRATEGY = 'GPU'

WEIGHT_DECAY = 2e-4

LEARNING_RATE_DECAY_FACTOR = 0.94

def tower_fn(is_training, feature, label, data_format):
    """Build computation tower
    Args:
        is_training: true if is training graph.
        feature: a Tensor.
        label: a Tensor.
        data_format: Not implemented yet, but change the inception_resnet model
                to support channels_last (NHWC) or channels_first (NCHW).
    Returns:
        A tuple with the loss for the tower, the gradients and parameters, and
        predictions.
    """
    logits, endpoints = inception_resnet_v2(feature, num_classes=NUM_CLASSES,
                                    is_training=is_training)

    tower_pred = {
        'classes': tf.argmax(input=logits, axis=1),
        'probabilities': tf.nn.softmax(logits)
    }

    tower_loss = tf.losses.sparse_softmax_cross_entropy(logits=logits, labels=label)
    
    aux_tower_loss = 0.4 * tf.losses.sparse_softmax_cross_entropy(logits=logits,
                                 labels=label,
                                 scope='aux_loss')

    tower_loss = tf.reduce_mean(tower_loss + aux_tower_loss)

    model_params = tf.trainable_variables()
    tower_loss += WEIGHT_DECAY * tf.add_n(
        [tf.nn.l2_loss(v) for v in model_params])

    tower_grad = tf.gradients(tower_loss, model_params)

    return tower_loss, zip(tower_grad, model_params), tower_pred


def model_fn(features, labels, mode, params):
    """Inception_Resnet_V2 model body.
    Support single host, one or more GPU training. Parameter distribution can
    be either one of the following scheme.
    1. CPU is the parameter server and manages gradient updates.
    2. Parameters are distributed evenly across all GPUs, and the first GPU
       manages gradient updates.
    Args:
      features: a list of tensors, one for each tower
      labels: a list of tensors, one for each tower
      mode: ModeKeys.TRAIN or EVAL
      params: Hyperparameters suitable for tuning
    Returns:
      A EstimatorSpec object.
    """
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    tower_features = features
    tower_labels = labels
    tower_losses = []
    tower_gradvars = []
    tower_preds = []

    # channels first (NCHW) is normally optimal on GPU and channels last (NHWC)
    # on CPU. The exception is Intel MKL on CPU which is optimal with
    # channels_last.
    data_format = None
    if not data_format:
        if GPU_COUNT == 0:
            data_format = 'channels_last'
        else:
            data_format = 'channels_first'

    if GPU_COUNT == 0:
        num_devices = 1
        device_type = 'cpu'
    else:
        num_devices = GPU_COUNT
        device_type = 'gpu'

    for i in range(num_devices):
        worker_device = '/{}:{}'.format(device_type, i)
        if VARIABLE_STRATEGY == 'CPU':
            device_setter = utils.local_device_setter(worker_device=worker_device)
        elif VARIABLE_STRATEGY == 'GPU':
            device_setter = utils.local_device_setter(
                ps_device_type='gpu',
                worker_device=worker_device,
                ps_strategy=tf.contrib.training.GreedyLoadBalancingStrategy(
                    GPU_COUNT, tf.contrib.training.byte_size_load_fn))
        with tf.variable_scope('InceptionResnetV2', reuse=bool(i != 0)):
            with tf.name_scope('tower_%d' % i) as name_scope:
                with tf.device(device_setter):
                    loss, gradvars, preds = tower_fn(is_training, tower_features[i],
                                                     tower_labels[i], data_format)
                    tower_losses.append(loss)
                    tower_gradvars.append(gradvars)
                    tower_preds.append(preds)
                    if i == 0:
                        # Only trigger batch_norm moving mean and variance update from
                        # the 1st tower. Ideally, we should grab the updates from all
                        # towers but these stats accumulate extremely fast so we can
                        # ignore the other stats from the other towers without
                        # significant detriment.
                        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS,
                                                       name_scope)  

    # Now compute global loss and gradients.
    gradvars = []
    with tf.name_scope('gradient_ing'):
        all_grads = {}
        for grad, var in itertools.chain(*tower_gradvars):
            if grad is not None:
                all_grads.setdefault(var, []).append(grad)
        for var, grads in six.iteritems(all_grads):
            # Average gradients on the same device as the variables
            # to which they apply.
            with tf.device(var.device):
                if len(grads) == 1:
                    avg_grad = grads[0]
                else:
                    avg_grad = tf.multiply(tf.add_n(grads), 1. / len(grads))
            gradvars.append((avg_grad, var))

    # Device that runs the ops to apply global gradient updates.
    consolidation_device = '/gpu:0' if VARIABLE_STRATEGY == 'GPU' else '/cpu:0'
    with tf.device(consolidation_device):
        loss = tf.reduce_mean(tower_losses, name='loss')

        examples_sec_hook = utils.ExamplesPerSecondHook(BATCH_SIZE, every_n_steps=10)

        # Define your exponentially decaying learning rate
        learning_rate = tf.train.exponential_decay(
            learning_rate = LEARNING_RATE,
            global_step = tf.train.get_global_step(),
            decay_steps = STEPS_PER_EPOCH * 2,
            decay_rate = LEARNING_RATE_DECAY_FACTOR)

        tensors_to_log = {'learning_rate': learning_rate, 'loss': loss}

        logging_hook = tf.train.LoggingTensorHook(
            tensors=tensors_to_log, every_n_iter=100)

        train_hooks = [logging_hook, examples_sec_hook]

        optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate, decay=DECAY)

        # Create single grouped train op
        train_op = [
            optimizer.apply_gradients(
                gradvars, global_step=tf.train.get_global_step())
        ]
        train_op.extend(update_ops)
        train_op = tf.group(*train_op)

        predictions = {
            'classes':
                tf.concat([p['classes'] for p in tower_preds], axis=0),
            'probabilities':
                tf.concat([p['probabilities'] for p in tower_preds], axis=0)
        }
        stacked_labels = tf.concat(labels, axis=0)
        metrics = {
            'accuracy':
                tf.metrics.accuracy(stacked_labels, predictions['classes'])
        }

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op,
        training_hooks=train_hooks,
        eval_metric_ops=metrics)

def input_fn(split_name, is_training):
    """Create input graph for model.
    Args:
      split_name: one of 'train', 'validate' and 'eval'.
    Returns:
      two lists of tensors for features and labels, each of GPU_COUNT length.
    """
    with tf.device('/cpu:0'):
        dataset = dataset_utils.get_split(split_name, DATASET_DIR, NUM_CLASSES,
                                          TFRECORD_FILE_PATTERN, FILE_PATTERN_FOR_COUNTING)
        image_batch, _, label_batch = dataset_utils.load_batch(dataset, \
            BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, is_training)
        if GPU_COUNT <= 1:
            # No GPU available or only 1 GPU.
            return [image_batch], [label_batch]

        # Note that passing num=batch_size is safe here, even though
        # dataset.batch(batch_size) can, in some cases, return fewer than batch_size
        # examples. This is because it does so only when repeating for a limited
        # number of epochs, but our dataset repeats forever.
        image_batch = tf.unstack(image_batch, num=BATCH_SIZE, axis=0)
        label_batch = tf.unstack(label_batch, num=BATCH_SIZE, axis=0)
        feature_shards = [[] for i in range(GPU_COUNT)]
        label_shards = [[] for i in range(GPU_COUNT)]
        for i in range(BATCH_SIZE):
            idx = i % GPU_COUNT
            feature_shards[idx].append(image_batch[i])
            label_shards[idx].append(label_batch[i])
        feature_shards = [tf.parallel_stack(x) for x in feature_shards]
        label_shards = [tf.parallel_stack(x) for x in label_shards]
        return feature_shards, label_shards

def experiment_fn(run_config, hparams):
    """
    This is a method passed to tf.contrib.learn.learn_runner that will
    return an instance of an Experiment.
    """

    train_input_fn = functools.partial(
        input_fn,
        split_name='train',
        is_training=True)

    eval_input_fn = functools.partial(
        input_fn,
        split_name='validation',
        is_training=False)

    classifier = tf.estimator.Estimator(
        model_fn=model_fn,
        config=run_config,
        params=hparams)

    return tf.contrib.learn.Experiment(
        classifier,
        train_input_fn=train_input_fn,
        eval_input_fn=eval_input_fn,
        train_steps=None, # Train forever
        eval_steps=VALIDATION_STEPS)



def train():
    """
    Begins training the entire architecture.
    """
    # Session configuration.
    sess_config = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False,
        intra_op_parallelism_threads=0, # Autocompute how many threads to run
        gpu_options=tf.GPUOptions(force_gpu_compatible=True))

    config = tf.contrib.learn.RunConfig(session_config=sess_config, model_dir=LOG_DIR)
    tf.contrib.learn.learn_runner.run(
        experiment_fn,
        run_config=config,
        hparams=tf.contrib.training.HParams(is_chief=config.is_chief))


if __name__ == '__main__':
    # A (supposed) 5% percent boost in certain GPUs by using faster convolution operations
    os.environ['TF_SYNC_ON_FINISH'] = '0'
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
    train()
