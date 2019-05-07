import numpy as np
import tensorflow as tf

from config import *
from nets import *

from dataset import vkitti
from sqdet_utils.util import sparse_to_dense

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'checkpoint', './data/model_checkpoints/squeezeDet/model.ckpt-87000',
    """Path to the model parameter file.""")
tf.app.flags.DEFINE_string('data_path', '', """Root directory of data""")
tf.app.flags.DEFINE_string('image_set', 'test',
                           """Only used for VOC data."""
                           """Can be train, trainval, val, or test""")
tf.app.flags.DEFINE_string(
    'demo_net', 'squeezeDet', """Neural net architecture.""")
tf.app.flags.DEFINE_string(
    'loss_types', 'all', """comma sep. string of losses (classification, confidence, bbox); e.g. confidence,bbox""")

def load_data(imdb, model, mc):
    # read batch input
    image_per_batch, label_per_batch, box_delta_per_batch, aidx_per_batch, \
    bbox_per_batch, batch_idx = imdb.read_batch(return_batch_idx=True) #TODO: Modify to return batch idxs

    label_indices, bbox_indices, box_delta_values, mask_indices, box_values, \
        = [], [], [], [], []
    aidx_set = set()
    num_discarded_labels = 0
    num_labels = 0
    for i in range(len(label_per_batch)):  # batch_size
        for j in range(len(label_per_batch[i])):  # number of annotations
            num_labels += 1
            if (i, aidx_per_batch[i][j]) not in aidx_set:
                aidx_set.add((i, aidx_per_batch[i][j]))
                label_indices.append(
                    [i, aidx_per_batch[i][j], label_per_batch[i][j]])
                mask_indices.append([i, aidx_per_batch[i][j]])
                bbox_indices.extend(
                    [[i, aidx_per_batch[i][j], k] for k in range(4)])
                box_delta_values.extend(box_delta_per_batch[i][j])
                box_values.extend(bbox_per_batch[i][j])
            else:
                num_discarded_labels += 1

    if mc.DEBUG_MODE:
        print ('Warning: Discarded {}/({}) labels that are assigned to the same '
               'anchor'.format(num_discarded_labels, num_labels))


    image_input = model.image_input
    input_mask = model.input_mask
    box_delta_input = model.box_delta_input
    box_input = model.box_input
    labels = model.labels

    feed_dict = {
        image_input: image_per_batch,
        input_mask: np.reshape(
            sparse_to_dense(
                mask_indices, [mc.BATCH_SIZE, mc.ANCHORS],
                [1.0] * len(mask_indices)),
            [mc.BATCH_SIZE, mc.ANCHORS, 1]),
        box_delta_input: sparse_to_dense(
            bbox_indices, [mc.BATCH_SIZE, mc.ANCHORS, 4],
            box_delta_values),
        box_input: sparse_to_dense(
            bbox_indices, [mc.BATCH_SIZE, mc.ANCHORS, 4],
            box_values),
        labels: sparse_to_dense(
            label_indices,
            [mc.BATCH_SIZE, mc.ANCHORS, mc.CLASSES],
            [1.0] * len(label_indices)),
    }

    return feed_dict, image_per_batch, label_per_batch, bbox_per_batch, batch_idx


def evaluate_loss(image_set, data_path, checkpoint, loss_types):
  """Detect image."""

  with tf.Graph().as_default():

    # Load model
    mc = kitti_squeezeDet_config()
    imdb = vkitti(image_set, data_path, mc)

    mc.BATCH_SIZE = 1
    # model parameters will be restored from checkpoint
    mc.LOAD_PRETRAINED_MODEL = False
    model = SqueezeDet(mc)

    saver = tf.train.Saver(model.model_params)
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    saver.restore(sess, checkpoint)

    losses = {
      'classification': model.class_loss,
      'confidence': model.conf_loss,
      'bbox': model.bbox_loss
    }
    print("LOSS TYPES:", loss_types)

    losses_t = [losses[lt] for lt in loss_types]
    print(losses_t)
    def get_loss_output(image):
      # Detect
      # sess.run(assign_image, feed_dict={model.adv_image_input: image})

      # Loss
      loss = model.loss if loss_types == "all" else tf.add_n(losses_t)
      _loss = sess.run(loss, feed_dict=loss_feed_dict)
      return _loss

    losses = []
    for i in range(len(imdb.image_idx)):
      loss_feed_dict, input_image, labels, bboxs, batch_idx = load_data(imdb, model, mc)
      loss = get_loss_output(input_image)
      losses.append(loss)

    losses = np.array(losses)
    print("AVG LOSS", np.mean(losses), np.median(losses), np.std(losses), losses.shape)
    return losses


      # np.savez_compressed(FLAGS.pixel_save_npz, image_train_ids=image_ids, image_pixel_gradients=image_pixel_gradients)
def main(argv=None):
  loss_types = str(FLAGS.loss_types).split(",")
  evaluate_loss(FLAGS.image_set, FLAGS.data_path, FLAGS.checkpoint, loss_types)

if __name__ == '__main__':
    tf.app.run()

