import cv2

import numpy as np
import tensorflow as tf

from config import *
from nets import *
from dataset import vkitti
from sqdet_utils.util import sparse_to_dense


class SqueezeDetGrads():
  def __init__(self, model_checkpoint, loss_types="all", learning_rate=1):
    mc = kitti_squeezeDet_config()
    mc.BATCH_SIZE = 1
    # model parameters will be restored from checkpoint
    mc.LOAD_PRETRAINED_MODEL = False
    model = SqueezeDet(mc)

    saver = tf.train.Saver(model.model_params)
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    saver.restore(sess, model_checkpoint)

    self.sess = sess

    self.model = model
    self.mc = mc
    self.learning_rate = learning_rate

    losses = {
      'classification': model.class_loss,
      'confidence': model.conf_loss,
      'bbox': model.bbox_loss
    }
    self.loss = model.loss if loss_types == "all" else tf.add_n([losses[lt] for lt in loss_types])
    self.grads = tf.gradients(self.loss, [model.image_input])
    self.imdb = None

  def load_dataset(self, data_path, imageset):
    self.imdb = vkitti(imageset, data_path, self.mc)

  def load_data(self, shuffle=False):
    if self.imdb is None:
      raise ValueError("Dataset not loaded, need to call load_dataset() first.")
    model, mc = self.model, self.mc
    # read batch input
    image_per_batch, label_per_batch, box_delta_per_batch, aidx_per_batch, \
    bbox_per_batch, batch_idx = self.imdb.read_batch(return_batch_idx=True, shuffle=shuffle)  # TODO: Modify to return batch idxs

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

  def set_feed_dict_image(self, image, loss_feed_dict):
    model, mc = self.model, self.mc
    image_input = model.image_input
    loss_feed_dict[image_input] = [image]

    return loss_feed_dict


  def get_pixel_gradients(self, loss_feed_dict):
    grads, loss = self.sess.run([self.grads, self.loss], feed_dict=loss_feed_dict)
    grads = grads[0]
    return grads, loss


if __name__ == '__main__':
  adversary = SqueezeDetGrads("model_checkpoints/squeezeDet/model.ckpt-87000")
  adversary.load_dataset("./data/VKITTI", "vkitti_originals_poster")
  loss_feed_dict, input_image, labels, bboxs, batch_idx = adversary.load_data()

  grad, loss = adversary.get_pixel_gradients(loss_feed_dict)
  print("LOSS:", loss)

  new_img = input_image + np.sign(grad)
  adversary.set_feed_dict_image(new_img[0], loss_feed_dict)

  grad, loss = adversary.get_pixel_gradients(loss_feed_dict)
  print("LOSS of adv example:", loss)