import cv2
import time
import sys
import os
import glob

import numpy as np
import tensorflow as tf

from config import *
from train import _draw_box
from nets import *

from dataset import pascal_voc, kitti
from utils.util import sparse_to_dense, bgr_to_rgb, bbox_transform

class PGDAttack():
    def __init__(self, x, x_hat, loss, epsilon=0.1, learning_rate=1e-2):
        self.x, self.x_hat = x, x_hat
        self.assign_op = tf.assign(x_hat, self.x)
        self.loss = loss

        self.optim_step = tf.train.GradientDescentOptimizer(
            learning_rate).minimize(loss, var_list=[x_hat])

        below = x - epsilon
        above = x + epsilon
        projected = tf.clip_by_value(tf.clip_by_value(x_hat, below, above), -255, 255)
        with tf.control_dependencies([projected]):
            self.project_step = tf.assign(x_hat, projected)

    def run_attack(self, img, loss_feed_dict, sess, num_gd_steps=100):

        # initialization step
        sess.run(self.assign_op, feed_dict={self.x: img})

        # projected gradient descent
        for i in range(num_gd_steps):
            # gradient descent step
            _, loss_value = sess.run([self.optim_step, self.loss],
                            feed_dict=loss_feed_dict)

            # project step
            sess.run(self.project_step, feed_dict={self.x: img})
            if i % 5 == 0:
                print('step %d, loss=%g' % (i, loss_value))

        adv_example = self.x_hat.eval(sess)
        return adv_example

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'mode', 'image', """'image' or 'video'.""")
tf.app.flags.DEFINE_string(
    'checkpoint', './data/model_checkpoints/squeezeDet/model.ckpt-87000',
    """Path to the model parameter file.""")
tf.app.flags.DEFINE_string(
    'input_path', './data/sample.png',
    """Input image or video to be detected. Can process glob input such as """
    """./data/00000*.png.""")
tf.app.flags.DEFINE_string(
    'out_dir', './data/out/', """Directory to dump output image or video.""")
tf.app.flags.DEFINE_string(
    'demo_net', 'squeezeDet', """Neural net architecture.""")


def setup_pgd(model):
    x, x_hat = model.adv_image_input, model.image_input
    loss = -model.bbox_loss

    pgd = PGDAttack(x, x_hat, loss, epsilon=8, learning_rate=1e-2)

    return pgd



def pgd_demo():
  """Detect image."""

  assert FLAGS.demo_net == 'squeezeDet' or FLAGS.demo_net == 'squeezeDet+', \
      'Selected nueral net architecture not supported: {}'.format(FLAGS.demo_net)

  with tf.Graph().as_default():
    # Load model
    if FLAGS.demo_net == 'squeezeDet':
      mc = kitti_squeezeDet_config()
      mc.BATCH_SIZE = 1
      # model parameters will be restored from checkpoint
      mc.LOAD_PRETRAINED_MODEL = False
      model = SqueezeDetAdv(mc, FLAGS.gpu)
    elif FLAGS.demo_net == 'squeezeDet+':
      mc = kitti_squeezeDetPlus_config()
      mc.BATCH_SIZE = 1
      mc.LOAD_PRETRAINED_MODEL = False
      model = SqueezeDetPlus(mc, FLAGS.gpu)

    imdb = kitti(FLAGS.image_set, FLAGS.data_path, mc)

    def _load_data():
        # read batch input
        image_per_batch, label_per_batch, box_delta_per_batch, aidx_per_batch, \
        bbox_per_batch = imdb.read_batch()

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


        image_input = model.adv_image_input
        input_mask = model.input_mask
        box_delta_input = model.box_delta_input
        box_input = model.box_input
        labels = model.labels

        feed_dict = {
            # image_input: image_per_batch,
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

        return feed_dict, image_per_batch, label_per_batch, bbox_per_batch

    saver = tf.train.Saver(model.model_params)

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
      saver.restore(sess, FLAGS.checkpoint)

      x, x_hat = model.adv_image_input, model.image_input

      # LOSS can be one or sum of any of model.conf_loss, model.class_loss, model.bbox_loss
      # loss = -model.bbox_loss
      loss = -(model.conf_loss + model.class_loss)

      pgd = PGDAttack(x, x_hat, loss, epsilon=8, learning_rate=100)

      for f in glob.iglob(FLAGS.input_path):
        im = cv2.imread(f)
        im = im.astype(np.float32, copy=False)
        im = cv2.resize(im, (mc.IMAGE_WIDTH, mc.IMAGE_HEIGHT))
        print(im.max(), im.min())
        input_image = im - mc.BGR_MEANS
        print(type(im), im.shape)


        loss_feed_dict, input_image, labels, bboxs = _load_data()
        # print("imgage", input_image)
        # print("image", len(image), image[0].shape, "input", input_image.shape)

        print("bounds", input_image[0].max(), input_image[0].min())
        # Assign
        assign_image = tf.assign(model.image_input, model.adv_image_input)
        print("Ran assignment")
        sess.run(assign_image, feed_dict={model.adv_image_input: input_image})

        # Generate Adv
        adv = pgd.run_attack(input_image, loss_feed_dict, sess, num_gd_steps=100)
        print(adv.shape)
        print("linf norm", np.max(adv-input_image), "l2 norm", np.linalg.norm(adv - input_image))

        for name, image in [("benign", input_image), ("adv", adv)]:
            im = image[0] + mc.BGR_MEANS
            print(name, type(im), im.shape)
            # Detect
            print("Running forward pass")
            # print("type", type(model.image_input), type(model.ph_image_input))
            sess.run(assign_image, feed_dict={model.adv_image_input: image})
            det_boxes, det_probs, det_class = sess.run(
                [model.det_boxes, model.det_probs, model.det_class],
                    # feed_dict={model.image_input:[input_image]}
            )

            # Loss
            print("Running loss")
            _loss = sess.run(loss, feed_dict=loss_feed_dict)
            print("LOSS:", _loss)


            # Filter
            final_boxes, final_probs, final_class = model.filter_prediction(
                det_boxes[0], det_probs[0], det_class[0])

            keep_idx    = [idx for idx in range(len(final_probs)) \
                              if final_probs[idx] > mc.PLOT_PROB_THRESH]
            final_boxes = [final_boxes[idx] for idx in keep_idx]
            final_probs = [final_probs[idx] for idx in keep_idx]
            final_class = [final_class[idx] for idx in keep_idx]

            # TODO(bichen): move this color dict to configuration file
            cls2clr = {
                'car': (255, 191, 0),
                'cyclist': (0, 191, 255),
                'pedestrian':(255, 0, 191)
            }

            # Draw boxes
            _draw_box(
                im, final_boxes,
                [mc.CLASS_NAMES[idx]+': (%.2f)'% prob \
                    for idx, prob in zip(final_class, final_probs)],
                cdict=cls2clr,
            )

            file_name = os.path.split(f)[1]
            out_file_name = os.path.join(FLAGS.out_dir, 'out_'+ name + "_" + file_name)
            cv2.imwrite(out_file_name, im)
            print ('Image detection output saved to {}'.format(out_file_name))


def main(argv=None):
  if not tf.gfile.Exists(FLAGS.out_dir):
    tf.gfile.MakeDirs(FLAGS.out_dir)
  pgd_demo()

if __name__ == '__main__':
    tf.app.run()

# NET = 'squeezeDet'
# def main():
#   """Train SqueezeDet model"""
#   os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
#
#   with tf.Graph().as_default():
#
#     if NET == 'squeezeDet':
#       mc = kitti_squeezeDet_config()
#       mc.IS_TRAINING = True
#       mc.PRETRAINED_MODEL_PATH = FLAGS.pretrained_model_path
#       mc.BATCH_SIZE = 1
#       model = SqueezeDet(mc)
#     elif NET == 'squeezeDet+':
#       mc = kitti_squeezeDetPlus_config()
#       mc.IS_TRAINING = True
#       mc.PRETRAINED_MODEL_PATH = FLAGS.pretrained_model_path
#       mc.BATCH_SIZE = 1
#       model = SqueezeDetPlus(mc)

