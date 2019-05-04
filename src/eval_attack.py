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

from dataset import pascal_voc, kitti, vkitti
from utils.util import sparse_to_dense, bgr_to_rgb, bbox_transform

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


tf.app.flags.DEFINE_integer('pgd_iters', 100, "number of iterations to run PGD")
tf.app.flags.DEFINE_float('learning_rate', 1e2, "learning rate for PGD/FGSM")
tf.app.flags.DEFINE_boolean('pixel_space', False, "generate examples only in pixel space")
tf.app.flags.DEFINE_boolean('gradient_sign', False, "update examples with only gradient signs")
tf.app.flags.DEFINE_integer('num_attacks', 1, "number of adv. examples to generate")

tf.app.flags.DEFINE_boolean('save_pixel_gradients', False, "whether to save pixel gradients")
tf.app.flags.DEFINE_string(
    'pixel_save_npz', 'image_grad.npz', """npz to save grads in""")

class SqueezeDetGradientAdversary():
    def __init__(self, model, loss=None, loss_type=None, learning_rate=1e-2):
        x, x_hat = model.adv_image_input, model.image_input
        self.x, self.x_hat = x, x_hat
        self.assign_op = tf.assign(x_hat, self.x)

        # LOSS can be one or sum of any of model.conf_loss, model.class_loss, model.bbox_loss
        # loss = -model.bbox_loss
        # TODO: allow combos of losses
        self.losses = {
            "bbox": -model.bbox_loss,
            "class": -model.class_loss,
            "conf": -model.conf_loss
        }

        if loss is not None:
            self.loss = loss
        else:
            self.loss = self.losses[loss_type]

        self.optimzer = tf.train.GradientDescentOptimizer(
            learning_rate)

        self.pixel_gradients = self.optimzer.compute_gradients(self.loss, var_list=[x_hat])

    # Pixel gradients already scaled by learning rate
    def get_pixel_gradients(self, img, sess, loss_feed_dict):
        sess.run(self.assign_op, feed_dict={self.x: img})
        # dL/dPixels of dimension same as img
        gradients, loss = sess.run([self.pixel_gradients, self.loss], feed_dict=loss_feed_dict)
        gradients, var = gradients[0]
        return gradients, loss

    # TODO: Fill in with renderer function @Lakshya @Wilson @Aish
    # z = textural code OR geometric vector
    def get_renderer_gradients(self, z, pixel_gradients):
        # dPixels/dTexture of dimension Texture OR dPixels/dGeometric of dimension Geometric
        gradients = None
        return gradients

    # Given z, a textural code or geometric vector, renders and returns a pixel space img
    def render(self, z):
        img = None
        return img

    def derender(self, img):
        z = None
        return z

    def _sign_gradient_step(self, x, grad):
        return x - np.sign(grad)

    def _gradient_step(self, x, grad):
        return x - grad

    def run_pgd(self, img, sess, loss_feed_dict, iters=100, epsilon=8,
                renderer_gradients=True, gradient_sign=False, log_iters=1,
                pixel_gradients_cache=None):
        if renderer_gradients:
            z = self.derender(img)
            x, x_hat = z, z
            low, high = -255, 255  # TODO: bounds of z?
        else:
            x, x_hat = img, img
            low, high = -124, 152 # [0, 255] - cfg.BGR_MEANS = np.array([[[103.939, 116.779, 123.68]]])

        print("x", x.shape)

        below = x - epsilon
        above = x + epsilon

        # x_hat is running adv. example
        for i in range(iters):
            if renderer_gradients:
                img_hat = self.render(x_hat)  # z -> img
                pixel_grad, curr_loss = self.get_pixel_gradients(img_hat, sess, loss_feed_dict)  # dL/dimg
                rend_grad = self.get_renderer_gradients(x_hat, pixel_grad)  # dimg/dz
                grad = rend_grad
            else:
                # print("x_hat", x_hat.shape)
                grad, curr_loss = self.get_pixel_gradients(x_hat, sess, loss_feed_dict)
                # print("grad", grad.shape)

                if pixel_gradients_cache is not None:
                  assert(iters == 1) # Need to gen grad one iter at a time
                  pixel_gradients_cache.append(np.squeeze(grad))

            x_hat = self._sign_gradient_step(x_hat, grad) if gradient_sign else self._gradient_step(x_hat, grad)
            # print("x_hat_updated", x_hat.shape)
            x_hat = np.clip(np.clip(x_hat, below, above), low, high)
            # print("x_hat_clipped", x_hat.shape)

            if i % log_iters == 0:
                print('step %d, loss=%g' % (i, curr_loss))

        return x_hat

    def run_fgsm(self, img, sess, loss_feed_dict, epsilon=8, renderer_gradients=True):
        return self.run_pgd(img, sess, loss_feed_dict, epsilon=epsilon, renderer_gradients=renderer_gradients, iters=1,
                            gradient_sign=True)

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

    return feed_dict, image_per_batch, label_per_batch, bbox_per_batch, batch_idx


def run_attack():
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

    imdb = vkitti(FLAGS.image_set, FLAGS.data_path, mc)

    saver = tf.train.Saver(model.model_params)
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    saver.restore(sess, FLAGS.checkpoint)

    # Save image IDs
    image_ids = []
    image_pixel_gradients = []

    for i in range(FLAGS.num_attacks):
      x, x_hat = model.adv_image_input, model.image_input

      # LOSS can be one or sum of any of model.conf_loss, model.class_loss, model.bbox_loss
      # loss = -model.bbox_loss
      # loss = -(model.class_loss)

      loss = model.loss
      adversary = SqueezeDetGradientAdversary(model, loss=loss, learning_rate=FLAGS.learning_rate)

      # for f in glob.iglob(FLAGS.input_path):
      # TODO: modify to allow reading from file?
      # im = cv2.imread(f)
      # im = im.astype(np.float32, copy=False)
      # im = cv2.resize(im, (mc.IMAGE_WIDTH, mc.IMAGE_HEIGHT))
      # print(im.max(), im.min())
      # input_image = im - mc.BGR_MEANS
      # print(type(im), im.shape)

      loss_feed_dict, input_image, labels, bboxs, batch_idx = load_data(imdb, model, mc)
      image_ids.append(batch_idx[0])

      print("Example:", i, batch_idx[0])
      print("bounds", input_image[0].max(), input_image[0].min())

      # Generate Adv
      adv_example = adversary.run_pgd(np.array(input_image), sess, loss_feed_dict,
                                      iters=FLAGS.pgd_iters, epsilon=8, renderer_gradients=(not FLAGS.pixel_space),
                                      gradient_sign=FLAGS.gradient_sign, pixel_gradients_cache=image_pixel_gradients if FLAGS.save_pixel_gradients else None
                                      )

      # TODO: Save adv as raw image
      # TODO: When generating adv. example, increment adv_idx = 00000, save example as KITTI/training_adv/image_2/adv_idx.png, and add adv_idx to ImageSets.txt
      # TODO: Then take batch_idx (batch of 1), grab KITTI/training/label_2/batch_idx.txt, and copy file into
      # TODO: KITTI/training_adv/label_2/adv_idx.txt
      adv_example = [adv_example[0,:,:,:]]
      todo_file_name = os.path.join(FLAGS.out_dir, "need_to_fix.png")
      im = adv_example[0] + mc.BGR_MEANS
      cv2.imwrite(todo_file_name, im)

      ### Evaluate Adv
      # Adv shape and distortion
      diff = np.array(adv_example) - np.array(input_image)
      print("linf norm", np.max(diff), "l2 norm", np.linalg.norm(diff))

      # Adv loss
      assign_image = tf.assign(model.image_input, model.adv_image_input)
      for name, image in [("benign", input_image), ("adv", adv_example)]:
          # Detect
          sess.run(assign_image, feed_dict={model.adv_image_input: image})

          # Loss
          print("Running loss")
          _attack_loss, _model_loss = sess.run([loss, model.loss], feed_dict=loss_feed_dict)
          print("LOSS: (attack)", _attack_loss, "(model)", _model_loss)

          # Render and save image
          det_boxes, det_probs, det_class = sess.run(
              [model.det_boxes, model.det_probs, model.det_class],
              # feed_dict={model.image_input:[input_image]}
          )

          im = image[0] + mc.BGR_MEANS
          print(name, type(im), im.shape)

          if name == "adv":
            out_file_name = os.path.join(FLAGS.out_dir, "image_2", batch_idx[0] + '.png')
            cv2.imwrite(out_file_name, im)

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

          file_name = 'sample.png'
          out_file_name = os.path.join(FLAGS.out_dir, 'out_'+ name + "_" + batch_idx[0] + '_' + file_name)
          cv2.imwrite(out_file_name, im)
          print ('Image detection output saved to {}'.format(out_file_name))

      np.savez_compressed(FLAGS.pixel_save_npz, image_train_ids=image_ids, image_pixel_gradients=image_pixel_gradients)
def main(argv=None):
  if not tf.gfile.Exists(FLAGS.out_dir):
    tf.gfile.MakeDirs(FLAGS.out_dir)
  run_attack()

if __name__ == '__main__':
    tf.app.run()

