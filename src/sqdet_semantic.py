import numpy as np

from sqdet_adv import SqueezeDetGrads
from pipeline_3dsdn import *

GEO_PARAM_DICT = {'translate': '_translation2ds', 'rotate': '_theta_deltas', 'ffd': '_ffd_coeffs'}
TEXTURE_PARAM_DICT = {'road': 5, 'sky': 6, 'sidewalk': 7, 'foliage': 10, 'cars': -1}

adversary = SqueezeDetGrads("model_checkpoints/squeezeDet/model.ckpt-87000")

def example_pixel_adv():
  adversary.load_dataset("./data/VKITTI_FGSM", "vkitti_originals")
  loss_feed_dict, input_image, labels, bboxs, batch_idx = adversary.load_data()

  grad, loss = adversary.get_pixel_gradients(loss_feed_dict)
  print("LOSS:", loss)

  new_img = input_image + np.sign(grad)
  adversary.set_feed_dict_image(new_img[0], loss_feed_dict)

  grad, loss = adversary.get_pixel_gradients(loss_feed_dict)
  print("LOSS of adv example:", loss)

def iterative_fgsm(dataset_root, img_id, num_iterations, param, eps=.01):
  adversary.load_dataset("./data/VKITTI_FGSM", "vkitti_originals")
  loss_feed_dict, input_image, labels, bboxs, batch_idx = adversary.load_data() # Loads one image
  grad, loss = adversary.get_pixel_gradients(loss_feed_dict)

  # real_img = load_image(dataset_root, img_id)
  real_img = np.transpose(input_image, (2, 0, 1))
  segm_map, geo_params, geo_blob, tex_codes = derender(dataset_root, img_id)
  out = render(real_img, segm_map, geo_params, geo_blob, tex_codes)
  for i in range(num_iterations):
      out_tf = out.cpu().data.numpy().transpose((1,2,0))
      torch.cuda.empty_cache()
      loss_feed_dict = adversary.set_feed_dict_image(out_tf, loss_feed_dict)
      grads, loss = adversary.get_pixel_gradients(loss_feed_dict)
      print("LOSS: " + loss)
      grads = torch.from_numpy(grads[0, : , :, :]).permute(2,0,1)
      out.backward(grads.cuda())
      torch.cuda.empty_cache()
      if param in GEO_PARAM_DICT.keys():
          param = GEO_PARAM_DICT[param]
          arg_grad = geo_blob[param].grad.sign()
          new_blob = dict(geo_blob)
          new_blob[param] = geo_blob[param] + eps * arg_grad
          out = render(real_img, segm_map, geo_params, new_blob, tex_codes)
      elif param in TEXTURE_PARAM_DICT.keys():
          new_tex_codes = dict(tex_codes)
          if param == 'cars':
              instances = [k for k in tex_codes.keys() if k % 1000 == 0]
          else:
              instances = [TEXTURE_PARAM_DICT[param]]
          for i in instances:
              arg_grad = tex_codes[i].grad.sign()
              new_tex_codes[i] = tex_codes[i] + eps * arg_grad
          torch.cuda.empty_cache()
          out = render(real_img, segm_map, geo_params, geo_blob, new_tex_codes)
      out.zero_grad()
      print("DO WE GET HERE")
#   return out

if __name__ == '__main__':
    # out = iterative_fgsm('/home/lakshya/3D-SDN/datasets/vkitti_1.3.1_rgb/', '0006/30-deg-right/00043.png', 5, 'translate', eps=.02)
    # show_image_tensor(out, save=True, filename="demo_fgsm.png")

    # example_pixel_adv()