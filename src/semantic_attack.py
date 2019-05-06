import tensorflow as tf
from squeezeDet.src.config import *
from squeezeDet.src.nets import SqueezeDet
from pipeline_3dsdn import *
class ModelGrads():
    def __init__(self, model, learning_rate = 1e-3):
        self.model = model
        self.learning_rate = learning_rate
        self.loss = -model.conf_loss
        print(self.loss)
        self.optimizer = tf.train.AdamOptimizer(learning_rate)
        #im_in = tf.transpose(model.image_input, (0, 1, 2, 3))
        self.grads = tf.gradients(self.loss, [model.image_input])
        #self.grads = self.optimizer.compute_gradients(self.loss, var_list=[model.image_input])

GEO_PARAM_DICT = {'translate': '_translation2ds', 'rotate': '_theta_deltas', 'ffd': '_ffd_coeffs'}
TEXTURE_PARAM_DICT = {'road': 5, 'sky': 6, 'sidewalk': 7, 'foliage': 10, 'cars': -1}

def iterative_fgsm(dataset_root, img_id, num_iterations, param, eps=.01):
    mc = kitti_squeezeDet_config()
    mc.BATCH_SIZE = 1
    # model parameters will be restored from checkpoint
    mc.LOAD_PRETRAINED_MODEL = False
    model = SqueezeDet(mc)
    sess = tf.Session()
    real_img = load_image(dataset_root, img_id)
    segm_map, geo_params, geo_blob, tex_codes = derender(dataset_root, img_id)
    model_grads = ModelGrads(model)
    out = render(real_img, segm_map, geo_params, geo_blob, tex_codes)
    for i in range(num_iterations):
        torch.save(out, 'out_torch.pt')
        torch.save(segm_map, 'segm_map.pt')
        #torch.save(geo_params, 'model_grads.pt')
        #torch.save(geo_blob, 'geo_blob.pt')
        torch.save(tex_codes, 'tex_codes.pt')
        out_tf = out.cpu().data.numpy().transpose((1,2,0))
        del out, segm_map, tex_codes
        torch.cuda.empty_cache()
        print("OUT SHAPE")
        print(out_tf)
        grads, loss = sess.run([model_grads.grads, model_grads.loss],feed_dict={model.image_input:[out_tf]})
        print("LOSS: " + loss)
        grads = torch.from_numpy(grads[0, : , :, :]).permute(2,0,1)
        out = torch.load('out_torch.pt')
        segm_map = torch.load('segm_map.pt')
        #geo_params = torch.load('model_grads.pt')
        #geo_blob = torch.load('geo_blob.pt')
        tex_codes = torch.load('tex_codes.pt')
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
    return out

if __name__ == '__main__':
    out = iterative_fgsm('/home/lakshya/3D-SDN/datasets/vkitti_1.3.1_rgb/', '0006/30-deg-right/00043.png', 5, 'translate', eps=.02)
    show_image_tensor(out, save=True, filename="demo_fgsm.png")
