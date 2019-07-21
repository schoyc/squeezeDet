import numpy as np

from sqdet_adv import SqueezeDetGrads
from pipeline_3dsdn import *
import argparse
import scipy.ndimage as ndimage

parser = argparse.ArgumentParser()
parser.add_argument("image_index", help = "index to work with", type=int)
parser.add_argument("use_pgd", type=int)
parser.add_argument("-c", "--combine", type=int, default=0)
parser.add_argument("--attack_type", type=int, default=0)
parser.add_argument("-id", "--identity", type=int, default=0)
parser.add_argument("-l", "--loss_type", type=int, default=0)
args = parser.parse_args()
IMAGE_IDX = args.image_index
USE_PGD = args.use_pgd
combine = args.combine
loss_type = args.loss_type
if loss_type == 1:
    loss_dict = ['bbox', 'classification']
elif loss_type == 2:
    loss_dict = ['confidence']
else:
    loss_dict = ['bbox', 'classification', 'confidence']

modify_texture_only = False
if not combine:
    attack_dict = {0 : ('translate', .02), 1: ('rotate', .005), 2: ('ffd', .02), 3: ('sky', .25), 4: ('foliage', .25), 5: ('cars', .05)}
    attack_type, attack_eps = attack_dict[args.attack_type]
    if args.attack_type in list(range(3)):
        tex_feature_keys = []
        geo_feature_keys = [(attack_type, attack_eps)]
    else:
        geo_feature_keys = []
        tex_feature_keys = [(attack_type, attack_eps)]
        modify_texture_only = True
else:
    if not USE_PGD:
        geo_feature_keys = [('translate', .025), ('rotate', .005)]
        tex_feature_keys = [('sky', .25), ('cars', .05)]
    else:
        geo_feature_keys = [('translate', .025), ('rotate', .005)]
        tex_feature_keys = [('sky', .25), ('cars', .05)]
identity = args.identity

GEO_PARAM_DICT = {'translate': '_translation2ds', 'rotate': '_theta_deltas', 'ffd': '_ffd_coeffs'}
TEXTURE_PARAM_DICT = {'road': 5, 'sky': 6, 'sidewalk': 7, 'foliage': 10, 'cars': -1}
# ipdb.set_trace()
# adversary = SqueezeDetGrads("../model_checkpoints/squeezeDet/model.ckpt-87000")

def parse_mapping_ids(fname):
    mappings = dict()
    with open(fname, 'r') as f:
        for line in f:
            data = line.strip('\n').split(' ')
            index, img_id = data[0], data[1]
            mappings[index] = img_id 
    return mappings

def example_pixel_adv():
    adversary.load_dataset("./../data/VKITTI_FGSM", "vkitti_originals")
    loss_feed_dict, input_image, labels, bboxs, batch_idx = adversary.load_data()
    grad, loss = adversary.get_pixel_gradients(loss_feed_dict)
    print("LOSS:", loss)

    new_img = input_image + np.sign(grad)
    adversary.set_feed_dict_image(new_img[0], loss_feed_dict)

    grad, loss = adversary.get_pixel_gradients(loss_feed_dict)
    print("LOSS of adv example:", loss)

def adv_semantic_attack(dataset_root, num_iterations, geo_feature_keys, tex_feature_keys, pgd=False):
    if modify_texture_only:
        adversary = SqueezeDetGrads("../model_checkpoints/best_rerendered_checkpoints/model.ckpt-8400")
    elif loss_type == 1 or loss_type == 2:
        adversary = SqueezeDetGrads("../model_checkpoints/best_rerendered_checkpoints/model.ckpt-8400", loss_types=loss_dict)
    else:
        adversary = SqueezeDetGrads("../model_checkpoints/best_rerendered_checkpoints/model.ckpt-8400", loss_types=['classification', 'confidence'])
    # mappings = parse_mapping_ids('vkitti_originals_retraining_ids.txt')
    # adversary.load_dataset("./../data/ADV_ATTACK_SET/", "vkitti_originals_retraining")
    mappings = parse_mapping_ids()
    adversary.load_dataset("./../data/VKITTI_TEST/", "vkitti_originals")
    # tex_opt = get_opt()
    # pix2pix_model = create_pix2pix_model(tex_opt)
    loss_feed_dict, input_image, labels, bboxs, batch_idx = adversary.load_data([IMAGE_IDX]) # Loads one image
    print("BEEP", batch_idx[0])
    img_id = mappings[batch_idx[0]].replace('_', '/')
    grad, loss = adversary.get_pixel_gradients(loss_feed_dict)

    # real_img = load_image(dataset_root, img_id)
    real_img = np.transpose(input_image[0], (2, 0, 1))
    segm_map, geo_params, geo_blob, tex_codes = derender(dataset_root, img_id)
    real_img = torch.FloatTensor(real_img).cuda()
    torch.cuda.empty_cache()
    geom_feats = ['_theta_deltas', '_translation2ds',
                  '_log_scales', '_ffd_coeffs',]
    for feat in geom_feats:
        geo_blob[feat] = Variable(geo_blob[feat].cuda(), requires_grad=True)

    for k in tex_codes:
        tex_codes[k] = Variable(torch.FloatTensor(tex_codes[k]).cuda(), requires_grad=True)

    # print(real_img)
    out = render(real_img, segm_map, geo_params, geo_blob, tex_codes)
    #save original image
    show_image_tensor(out, save=True, filename="../../3D-SDN/results/experiments/orig/" + mappings[batch_idx[0]])
    loss_tracker = []
    var_lst = []
    img_outs = []
    toughest_img = None
    worst_loss = -1
    for i in range(num_iterations):
        out_tf = out.cpu().data.numpy().transpose((1,2,0))
        torch.cuda.empty_cache()
        loss_feed_dict = adversary.set_feed_dict_image(out_tf, loss_feed_dict)
        grads, loss = adversary.get_pixel_gradients(loss_feed_dict)
        print("LOSS: ", loss)
        loss_tracker += [loss]
        if i > 0 and loss > worst_loss:
            del toughest_img
            torch.cuda.empty_cache()
            toughest_img = out.data.cpu()
            worst_loss = loss
        grads = torch.from_numpy(grads[0, : , :, :]).permute(2,0,1)
        out.backward(grads.cuda(), retain_graph = True)
        var_lst = []
        torch.cuda.empty_cache()
        # new_blob = dict(geo_blob)
        print("---")
        for param, eps in geo_feature_keys:
            pname = GEO_PARAM_DICT[param]
            print(param)
            arg_grad = geo_blob[pname].grad.data
            if not pgd:
                arg_grad = arg_grad.sign()
                # print("ADDENDUM:", torch.max(geo_blob[pname]))
                # print("BLOB COMPONENT:", geo_blob[pname].data)
                # print("GRADS:", arg_grad)
                geo_blob[pname].data = geo_blob[pname] + eps * arg_grad
            else:
                geo_blob[pname].data = geo_blob[pname] + eps * arg_grad
        mroi = geo_blob['_mroi_norms']
        droi = geo_blob['_droi_norms']
        old_translation2ds = geo_blob['_translation2ds'].detach()
        old_log_depth = geo_blob['_log_depths'].detach()
        for feat in geom_feats:
            var_lst += [geo_blob[feat]]
        geo_blob['_log_depths'] = geo_blob['_log_depths'] + log_perspective_rescale(old_translation2ds, geo_blob['_translation2ds'], old_log_depth, mroi, droi)
        new_tex_codes = dict(tex_codes)
        for param, eps in tex_feature_keys:
            if param == 'cars':
                instances = [k for k in tex_codes.keys() if k % 1000 == 0]
            else:
                instances = [TEXTURE_PARAM_DICT[param]]
            for j in instances:
                arg_grad = tex_codes[j].grad.data
                if not pgd:
                    arg_grad = arg_grad.sign()
                    tex_codes[j].data = tex_codes[j] + torch.max(tex_codes[j]) * eps * arg_grad
                else:
                    tex_codes[j].data = tex_codes[j] + eps * arg_grad
                # old_code = tex_codes[j].clone()
                # tex_codes[j].data = tex_codes[j] + torch.max(.05 * tex_codes[j], torch.min(0.05 * tex_codes[j], eps * arg_grad))
                # tex_codes[j].data = tex_codes[j] + eps * arg_grad
                var_lst += [tex_codes[j]]
        torch.cuda.empty_cache()
        # print(tex_codes)
        # for k in tex_codes:
        #     print(new_tex_codes[k])
        #     tex_codes[k] = Variable(torch.FloatTensor(new_tex_codes[k].detach()), requires_grad=True)
        # print(param)
        del out
        torch.cuda.empty_cache()
        out = render(real_img, segm_map, geo_params, geo_blob, tex_codes)
        # img_outs += [out]
        for v in var_lst:
            if v.grad is not None:
                v.grad.data.zero_()

    out_tf = out.cpu().data.numpy().transpose((1,2,0))
    torch.cuda.empty_cache()
    loss_feed_dict = adversary.set_feed_dict_image(out_tf, loss_feed_dict)
    grads, loss = adversary.get_pixel_gradients(loss_feed_dict)
    loss_tracker += [loss]
    if loss > worst_loss:
        toughest_img = out
        worst_loss = loss
    print("worst loss:", worst_loss)
    if combine:
        direc = "combine"
    else:
        direc = param
    if not pgd:
        # show_image_tensor(toughest_img, save=True, filename="../../3D-SDN/results/fgsm/fgsm_" + direc + "/" + mappings[batch_idx[0]])
        # show_image_tensor(toughest_img, save=True, filename="../../3D-SDN/results/retraining/fgsm_" + direc + "/" + mappings[batch_idx[0]])
        show_image_tensor(toughest_img, save=True, filename="../../3D-SDN/results/experiments/fgsm_" + direc + "_" + str(loss_type) + "/" + mappings[batch_idx[0]])        
        # print("../../3D-SDN/results/fgsm/fgsm_" + direc + "/" + mappings[batch_idx[0]])
    else:
        show_image_tensor(toughest_img, save=True, filename="../../3D-SDN/results/pgd/pgd_" + direc + "/" + mappings[batch_idx[0]])
        # show_image_tensor(toughest_img, save=True, filename="../../3D-SDN/results/retraining/pgd_" + direc + "/" + mappings[batch_idx[0]])

    # print(loss_tracker)
    return out

def identity_transform(dataset_root, img_idx, pgd=False):
    if modify_texture_only:
        adversary = SqueezeDetGrads("../model_checkpoints/squeezeDet/model.ckpt-87000")
    else:
        adversary = SqueezeDetGrads("../model_checkpoints/squeezeDet/model.ckpt-87000", loss_types=['classification', 'confidence'])
    # mappings = parse_mapping_ids('vkitti_sampled_ids.txt')
    adversary.load_dataset("./../data/Sampled_Dataset/orig_training_images", "vkitti_originals_retraining")
    mappings = parse_mapping_ids('vkitti_originals_retraining_ids.txt')
    loss_feed_dict, input_image, labels, bboxs, batch_idx = adversary.load_data([IMAGE_IDX]) # Loads one image
    print("BEEP", batch_idx[0])
    img_id = mappings[batch_idx[0]].replace('_', '/')
    # adversary.load_dataset("./../data/VKITTI_FGSM", "vkitti_originals")
    # tex_opt = get_opt()
    # pix2pix_model = create_pix2pix_model(tex_opt)
    # loss_feed_dict, input_image, labels, bboxs, batch_idx = adversary.load_data([IMAGE_IDX]) # Loads one image

    # grad, loss = adversary.get_pixel_gradients(loss_feed_dict)

    # real_img = load_image(dataset_root, img_id)
    # real_img = np.transpose(input_image[0], (2, 0, 1))
    real_img = ndimage.imread(dataset_root+img_id)
    segm_map, geo_params, geo_blob, tex_codes = derender(dataset_root, img_id)
    real_img = torch.FloatTensor(real_img).cuda()
    torch.cuda.empty_cache()
    geom_feats = ['_theta_deltas', '_translation2ds',
                  '_log_scales', '_ffd_coeffs',]
    for feat in geom_feats:
        geo_blob[feat] = Variable(geo_blob[feat].cuda(), requires_grad=True)

    for k in tex_codes:
        tex_codes[k] = Variable(torch.FloatTensor(tex_codes[k]).cuda(), requires_grad=True)

    # print(real_img)
    out = render(real_img, segm_map, geo_params, geo_blob, tex_codes)
    #save original image
    show_image_tensor(out, save=True, filename="../data/Sampled_Dataset/retraining_rerendered/" + mappings[str(img_idx).zfill(5)])
    return out

if __name__ == '__main__':
    if identity:
        identity_transform('/home/lakshya/3D-SDN/datasets/vkitti_1.3.1_rgb/', IMAGE_IDX)
    else:
        out = adv_semantic_attack('/home/lakshya/3D-SDN/datasets/vkitti_1.3.1_rgb/', 6, geo_feature_keys, tex_feature_keys, pgd=USE_PGD)
    print("HEY")
