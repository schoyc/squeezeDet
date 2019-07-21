import os
import shutil

original_set = "vkitti_originals_retraining"

mapping_f = open("%s_ids.txt" % original_set, 'r')
mappings = {}
for line in mapping_f:
        img_id, img_name = line.split()
        mappings[img_name] = img_id
        print("ID", img_id, "NAME", img_name)
root_dir = "../../../data/Sampled_Dataset/orig_training_images/imgs" # Data of fgsm attacks
#root_dir = "../../../3D-SDN/results/retraining/fgsm_combine/"
for attack_dir in os.listdir(root_dir):
        if "DS" in attack_dir:
                continue
        dataset_dir = "vkitti_set3_" + attack_dir
        print(dataset_dir)
        if not os.path.isdir(dataset_dir):
                os.mkdir(dataset_dir)
                os.mkdir(os.path.join(dataset_dir, "image_2"))
                os.mkdir(os.path.join(dataset_dir, "label_2"))


        imageset_f = open("./ImageSets/" + dataset_dir + ".txt", 'w+')

        for img in os.listdir(os.path.join(root_dir, attack_dir)):
                # Map img to ID
                img_id = mappings[img]
                print(img_id)
                imageset_f.write(img_id + '\n')

                # Copy img to ID.png, copy ID.txt from vkitti_rerendered_set2 to label_2
                # vkitti_rerendered_set2/label_2/
                shutil.copyfile(os.path.join(root_dir, attack_dir, img), os.path.join(dataset_dir, "image_2", "%s.png" % img_id))
                shutil.copyfile(os.path.join(original_set, "label_2", "%s.txt" % img_id), os.path.join(dataset_dir, "label_2", "%s.txt" % img_id))


