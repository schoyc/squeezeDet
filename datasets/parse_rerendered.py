import os
import re 
import shutil

#input_dir = "./fgsm_data_proj/fgsm_combine/"
input_dir = "../../../data/Sampled_Dataset/orig_training_images/imgs/img"
set_name = "vkitti_originals_retraining"
mapping_f = open("%s_ids.txt" % set_name, 'w+')
imageset_f = open("./ImageSets/%s.txt" % set_name, 'w+')
ids = set([])

dataset_dir = set_name
if not os.path.isdir(dataset_dir):
        print("DIR", dataset_dir)
        os.mkdir(dataset_dir)
        os.mkdir(os.path.join(dataset_dir, "image_2"))
        os.mkdir(os.path.join(dataset_dir, "label_2"))

for new_id, img in enumerate(os.listdir(input_dir)):
        match = re.match(r'([0-9]{4}_.*)_([0-9]{5})\.png', img)
        print("new_id", new_id, "img", img)
        world = match.group(1)
        img_id = match.group(2)

        labels = "./vkitti_" + world + "/label_2/" + img_id + ".txt"
        if not os.path.isfile(labels):
                print("REMOVE", labels)
                os.remove(input_dir + "/" + img)
                continue

        imageset_f.write("%05d" % new_id + "\n")
        
        mapping = "%05d" % new_id + " " + img
        mapping_f.write(mapping + "\n")

        
        shutil.copyfile(os.path.join(input_dir, img), "./%s/image_2/" % set_name + "%05d.png" % new_id)
        shutil.copyfile(labels, "./%s/label_2/" % set_name + "%05d.txt" % new_id)
        # map img name to new numerical ID
