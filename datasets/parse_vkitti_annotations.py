import pandas as pd
import os

worlds_old = [
	'0018_clone', '0006_clone', '0018_sunset', '0006_rain', '0006_morning', '0018_15-deg-right', '0006_fog', '0006_sunset', '0006_15-deg-right', '0006_overcast', '0018_30-deg-right', '0006_30-deg-left', '0006_15-deg-left', '0020_30-deg-right'
]

worlds = [
'0002_30-deg-left', '0001_rain', '0002_15-deg-right', '0002_rain', 
'0002_morning', '0001_30-deg-left', '0020_sunset', '0018_15-deg-left', 
'0001_15-deg-left', '0002_fog', '0018_morning', '0020_15-deg-right', 
'0001_overcast', '0001_15-deg-right', '0002_sunset', '0018_fog', 
'0018_30-deg-left', '0001_clone', '0020_clone', '0002_overcast', '0002_clone', 
'0020_fog', '0002_15-deg-left', '0001_fog', '0018_overcast', 
'0002_30-deg-right', '0001_30-deg-right'
]

worlds = worlds + worlds_old
annotations_dir = "vkitti_1.3.1_motgt"

for world in worlds:
	world_dir = "vkitti_" + world
	if not os.path.isdir(world_dir):
		os.mkdir(world_dir)
		os.mkdir(os.path.join(world_dir, "image_2"))
		os.mkdir(os.path.join(world_dir, "label_2"))

	motgt = pd.read_csv("./%s/%s.txt" % (annotations_dir, world), sep=" ", index_col=False)
	f = open("./ImageSets/vkitti_%s.txt" % world, 'w+')

	for i in range(240):
		m = motgt[(motgt['frame'] == i) & (motgt['label'] != 'DontCare')]
		if m.shape[0] <= 0:
			continue
		f.write("%05d\n" % i)
		m = m.iloc[:,2:17]
		m.to_csv("./vkitti_%s/label_2/%05d.txt" % (world, i), sep=' ', header=False, index=False)

