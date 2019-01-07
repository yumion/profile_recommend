import glob
import os

for season in ['spring', 'summer', 'autumn', 'winter']:
    i = 0
    img_list = []
    for image_suffix in ['jpg', 'png', 'jpeg', 'JPG', 'PNG']:
        image_paths = 'dataset/'+season+'/*.' + image_suffix
        img_list += glob.glob(image_paths)
    for file in img_list:
        suffix = file.split('.')[-1]
        os.rename(file, 'dataset/' + season + '/%03d.'%(i) + suffix)
        i += 1
