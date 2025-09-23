import os
from glob import glob

cityscapes_root = '/home/ammiellewb/resources/datasets/cityscapes' # change this to your own path

def make_list(split):
    img_root = os.path.join(cityscapes_root, 'leftImg8bit', split)
    target_root = os.path.join(cityscapes_root, 'gtFine-relabeled', split)
    img_paths = sorted(glob(f"{img_root}/*/*.png"))
    target_paths = [
        os.path.join(target_root, city, os.path.basename(img_path).replace("_leftImg8bit.png", "_gtFine_labelIds.png"))
        for img_path in img_paths
        for city in [os.path.basename(os.path.dirname(img_path))]
    ]

    with open(os.path.join(cityscapes_root, f"{split}_img_paths.txt"), "w") as f_img, \
                open(os.path.join(cityscapes_root, f"{split}_target_paths.txt"), "w") as f_target:
        f_img.write("\n".join(img_paths))
        f_target.write("\n".join(target_paths))

for split in ['train', 'val', 'test']:
    make_list(split)