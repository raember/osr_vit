import shutil
from pathlib import Path

from tqdm import tqdm

val_folder = Path('./data/tiny-imagenet-200/val')
val_orig_folder = Path('./data/tiny-imagenet-200/val_orig')

shutil.rmtree(val_folder, ignore_errors=True)
val_folder.mkdir()

with open(val_orig_folder / 'val_annotations.txt') as fp:
    for line in tqdm(fp.readlines()):
        img_name, cls, bbox = line.split('\t', maxsplit=2)
        img_orig_fp = val_orig_folder / 'images' / img_name
        assert img_orig_fp.exists()
        cls_folder = val_folder / cls
        img_folder = cls_folder / 'images'
        img_folder.mkdir(parents=True, exist_ok=True)
        img_fp = img_folder / img_orig_fp.name
        img_fp.symlink_to(img_orig_fp.resolve(img_fp))
        with open(cls_folder / f"{cls}_boxes.txt", 'a') as ann_fp:
            ann_fp.write(f"{img_fp.name}\t{bbox}\n")
print('Done')
