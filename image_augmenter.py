from torchvision import transforms
from PIL import ImageFile
import os

resized_size = 128

input_dir = "./Input_Images"
output_dir = "./Augmented_Images"

num_images = 0

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter((0.95, 1.0), (0.95, 1.0), (0.95, 1.0), (-0.01, 0.01)),
])

num_created = 0
num_passes = 0
max_num = 50000

if os.path.isdir(input_dir):
    num_images = len(os.listdir(input_dir))

    while num_created <= max_num:

        if num_created >= max_num:
            break

        for i in range(len(os.listdir(input_dir))):

            if num_created >= max_num:
                break

            orig_fname = os.listdir(input_dir)[i]
            orig_fpath = input_dir + "/" + orig_fname

            try:
                img = ImageFile.Image.open(orig_fpath)
                new_img = img.convert('RGB')
                new_img = transform(new_img)
                new_fpath = output_dir + "/augmented_08_train_" + str(num_passes + 1) + "_" + orig_fname
                new_img.save(new_fpath, "JPEG")
                num_created += 1

            except OSError:
                print("Cannot load : {}".format(orig_fpath))

        num_passes += 1

exit()
