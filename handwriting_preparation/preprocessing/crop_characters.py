import os
import glob

from PIL import Image, ImageEnhance

from model.preprocessing_helper import draw_single_char, CANVAS_SIZE, CHAR_SIZE


def char_img_iter(image_path, box_path):
    assert os.path.isfile(image_path), "image file doesn't exist: %s" % image_path
    assert os.path.isfile(box_path), "image box file doesn't exist %s" % box_path

    n = 0
    img = Image.open(image_path)
    with open(box_path, "r") as f:
        for line in f:
            if n >= 156 and "test_image.jpg" in image_path: break  # custom rules, so that you don't have fix all the box results

            ch, x1, y1, x2, y2, _ = line.rstrip().split(' ')
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)

            # Crop the character based on Box result
            char_img = img.crop((x1, img.size[1] - y2, x2, img.size[1] - y1))

            # Leave enough space and resize to canvas_size
            char_img = draw_single_char(char_img, canvas_size=CANVAS_SIZE, char_size=CHAR_SIZE)

            # Add contrast
            contrast = ImageEnhance.Contrast(char_img)
            char_img = contrast.enhance(2.)

            # Add brightness
            brightness = ImageEnhance.Brightness(char_img)
            char_img = brightness.enhance(2.)

            yield ch, char_img
            n += 1


def pre_cropped_char_img_iter(image_dir, thresh = 230):
    assert os.path.isdir(image_dir), "image directory doesn't exist: %s" % image_dir

    n = 0
    image_paths = glob.glob(os.path.join(image_dir, '*'))

    for image_path in image_paths:

        char_img = Image.open(image_path)
        ch = os.path.basename(image_path).split('_')[0]

        # Leave enough space and resize to canvas_size
        char_img = draw_single_char(char_img, canvas_size=CANVAS_SIZE, char_size=CHAR_SIZE)

        # Add contrast
        contrast = ImageEnhance.Contrast(char_img)
        char_img = contrast.enhance(2.0)

        # Add brightness
        brightness = ImageEnhance.Brightness(char_img)
        char_img = brightness.enhance(1.3)

        # Binarize
        fn = lambda x : 255 if x > thresh else 0
        char_img = char_img.convert('L').point(fn, mode='1')

        yield ch, char_img
        n += 1


if __name__ == '__main__':
    image_path = "/media/jscarlson/ADATASE800/Japan/font_gen/paired_training_data/tk/labeled_validated_char_crops"
    out_dir = "/home/jscarlson/Downloads"

    # makedir
    try:
        os.makedirs(out_dir)
    except:
        pass

    # For debug only
    debug_counter = 0
    for ch, char_img in pre_cropped_char_img_iter(image_path):
        char_img.save(os.path.join(out_dir, ch + ".jpg"), "JPEG", quality=100)
        debug_counter += 1
        if debug_counter > 10:
            print("Exiting...")
            exit(0)