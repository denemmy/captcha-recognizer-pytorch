from matplotlib import pyplot as plt
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import random
import math
from os import mkdir
from os.path import join, isdir, splitext
import tqdm

PATH_TO_FONT = '/Users/denemmy/Downloads/msuighub.ttf'


def get_random_symbol():
    first = ord('a')
    count = ord('z') - ord('a') + 1
    random_symbol = first + np.random.randint(0, count)
    return chr(random_symbol)


def get_symbol_code(symbol):

    assert ord(symbol) <= ord('z') and ord(symbol) >= ord('a')
    code = ord(symbol) - ord('a')

    return code

def get_encoding(text):
    encoding = []
    for s in text:
        code = get_symbol_code(s)
        encoding.append(code)
    return encoding


def get_random_color():
    color = tuple(np.random.randint(0, 256, 3))
    return color


def draw_rotated_text2(img, angle, xy, text, color_font, *args, **kwargs):
    """ Draw text at an angle into an image, takes the same arguments
        as Image.text() except for:

    :param image: Image to write text into
    :param angle: Angle to write text at
    """
    # get the size of our image
    width, height = img.size
    max_dim = max(width, height)

    # build a transparency mask large enough to hold the text
    mask_size = (max_dim * 2, max_dim * 2)
    mask = Image.new('L', mask_size, 0)

    # add text to mask
    draw = ImageDraw.Draw(mask)
    draw.text((max_dim, max_dim), text, 255, *args, **kwargs)

    if angle % 90 == 0:
        # rotate by multiple of 90 deg is easier
        rotated_mask = mask.rotate(angle)
    else:
        # rotate an an enlarged mask to minimize jaggies
        bigger_mask = mask.resize((max_dim * 8, max_dim * 8),
                                  resample=Image.BICUBIC)
        rotated_mask = bigger_mask.rotate(angle).resize(
            mask_size, resample=Image.LANCZOS)

    # crop the mask to match image
    mask_xy = (max_dim - xy[0], max_dim - xy[1])
    b_box = mask_xy + (mask_xy[0] + width, mask_xy[1] + height)
    mask = rotated_mask.crop(b_box)

    # paste the appropriate color, with the text transparency mask
    color_image = Image.new('RGBA', img.size, color_font)
    img.paste(color_image, mask)


def generate_captcha(width=200, height=70, color_fill=(255, 255, 255), color_font=None):
    if color_font is None:
        color_font = get_random_color()
    img = Image.new('RGB', (width, height), color=color_fill)
    d = ImageDraw.Draw(img)

    d.rectangle([0, 0, width + 1, height + 1], fill=color_fill)
    number_of_symbols = 5

    base_fnt_sz = 75 + np.random.randint(-4, 4)

    x_pos = 35 + np.random.randint(0, 30)
    T = 230
    w = 2 * math.pi / T
    phase = math.radians(np.random.randint(0, 360))
    resolve_text = ''
    amp = 7 + np.random.randint(-2, 2)

    for i in range(number_of_symbols):
        symbol = get_random_symbol()
        resolve_text += symbol
        y_shift = amp * math.sin(w * x_pos + phase)
        fnt_sz = base_fnt_sz + np.random.randint(-2, 2)
        fnt = ImageFont.truetype(PATH_TO_FONT, fnt_sz)
        rad = math.atan(amp * w * math.cos(w * x_pos + phase))
        degrees = math.degrees(rad)
        draw_rotated_text2(img, degrees, (x_pos, y_shift), symbol, color_font, font=fnt)

        x_pos += d.textsize(symbol, font=fnt)[0] + np.random.randint(-6, -1)

    T = 150 + np.random.randint(0, 100)
    w = 2 * math.pi / T
    phase = math.radians(np.random.randint(0, 360))
    amp = 5 + np.random.randint(0, 10)
    for i in range(width):
        if np.random.randint(0, 4) > 1:
            y_shift = np.random.randint(-1, 1)
            d.point([i, height / 2 + amp * math.sin(w * i + phase) + y_shift], color_font)

    for i in range(800 + np.random.randint(0, 1200)):
        d.point((np.random.randint(0, width), np.random.randint(0, height)), fill=color_fill)

    for i in range(7 + np.random.randint(0, 3)):
        x1 = np.random.randint(0, width)
        y1 = np.random.randint(0, height)
        x2 = np.random.randint(0, width)
        y2 = np.random.randint(0, height)
        d.line([x1, y1, x2, y2], color_fill, 2)
        if ((i % 3) == 0):
            d.line([x1, y1 + 3, x2, y2 + 3], color_fill, 1)
        if (i % 3 == 1):
            d.line([x1 + 3, y1, x2 + 3, y2], color_fill, 1)

    img = np.array(img)

    return img, resolve_text


def prepare_synthetic_dataset():
    output_dir = '/media/denemmy/hdd/data/captcha_v2'
    subdirname = 'generated'
    output_subdir = join(output_dir, subdirname)
    if not isdir(output_dir):
        mkdir(output_dir)
    if not isdir(output_subdir):
        mkdir(output_subdir)
    number_of_captchas = 30000
    data = {}
    with tqdm.tqdm(total=number_of_captchas) as pb:
        for i in range(number_of_captchas):
            img, resolve_text = generate_captcha()
            imname = '{}.jpg'.format(''.join([str(l) for l in resolve_text]))
            if imname in data:
                imname = '{0}_{2}{1}'.format(*splitext(imname), np.random.randint(1000))
            cv2.imwrite(join(output_subdir, imname), img)
            # encoding = get_encoding(resolve_text)
            data[imname] = resolve_text
            pb.update(1)

    samples = [(imname, data[imname]) for imname in data]

    def write_data(output_filename, subdirname, samples):
        with open(output_filename, 'w') as fp:
            for imname, labels in samples:
                # labels_str = ','.join([str(l) for l in labels])
                labels_str = labels
                fp.write('{}/{};{}\n'.format(subdirname, imname, labels_str))

    def split_test_train(samples, test_ratio=0.2):

        n_samples = len(samples)
        n_test = int(n_samples * test_ratio)

        test_idx = np.random.choice(n_samples, n_test, replace=False)
        test_mask = np.zeros(n_samples, dtype=np.bool)
        test_mask[test_idx] = True
        train_mask = ~test_mask
        train_idx = np.nonzero(train_mask)[0]

        test_samples = [samples[idx] for idx in test_idx]
        train_samples = [samples[idx] for idx in train_idx]

        return train_samples, test_samples

    print('total number of samples: {}'.format(len(samples)))
    write_data(join(output_dir, 'samples.txt'), subdirname, samples)

    train_samples, test_samples = split_test_train(samples)
    print('{} = {} (train) + {} (test)'.format(len(samples), len(train_samples), len(test_samples)))

    write_data(join(output_dir, 'train.txt'), subdirname, train_samples)
    write_data(join(output_dir, 'val.txt'), subdirname, test_samples)
    print('all done.')


def generate_one_captcha():
    color_fill = (255, 255, 255)
    color_font = get_random_color()
    width = 200
    height = 70

    captcha, resolve_text = generate_captcha(width=width, height=height,
                                             color_fill=color_fill, color_font=color_font)
    plt.axes()
    plt.axis('equal')
    plt.imshow(captcha)
    plt.show()


if __name__ == '__main__':
    prepare_synthetic_dataset()