import numpy as np
from os import listdir, makedirs, mkdir, walk, sep
from os.path import join, splitext, isfile, isdir, dirname, abspath, islink


def list_images(base_dir, valid_exts=['.jpg', '.jpeg', '.png', '.bmp', '.ppm']):
    images_list = []
    for f in listdir(base_dir):
        if not isfile(join(base_dir, f)):
            continue
        filext = splitext(f.lower())[1]
        if filext not in valid_exts:
            continue
        images_list.append(f)
    return images_list


def list_subdirs(base_dir):
    subdirs = []
    for f in listdir(base_dir):
        if not isdir(join(base_dir, f)):
            continue
        subdirs.append(f)
    return subdirs


def list_files_with_ext_rec(base_dir, images, valid_exts):
    assert isdir(base_dir), f'{base_dir} is not a valid directory'
    base_path_len = len(base_dir.split(sep))
    for root, dnames, fnames in sorted(walk(base_dir, followlinks=True)):
        root_parts = root.split(sep)
        root_m = sep.join(root_parts[base_path_len:])

        for fname in fnames:
            if not isfile(join(root, fname)):
                continue
            filext = splitext(fname.lower())[1]
            if filext not in valid_exts:
                continue
            path = join(root_m, fname)
            images.append(path)


def list_files_with_ext(base_dir, valid_exts, recursive=False):
    images = []

    if recursive:
        list_files_with_ext_rec(base_dir, images, valid_exts)
    else:
        assert isdir(base_dir) or islink(base_dir), f'{base_dir} is not a valid directory'
        base_path_len = len(base_dir.split(sep))
        for root, dnames, fnames in sorted(walk(base_dir)):
            root_parts = root.split(sep)
            root_m = sep.join(root_parts[base_path_len:])
            for fname in fnames:
                if not isfile(join(root, fname)):
                    continue
                filext = splitext(fname.lower())[1]
                if filext not in valid_exts:
                    continue
                path = join(root_m, fname)
                images.append(path)

    return images