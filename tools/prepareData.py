# Prepare Dataset: Center Crop, Dedup, Split
import os
import shutil
import numpy as np
import imagehash
from PIL import Image, ImageOps, ImageEnhance
from collections import defaultdict
from sklearn.model_selection import train_test_split

# Config
SOURCE_DIR   = '../data/skin_types'
OUTPUT_DIR   = '../data/raw/skin_types'
SPLITS       = {'train': 0.8, 'valid': 0.1, 'test': 0.1}
CLASSES      = ['dry', 'normal', 'oily']
HASH_THRESHOLD = 5


# Center Crop to Square
def center_crop_square(img: Image.Image) -> Image.Image:
    w, h   = img.size
    size   = min(w, h)
    left   = (w - size) // 2
    top    = (h - size) // 2
    return img.crop((left, top, left + size, top + size))


# Normalize Brightness and Saturation
def normalize_for_hash(img: Image.Image) -> Image.Image:
    img = ImageOps.autocontrast(img.convert('RGB'))
    img = ImageEnhance.Color(img).enhance(1.0)
    return img


# Generate 8 Orientations
def get_orientations(img: Image.Image) -> list:
    orientations = []
    for angle in [0, 90, 180, 270]:
        rotated = img.rotate(angle, expand=True)
        orientations.append(rotated)
        orientations.append(ImageOps.flip(rotated))   # vertical flip
    return orientations

# Calculate Set Hash
def compute_hash_set(img: Image.Image) -> set:
    normalized   = normalize_for_hash(img)
    orientations = get_orientations(normalized)
    return {imagehash.phash(o) for o in orientations}

# Detect and Remove Duplicates
def remove_duplicates(image_paths: list) -> list:
    hash_sets  = {}
    duplicates = set()

    for path in image_paths:
        try:
            img      = Image.open(path).convert('RGB')
            h_set    = compute_hash_set(img)
        except Exception as e:
            print(f'[SKIP] {path}: {e}')
            duplicates.add(path)
            continue

        is_dup = False
        for existing_path, existing_hset in hash_sets.items():
            for h_new in h_set:
                for h_old in existing_hset:
                    if h_new - h_old <= HASH_THRESHOLD:
                        is_dup = True
                        break
                if is_dup:
                    break
            if is_dup:
                break

        if is_dup:
            duplicates.add(path)
        else:
            hash_sets[path] = h_set

    print(f'Duplicates removed: {len(duplicates)}')
    return [p for p in image_paths if p not in duplicates]

# Run
def run():
    for cls in CLASSES:
        print(f'\n[{cls}]')
        src_dir = os.path.join(SOURCE_DIR, cls)
        if not os.path.exists(src_dir):
            print(f'  Skip: {src_dir} tidak ditemukan')
            continue

        all_paths  = [os.path.join(src_dir, f) for f in os.listdir(src_dir)
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
        cropped    = []
        tmp_dir    = f'/tmp/cropped_{cls}'
        os.makedirs(tmp_dir, exist_ok=True)

        for path in all_paths:
            try:
                img  = Image.open(path).convert('RGB')
                img  = center_crop_square(img)
                fname = os.path.basename(path)
                dst   = os.path.join(tmp_dir, fname)
                img.save(dst)
                cropped.append(dst)
            except Exception as e:
                print(f'  [SKIP crop] {path}: {e}')

        print(f'  Cropped: {len(cropped)} images')

        clean = remove_duplicates(cropped)
        print(f'  Clean: {len(clean)} images')

        train, temp  = train_test_split(clean, test_size=0.2, random_state=42)
        valid, test  = train_test_split(temp,  test_size=0.5, random_state=42)

        for split_name, split_paths in [('train', train), ('valid', valid), ('test', test)]:
            out_dir = os.path.join(OUTPUT_DIR, split_name, cls)
            os.makedirs(out_dir, exist_ok=True)
            for p in split_paths:
                shutil.copy2(p, os.path.join(out_dir, os.path.basename(p)))
            print(f'  {split_name}: {len(split_paths)}')

        shutil.rmtree(tmp_dir)

    print('\nDone.')

# Init
if __name__ == '__main__':
    run()