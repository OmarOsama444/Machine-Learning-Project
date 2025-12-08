import os
import cv2
import numpy as np
import random
from PIL import Image

dataset = "dataset"
augmented = "augmented"
os.makedirs(augmented, exist_ok=True)

def valid_image(path):
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except:
        return False

def augment(img):
    h, w = img.shape[:2]

    scale = random.uniform(0.8, 1.3)   
    scaled = cv2.resize(img, (0, 0), fx=scale, fy=scale)

    if scale > 1:
        nh, nw = scaled.shape[:2]
        start_y = (nh - h) // 2
        start_x = (nw - w) // 2
        img = scaled[start_y:start_y+h, start_x:start_x+w]
    else:
        nh, nw = scaled.shape[:2]
        pad_y = (h - nh) // 2
        pad_x = (w - nw) // 2
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[pad_y:pad_y+nh, pad_x:pad_x+nw] = scaled

    rotations = [90, 180, 270]
    rot_type = random.choice(rotations)
    if rot_type == 90:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif rot_type == 180:
        img = cv2.rotate(img, cv2.ROTATE_180)
    elif rot_type == 270:
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

    flip_type = random.choice([0, 1, -1])
    img = cv2.flip(img, flip_type)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:,:,0] = (hsv[:,:,0] + np.random.uniform(-5, 5)) % 180
    hsv[:,:,1] = np.clip(hsv[:,:,1] + np.random.uniform(-20, 20), 0, 255)
    hsv[:,:,2] = np.clip(hsv[:,:,2] + np.random.uniform(-20, 20), 0, 255)
    img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    std_dev = np.random.uniform(5, 15)
    gaussian = np.random.normal(0, std_dev, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + gaussian, 0, 255).astype(np.uint8)

    return img

for class_name in os.listdir(dataset):
    class_path = os.path.join(dataset, class_name)
    if not os.path.isdir(class_path):
        continue

    save_dir = os.path.join(augmented, class_name)
    os.makedirs(save_dir, exist_ok=True)

    images = [
        f for f in os.listdir(class_path)
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
        and valid_image(os.path.join(class_path, f))
    ]

    original_count = len(images)
    target_aug = 500 - original_count  
    aug_created = 0

    print(f"\nClass '{class_name}': {original_count} valid images found.")
    print(f"Target augmented images: {target_aug}")

    for img_name in images:
        src = os.path.join(class_path, img_name)
        dst = os.path.join(save_dir, img_name)
        cv2.imwrite(dst, cv2.imread(src))


    i = 0  

    while aug_created < target_aug:
        img_name = images[i % original_count]   
        i += 1  

        img_path = os.path.join(class_path, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue

        aug_img = augment(img)
        base = os.path.splitext(img_name)[0]

        save_path = os.path.join(save_dir, f"{base}_aug_{aug_created:03d}.jpg")
        cv2.imwrite(save_path, aug_img)

        aug_created += 1


print("\nAll augmentation completed successfully!")
