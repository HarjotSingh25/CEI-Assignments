import os
import cv2
import numpy as np

IMG_SIZE = 28

def load_images(data_dir):
    images = []

    for label in sorted(os.listdir(data_dir)):
        label_path = os.path.join(data_dir, label)

        if not os.path.isdir(label_path):
            continue

        for img_name in os.listdir(label_path):
            img_path = os.path.join(label_path, img_name)

            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if img is None:
                continue

            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = img.astype("float32") / 255.0

            images.append(img)

    images = np.array(images)
    images = np.expand_dims(images, axis=-1)

    return images


def add_noise(images, noise_factor=0.4):
    noisy_images = images + noise_factor * np.random.normal(
        loc=0.0,
        scale=1.0,
        size=images.shape
    )

    noisy_images = np.clip(noisy_images, 0., 1.)

    return noisy_images