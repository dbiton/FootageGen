import os
from typing import Dict, List
import tensorflow_hub as tf_hub
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def print_setup() -> None:
    print("reading setup...")
    # Check GPU availability
    gpus = tf.config.list_physical_devices('GPU')
    gpu_available = len(gpus) > 0

    # Print GPU details
    print("GPU Available:", gpu_available)
    if gpu_available:
        gpu_details = [tf.config.experimental.get_device_details(gpu) for gpu in gpus]
        for i, gpu in enumerate(gpus):
            print(f"GPU {i}: {gpu.name}")
            print(f"GPU Details: {gpu_details[i]}")

    # Print TensorFlow version
    print("TensorFlow Version:", tf.__version__)

def is_image(file_path: str) -> bool:
    return os.path.isfile(file_path) and file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff'))

def get_datasets(datasets_dir_path = "./datasets") -> Dict[str, Dict[str, Image.Image]]:
    print("loading datasets...")
    images_dict = {}

    for folder_name in os.listdir(datasets_dir_path):
        folder_path = os.path.join(datasets_dir_path, folder_name)
        if os.path.isdir(folder_path):
            for file_name in os.listdir(folder_path):
               file_path = os.path.join(folder_path, file_name)
               if is_image(file_path):
                img = Image.open(file_path).convert("RGB")
                if folder_name not in images_dict:
                   images_dict[folder_name] = {}
                images_dict[folder_name][file_name] = img
    print("loaded datasets!")
    return images_dict

def resize_image(image: Image.Image, factor: float) -> Image.Image:
    original_width, original_height = image.size
    new_width = max(1, int(original_width * factor))
    new_height = max(1, int(original_height * factor))
    downscaled_img = image.resize((new_width, new_height), Image.BICUBIC)
    return downscaled_img

def process_images(images: List[Image.Image]) -> List[np.array]:
    result = []
    for image in images:
        processed_image = np.array(image)
        processed_image = tf.cast(processed_image, tf.float32)
        processed_image = tf.expand_dims(processed_image, 0)
        result.append(processed_image)
    return result

def main():
    SCALING_FACTOR = 2
    OUTPUT_DIR = "./output"

    print_setup()

    datasets = get_datasets()

    models = {
       "esrgan": "https://tfhub.dev/captain-pool/esrgan-tf2/1",
       "naive": None  # For naive upscaling
    }

    for model_name, model_url in models.items():
        print(f'Evaluating model: {model_name}')
        model_output_dir = os.path.join(OUTPUT_DIR, model_name)

        if model_url is not None:
            model = tf_hub.load(model_url)
        else:
            model = None  # For naive upscaling

        for dataset_name, dataset in datasets.items():
            dataset_output_dir = os.path.join(model_output_dir, dataset_name)
            os.makedirs(dataset_output_dir, exist_ok=True)

            actual_images_dict = dataset
            actual_images = list(actual_images_dict.values())
            filenames = list(actual_images_dict.keys())
            downscaled_images = [resize_image(img, 1 / SCALING_FACTOR) for img in actual_images]

            if model_name == "naive":
                pred_images = [resize_image(img, SCALING_FACTOR) for img in downscaled_images]
                pred_images = [np.array(img) for img in pred_images]
            else:
                # Process images for model input
                downscaled_processed_images = process_images(downscaled_images)
                # Get predictions
                pred_images = [model(i) for i in downscaled_processed_images]
                # Convert predicted tensors to images
                pred_images = [tf.squeeze(img).numpy().astype(np.uint8) for img in pred_images]

            # Compute MSE and save images
            mse_list = []
            for i in range(len(actual_images)):
                actual_img = np.array(actual_images[i])
                pred_img = pred_images[i]

                # Ensure the images are the same size
                if actual_img.shape != pred_img.shape:
                    pred_img = tf.image.resize(pred_img, (actual_img.shape[0], actual_img.shape[1]), method='bicubic').numpy().astype(np.uint8)

                mse = np.mean((actual_img - pred_img) ** 2)
                mse_list.append(mse)

                # Save images
                actual_image_path = os.path.join(dataset_output_dir, f"{filenames[i]}_actual.png")
                pred_image_path = os.path.join(dataset_output_dir, f"{filenames[i]}_pred.png")
                downscaled_image_path = os.path.join(dataset_output_dir, f"{filenames[i]}_downscaled.png")

                Image.fromarray(actual_img).save(actual_image_path)
                Image.fromarray(pred_img).save(pred_image_path)
                Image.fromarray(np.array(downscaled_images[i])).save(downscaled_image_path)

            # Compute average MSE
            avg_mse = np.mean(mse_list)

            print(f"Dataset: {dataset_name}, Model: {model_name}, MSE: {avg_mse}")

if __name__ == "__main__":
    main()
