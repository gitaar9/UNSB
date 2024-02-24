import os
import shutil
import random
import argparse
from typing import Tuple


def create_destination_folders(destination_folder: str) -> None:
    """
    Create destination folders if they don't exist.
    """
    for folder_name in ['trainA', 'trainB', 'testA', 'testB']:
        os.makedirs(os.path.join(destination_folder, folder_name), exist_ok=True)


def split_files(source_folder: str, destination_folder: str, split_ratio: float) -> None:
    """
    Split files from source folder into train and test sets and move them to respective folders.
    """
    files = os.listdir(source_folder)
    random.shuffle(files)
    split_index = int(len(files) * split_ratio)
    train_files = files[:split_index]
    test_files = files[split_index:]

    for file_name in train_files:
        shutil.copy(os.path.join(source_folder, file_name), os.path.join(destination_folder, file_name))
    for file_name in test_files:
        shutil.copy(os.path.join(source_folder, file_name), os.path.join(destination_folder.replace('train', 'test'), file_name))


def split_images_into_folders(source_folder_A: str, source_folder_B: str, destination_folder: str,
                              split_ratio: float = 0.8) -> None:
    """
    Split images from source folders A and B into train and test sets and move them to respective folders.
    """
    create_destination_folders(destination_folder)
    split_files(source_folder_A, os.path.join(destination_folder, 'trainA'), split_ratio)
    split_files(source_folder_B, os.path.join(destination_folder, 'trainB'), split_ratio)


def parse_arguments() -> Tuple[str, str, str, float]:
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Split images from source folders A and B into train and test sets.")
    parser.add_argument("source_folder_A", type=str, help="Path to source folder A containing images")
    parser.add_argument("source_folder_B", type=str, help="Path to source folder B containing images")
    parser.add_argument("destination_folder", type=str,
                        help="Path to destination folder where train and test folders will be created")
    parser.add_argument("--split_ratio", type=float, default=0.8,
                        help="Ratio of images allocated to the training set (default: 0.8)")
    args = parser.parse_args()
    return args.source_folder_A, args.source_folder_B, args.destination_folder, args.split_ratio


if __name__ == "__main__":
    source_folder_A, source_folder_B, destination_folder, split_ratio = parse_arguments()
    split_images_into_folders(source_folder_A, source_folder_B, destination_folder, split_ratio)
