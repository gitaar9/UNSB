import os
import cv2
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
from typing import Tuple
import numpy as np


def detect_edges(image: np.ndarray, apply_blur: bool) -> np.ndarray:
    """
    Apply edge detection to the given image and return the edge image.

    Args:
        image (np.ndarray): Image read from the input file.
        apply_blur (bool): Whether to apply Gaussian blur.

    Returns:
        np.ndarray: Edge image.
    """
    if apply_blur:
        blurred = cv2.GaussianBlur(image, (3, 3), 0)
    else:
        blurred = image
    edges = cv2.Canny(blurred, 50, 150)
    return edges


def invert_edges(edges: np.ndarray) -> np.ndarray:
    """
    Invert the edges image.

    Args:
        edges (np.ndarray): Edge image.

    Returns:
        np.ndarray: Inverted edge image.
    """
    return cv2.bitwise_not(edges)


def save_edge_image(edge_img: np.ndarray, output_folder: str, image_name: str) -> None:
    """
    Save the edge image to the specified output folder.

    Args:
        edge_img (np.ndarray): Edge image.
        output_folder (str): Path to the output folder.
        image_name (str): Name of the input image file.
    """
    output_path = os.path.join(output_folder, image_name)
    cv2.imwrite(output_path, edge_img)


def plot_images(original_img: np.ndarray, edge_img: np.ndarray, image_name: str, view_plots: bool) -> None:
    """
    Plot both original and edge images side by side.

    Args:
        original_img (np.ndarray): Original image.
        edge_img (np.ndarray): Edge image.
        image_name (str): Name of the input image file.
        view_plots (bool): Whether to display the plots.
    """
    if view_plots:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        axes[1].imshow(edge_img, cmap='gray', vmin=0, vmax=255)
        axes[1].set_title('Edges')
        axes[1].axis('off')
        plt.suptitle(image_name, fontsize=16)
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)
        plt.show()


def main(folder_path: str, apply_blur: bool, output_folder: str, view_plots: bool) -> None:
    """
    Main function to perform edge detection, invert edges, and optionally save or view the results.

    Args:
        folder_path (str): Path to the folder containing the input images.
        apply_blur (bool): Whether to apply Gaussian blur.
        output_folder (str): Path to the folder where the resulting edge images will be saved.
        view_plots (bool): Whether to display the plots.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in tqdm(os.listdir(folder_path)):
        if filename.endswith('.jpg'):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            edge_img = detect_edges(gray_img, apply_blur)
            inverted_edge_img = invert_edges(edge_img)
            output_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_edges.jpg")
            save_edge_image(inverted_edge_img, output_folder, os.path.basename(output_path))
            if view_plots:
                plot_images(img, inverted_edge_img, filename, view_plots)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Edge Detection for Multiple Images")
    parser.add_argument("folder_path", type=str, help="Path to the folder containing the input images")
    parser.add_argument("--blur", action="store_true", help="Apply Gaussian blur (default: False)")
    parser.add_argument("--output_folder", type=str, default="edge_images", help="Path to the folder where the resulting edge images will be saved (default: edge_images)")
    parser.add_argument("--view_plots", action="store_true", help="Display plots (default: False)")
    args = parser.parse_args()
    main(args.folder_path, args.blur, args.output_folder, args.view_plots)
