import chromadb
import mimetypes
import os
from typing import List
from io import BytesIO
import math
import base64
import argparse
import tqdm
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader
import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
llm_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY"),
)

data_loader = ImageLoader()
embedding_function = OpenCLIPEmbeddingFunction()

chroma_client = chromadb.PersistentClient(path=".")


def resize_and_convert_to_base64(image_path):
    """
    Resize the image to have a maximum dimension of 800 and convert to base64.
    """
    img = Image.open(image_path)

    width, height = img.size

    if width > height:
        if width > 800:
            new_width = 800
            new_height = int((height * 800) / width)
    else:
        if height > 800:
            new_height = 800
            new_width = int((width * 800) / height)
        else:
            new_width, new_height = width, height

    if new_width != width or new_height != height:
        img = img.resize((new_width, new_height), Image.LANCZOS)

    buffered = BytesIO()
    img.save(buffered, format=img.format if img.format else "PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return img_str


def encode_image_to_data_url(image_path):
    """
    Convert image at image path to data URL that can be sent to the LLM
    """
    # To limit the size of files sent to the LLM, resize before converting to Base64.
    return f"data:image/jpeg;base64,{resize_and_convert_to_base64(image_path)}"


def image_path_to_array(file_path):
    """
    Convert an image file path to a numpy array.

    Parameters:
    -----------
    file_path : str
        Path to the image file

    Returns:
    --------
    numpy.ndarray
        Image as a numpy array with shape (height, width, channels)
    """
    try:
        img = Image.open(file_path)

        img_array = np.array(img)

        return (file_path, img_array)
    except Exception:
        print(f"Unable to convert image to array: {file_path}")
        return None


def get_image_files(directory: str) -> List[str]:
    """
    Recursively find all image files in a directory.

    Args:
        directory (str): Directory path to search

    Returns:
        List[str]: List of absolute paths to image files
    """
    mimetypes.init()

    if not os.path.isdir(directory):
        raise ValueError(f"Directory does not exist: {directory}")

    image_files = []

    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.abspath(os.path.join(root, file))

            mime_type, _ = mimetypes.guess_type(file_path)
            if (
                mime_type
                and mime_type.startswith("image/")
                and (
                    file_path.endswith("jpg")
                    or file_path.endswith("jpeg")
                    or file_path.endswith("png")
                )
            ):
                image_files.append(file_path)

    return image_files


def display_image_grid(
    image_paths, rows=None, cols=None, figsize=(15, 15), titles=None
):
    """
    Display a grid of images from a list of file paths.

    Parameters:
    -----------
    image_paths : list
        List of paths to image files
    rows : int, optional
        Number of rows in the grid. If None, calculated automatically.
    cols : int, optional
        Number of columns in the grid. If None, calculated automatically.
    figsize : tuple, optional
        Figure size (width, height) in inches
    titles : list, optional
        List of titles for each image. If None, no titles are displayed.
    """
    num_images = len(image_paths)

    # Calculate rows and columns if not provided
    if rows is None and cols is None:
        cols = min(4, num_images)  # Default to 4 columns or fewer
        rows = math.ceil(num_images / cols)
    elif rows is None:
        rows = math.ceil(num_images / cols)
    elif cols is None:
        cols = math.ceil(num_images / rows)

    # Create figure and axes
    fig, axes = plt.subplots(rows, cols, figsize=figsize)

    # Make sure axes is always a 2D array
    if rows * cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)

    # Populate the grid with images
    for i, image_path in enumerate(image_paths):
        if i < rows * cols:
            # Calculate row and column index
            row_idx = i // cols
            col_idx = i % cols

            # Read and display the image
            img = Image.open(image_path)
            img = ImageOps.exif_transpose(img)
            axes[row_idx, col_idx].imshow(img)

            # Add title if provided
            if titles is not None and i < len(titles):
                axes[row_idx, col_idx].set_title(titles[i])

            # Remove ticks
            axes[row_idx, col_idx].set_xticks([])
            axes[row_idx, col_idx].set_yticks([])

    # Hide any unused subplots
    for i in range(num_images, rows * cols):
        row_idx = i // cols
        col_idx = i % cols
        axes[row_idx, col_idx].axis("off")

    plt.tight_layout()
    plt.show()


def create_vector_database(folder):
    """
    Create new vector database for all image files in given folder.
    """
    images = get_image_files(folder)
    collection = chroma_client.create_collection(
        name="multimodal_collection",
        embedding_function=embedding_function,
        data_loader=data_loader,
    )
    batch_size = 10
    i = 0
    with tqdm.tqdm(total=len(images)) as progress:
        while i < len(images):
            batch = images[i : min(i + batch_size, len(images))]
            batch_images = [
                image
                for image in map(
                    lambda image_path: image_path_to_array(image_path), batch
                )
                if image
            ]
            collection.add(
                images=[image[1] for image in batch_images],
                ids=[image[0] for image in batch_images],
            )
            i += batch_size
            progress.update(batch_size)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command")
    parser.add_argument("-q", "--query")
    parser.add_argument("-f", "--folder")
    args = parser.parse_args()

    # Create vector database
    if args.command == "create" and args.folder:
        create_vector_database(args.folder)

    # Query existing vector database
    elif args.command == "query" and args.query:
        query = args.query
        collection = chroma_client.get_collection(
            name="multimodal_collection",
            embedding_function=embedding_function,
            data_loader=data_loader,
        )
        results = collection.query(
            query_texts=[query],
            n_results=3,
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": query},
                    *[
                        {"type": "image_url", "image_url": {"url": data_url}}
                        for data_url in map(
                            lambda image: encode_image_to_data_url(image),
                            results["ids"][0],
                        )
                    ],
                ],
            }
        ]
        completion = llm_client.chat.completions.create(
            model="anthropic/claude-3.7-sonnet",
            messages=messages,
        )
        print(completion.choices[0].message.content)
        display_image_grid(results["ids"][0])
    else:
        print("Invalid command")


if __name__ == "__main__":
    main()
