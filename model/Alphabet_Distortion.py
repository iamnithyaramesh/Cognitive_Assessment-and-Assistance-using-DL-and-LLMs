from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
import os

import os
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont


def create_clean_letter(letter, img_size=64, font_size=60, font_path="arial.ttf", distort=False):
        img = Image.new('L', (img_size, img_size), color=255)  # Ensure pure white background
        d = ImageDraw.Draw(img)
    
        try:
            font = ImageFont.truetype(font_path, size=font_size)
        except IOError:
            print(f"Error: Font file not found at {font_path}. Using default font instead.")
            font = ImageFont.load_default()

        bbox = d.textbbox((0, 0), letter, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        x = (img_size - text_width) / 2
        y = (img_size - text_height) / 2 - bbox[1]

        d.text((x, y), letter, fill=0, font=font)  # Ensure solid black text

        img_array = np.array(img)

        if distort:
            img_array = apply_distortion(img_array)

    # Ensure binary black-and-white output
        img_array = np.where(img_array > 128, 255, 0).astype(np.uint8)

        return img_array

def apply_distortion(img_array):
    """
    Applies distortion effects while ensuring the background remains white.

    Args:
        img_array: A NumPy array representing the image.

    Returns:
        A distorted version of the image.
    """
    rows, cols = img_array.shape

    # 1. Ensure background is white
    img_array[img_array > 0] = 255  # Convert all non-black pixels to white

    # 2. Pixel Dropout (Avoid Black Artifacts)
    dropout_mask = np.random.choice([255, 0], size=(rows, cols), p=[0.75, 0.25])  # 25% dropout
    img_array = np.maximum(img_array, dropout_mask)  # Keep the brightest pixels

    # 3. Add Random White Squares
    num_blocks = 30  # Number of random white blocks to add
    block_size = rows // 10  # Size of each block (adjust as needed)
    for _ in range(num_blocks):
        bx = np.random.randint(0, cols - block_size)
        by = np.random.randint(0, rows - block_size)
        img_array[by:by + block_size, bx:bx + block_size] = 255  # Add a white block

    return img_array


def save_letter_images(output_dir="letter_images", img_size=64, font_size=60, font_path="arial.ttf", distort=True):
    """
    Generates and saves images for all letters A to Z, with optional distortion.

    Args:
        output_dir: The directory to save the images.
        img_size: The size of the image (both width and height).
        font_size: The font size used to render the letter.
        font_path: The path to the .ttf font file.
        distort: Whether to apply distortion to the letters.
    """

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i in range(65, 91):  # ASCII for A-Z
        letter = chr(i)
        letter_image = create_clean_letter(letter, img_size, font_size, font_path, distort=distort)

        if letter_image is not None:
            filename = os.path.join(output_dir, f"{letter}.png")
            cv2.imwrite(filename, letter_image, [cv2.IMWRITE_PNG_COMPRESSION, 0])

            print(f"Saved {filename}")
        else:
            print(f"Failed to create image for letter {letter}")


def question():
    import random
    # Specify the directory containing images
    image_folder = "letter_images"
    # Check if the directory exists
    if not os.path.exists(image_folder):
        print(f"Directory {image_folder} does not exist.")
        return save_letter_images(img_size=512, font_size=500, font_path="arial.ttf", distort=True)
        
    # Get a list of all image files in the directory
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png'))]

    # Check if there are any images in the folder
    if image_files:
        # Select a random image
        random_image = random.choice(image_files)
        print(f"Selected Image: {random_image}")
        # Load the image using OpenCV
        image_path = os.path.join(image_folder, random_image)
        image = cv2.imread(image_path)

        # Display the image in a window
        cv2.imshow("Guess the Letter", image)
        cv2.waitKey(1)  # Display the image briefly

        # Generate quiz options
        correct_letter = random_image.split('.')[0]  # Extract the letter from the filename
        options = [correct_letter]
        while len(options) < 4:
            random_letter = chr(random.randint(65, 90))  # Random letter A-Z
            if random_letter not in options:
                options.append(random_letter)
        random.shuffle(options)

        # Display the quiz
        print("What letter is displayed in the image?")
        for i, option in enumerate(options):
            print(f"{i + 1}. {option}")

        # Get the user's answer
        try:
            user_choice = int(input("Enter the number corresponding to your choice: "))
            if options[user_choice - 1] == correct_letter:
                print("Correct!")
            else:
                print(f"Wrong! The correct answer was {correct_letter}.")
        except (ValueError, IndexError):
            print(f"Invalid input. The correct answer was {correct_letter}.")

        # Close the image window
        cv2.destroyAllWindows()
    else:
        print(f"No images found in the directory {image_folder}.")

