import os
import glob
import pandas as pd

def count_images_in_subfolders(root_folder, image_extensions=['jpg', 'jpeg', 'png', 'gif', 'bmp']):
    """
    Count the number of images in each subfolder of a root folder.

    Args:
    root_folder (str): The path to the root folder.
    image_extensions (list): List of image extensions to consider. Default is ['jpg', 'jpeg', 'png', 'gif', 'bmp'].

    Returns:
    pandas.DataFrame: A DataFrame containing the count of images in each subfolder.
    """
    # Ensure the root folder path is valid
    if not os.path.exists(root_folder):
        print(f"Error: The folder '{root_folder}' does not exist.")
        return None
    
    # Initialize an empty list to store results
    data = []

    # Iterate through subfolders
    for subfolder_name in os.listdir(root_folder):
        subfolder_path = os.path.join(root_folder, subfolder_name)

        # Check if it's a directory
        if os.path.isdir(subfolder_path):
            # Use glob to count images in the subfolder
            image_files = []
            for ext in image_extensions:
                image_files.extend(glob.glob(os.path.join(subfolder_path, f'*.{ext}')))

            # Count the number of images
            num_images = len(image_files)

            # Append the results to the list
            data.append({'Classes': subfolder_name, 'Number_Images': num_images})

    # Convert the list to a DataFrame
    df = pd.DataFrame(data)

    return df
