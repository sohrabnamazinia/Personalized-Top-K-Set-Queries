import os

def count_images_in_folder(folder_path="dataset_businesses/businesses_photos"):
    files = os.listdir(folder_path)
    
    image_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    return len(image_files)

image_count = count_images_in_folder()
print(f"Total images in the folder: {image_count}")
