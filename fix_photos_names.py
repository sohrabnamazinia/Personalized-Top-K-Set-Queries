import os

folder_path = "dataset_businesses/businesses_photos"

# Iterate over each file in the folder
for filename in os.listdir(folder_path):
    # Check if the file name ends with '.jpg.jpg'
    if filename.endswith('.jpg.jpg'):
        # Construct the current file path and the new file path
        current_path = os.path.join(folder_path, filename)
        new_filename = filename.replace('.jpg.jpg', '.jpg')
        new_path = os.path.join(folder_path, new_filename)
        
        # Rename the file
        os.rename(current_path, new_path)
        print(f"Renamed: {filename} -> {new_filename}")

print("Renaming complete.")
