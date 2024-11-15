from PIL import Image
import matplotlib.pyplot as plt

# Path to the image
image_path = "dataset_businesses/businesses_photos/Ll5cXzRW1xGRsyFKhUSHrg.jpg"

# Open the image
image = Image.open(image_path)

# Display the image using matplotlib
plt.imshow(image)
plt.axis('off')  # Hide axes
plt.show()
