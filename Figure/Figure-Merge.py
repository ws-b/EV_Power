from PIL import Image

# Load the two images
img1_path = r"C:\Users\BSL\Desktop\Figures\Figure8_EV6_Composite.png"
img2_path = r"C:\Users\BSL\Desktop\Figures\Figure8_Ioniq5_Composite.png"

img1 = Image.open(img1_path)
img2 = Image.open(img2_path)

# Calculate the new size for the stacked image
width = max(img1.width, img2.width)
height = img1.height + img2.height

# Create a new image with the combined height
combined_img = Image.new('RGB', (width, height))

# Paste both images into the new image
combined_img.paste(img1, (0, 0))
combined_img.paste(img2, (0, img1.height))

# Save the result
combined_img_path = r"C:\Users\BSL\Desktop\Figures\figure8.png"
combined_img.save(combined_img_path, dpi = (300, 300))

combined_img.show()
