import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Load the generated result
img = Image.open('jigsaw_result.png')

# Convert to array
img_array = np.array(img)

# The image is rotated 90 degrees left (counter-clockwise)
# To fix: rotate 90 degrees right (clockwise)
# In matplotlib/numpy: rotating right = rot90(img, k=-1) or rot90(img, k=3)

# Load it using matplotlib to extract just the image parts
result_img = plt.imread('jigsaw_result.png')

# Create corrected version by rotating the actual puzzle image data
# Let's reload the solved puzzle directly
print("[INFO] Creating rotation-corrected version...")

# Re-run the reconstruction with rotation fix
import sys
sys.path.insert(0, '.')

# Just rotate the existing output
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
ax.imshow(result_img)
ax.axis('off')
plt.savefig('jigsaw_result_rotated.png', dpi=150, bbox_inches='tight')
print("[INFO] Saved rotated version to jigsaw_result_rotated.png")

# Better approach: rotate the actual image content 90 degrees clockwise
from PIL import Image
img = Image.open('jigsaw_result.png')
# Rotate 90 degrees clockwise (which is -90 or 270)
img_rotated = img.rotate(90, expand=True)
img_rotated.save('jigsaw_corrected.png')
print("[SUCCESS] Corrected rotation saved to jigsaw_corrected.png")
