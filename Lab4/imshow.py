import numpy as np
import matplotlib.pyplot as plt

def load_octave_ascii_mat(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    # Skip header lines until we reach "512 512"
    data_lines = []
    start_reading = False

    for line in lines:
        if not start_reading:
            if line.strip() == "512 512":
                start_reading = True
            continue

        if line.strip() and not line.startswith('#'):
            data_lines.append(line)

    # Load numeric data
    data = np.loadtxt(data_lines, dtype=np.uint8)

    # Reshape into 512x512 image
    image = data.reshape((512, 512))
    image = image.T


    return image

# Load image
image = load_octave_ascii_mat('scrambled_lena.mat')
print("Loaded image shape:", image.shape)

# Display
plt.imshow(image, cmap='gray')
plt.axis('off')
plt.show()
