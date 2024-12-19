import numpy as np
import imageio.v2 as imageio
import matplotlib.pyplot as plt

# baca gambar original citra rendah
image = imageio.imread('lady_low_contrast_image.jpg') 

# buat fungsi Histogram Equalization
def histogram_equalization(image):
    # Hitung histogram
    histogram, bins = np.histogram(image.flatten(), bins=256, range=[0, 256])
    
    # Hitung CDF (Cumulative Distribution Function)
    cdf = histogram.cumsum()
    cdf_normalized = cdf * 255 / cdf[-1]  # Normalisasi CDF

    # Lakukan equalization
    image_equalized = np.interp(image.flatten(), bins[:-1], cdf_normalized)
    return image_equalized.reshape(image.shape).astype(np.uint8)

# Terapkan histogram equalization
image_equalized = histogram_equalization(image)

# Fungsi Peningkatan Kontras dengan Faktor
def contrast_stretching(image, level):
    # Terapkan peningkatan kontras
    image_contrast = np.clip(image * level, 0, 255)
    return image_contrast.astype(np.uint8)

# Peningkatan kontras dengan level 1.5
contrast_level = 1.5
image_contrast = contrast_stretching(image, contrast_level)

# Tampilkan hasil perbandingan
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.title("Citra Asli")
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title(f"Citra Contrast Level {contrast_level}")
plt.imshow(image_contrast, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("Citra Histogram Equalization")
plt.imshow(image_equalized, cmap='gray')
plt.axis('off')

plt.show()
