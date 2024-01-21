# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


import numpy as np
import pywt
from skimage import io
from skimage.util import img_as_float
from skimage.filters import gaussian
from matplotlib import pyplot as plt
from numpy.fft import fft2, ifft2, fftshift

def load_image(path):
    return io.imread(path)

# Funkcje do odszumiania za pomocą transformacji Fouriera
def apply_fourier_transform(image):
    return fftshift(fft2(image))

def inverse_fourier_transform(image):
    return ifft2(fftshift(image))

def denoise_image_in_frequency_domain(image, threshold=10):
    f_transform = apply_fourier_transform(image)
    magnitude_spectrum = np.abs(f_transform)
    f_transform[magnitude_spectrum < threshold] = 0
    return inverse_fourier_transform(f_transform)

# Funkcje do odszumiania za pomocą transformacji falkowej
def wavelet_denoise_channel(channel, wavelet='db1', level=1):
    coeffs = pywt.wavedec2(channel, wavelet, level=level)
    threshold = 0.8

    new_coeffs = []
    for coeff in coeffs:
        if isinstance(coeff, tuple):
            new_coeff = tuple(pywt.threshold(subband, value=threshold, mode='soft') for subband in coeff)
            new_coeffs.append(new_coeff)
        else:
            new_coeffs.append(pywt.threshold(coeff, value=threshold, mode='soft'))

    return pywt.waverec2(new_coeffs, wavelet)

def wavelet_denoise_image(image, wavelet='db1', level=1):
    channels = []
    for i in range(3):
        channel = image[:, :, i]
        channel_denoised = wavelet_denoise_channel(channel, wavelet=wavelet, level=level)
        channel_denoised = np.clip(channel_denoised, 0, 1)  # Zapewnienie, że wartości pozostają w zakresie [0, 1]
        channels.append(channel_denoised)

    denoised_image = np.stack(channels, axis=2)
    return denoised_image.astype(image.dtype)  # Konwersja typu danych z powrotem do oryginalnego typu obrazu wejściowego


# Odszumianie za pomocą rozmycia Gaussowskiego
def gaussian_blur(image, sigma=1):
    return gaussian(image, sigma=sigma)

def add_salt_and_pepper_noise(image, amount=0.05):
    row, col, ch = image.shape
    s_vs_p = 0.5
    out = np.copy(image)

    # Salt mode
    num_salt = np.ceil(amount * image.size * s_vs_p)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    out[coords[0], coords[1], :] = 1

    # Pepper mode
    num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    out[coords[0], coords[1], :] = 0

    return out

# Funkcja do przetwarzania kanałów obrazu
def process_image_channels(image, denoise_function):
    red_channel = denoise_function(image[:, :, 0])
    green_channel = denoise_function(image[:, :, 1])
    blue_channel = denoise_function(image[:, :, 2])
    return np.stack([np.abs(red_channel), np.abs(green_channel), np.abs(blue_channel)], axis=2)

# Zmodyfikuj ścieżkę do obrazu
image_path = "C:/Users/Szymon Nowicki/Desktop/CFA/Bayer/pandaczerwona.jpg"
image = load_image(image_path)
image = img_as_float(image)  # Konwersja do formatu zmiennoprzecinkowego

# Dodanie szumu 'sól i pieprz' do obrazu
noisy_image = add_salt_and_pepper_noise(image)

# Odszumianie za pomocą transformacji Fouriera
fourier_denoised = process_image_channels(image, lambda x: denoise_image_in_frequency_domain(x, threshold=10))
fourier_denoised = fourier_denoised.clip(0, 1)

# Odszumianie za pomocą rozmycia Gaussowskiego
gaussian_blurred = gaussian_blur(image, sigma=1)

# Odszumianie za pomocą transformacji falkowej
wavelet_denoised = wavelet_denoise_image(image, wavelet='db1', level=1)
wavelet_denoised = wavelet_denoised.clip(0, 1)


# Wyświetlenie obrazów
plt.figure(figsize=(24, 10))
plt.subplot(151), plt.imshow(image), plt.title('Original Image')
plt.subplot(152), plt.imshow(noisy_image), plt.title('Noisy Image')
plt.subplot(153), plt.imshow(fourier_denoised), plt.title('Fourier Denoising')
plt.subplot(154), plt.imshow(gaussian_blurred), plt.title('Gaussian Blur')
plt.subplot(155), plt.imshow(wavelet_denoised), plt.title('Wavelet Denoising')
plt.show()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
