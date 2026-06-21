import numpy as np
from tensorflow.keras.models import load_model

from src.data_loader import load_images
from src.data_loader import add_noise
from src.utils import visualize_results

x_test = load_images("data/testing")

x_test_noisy = add_noise(
    x_test,
    noise_factor=0.2
)

model = load_model(
    "models/best_denoising_autoencoder.keras"
)

decoded_images = model.predict(x_test_noisy)

visualize_results(
    x_test,
    x_test_noisy,
    decoded_images,
    "outputs/denoised_samples/results.png"
)

mse = np.mean(
    (x_test - decoded_images) ** 2
)

print(f"Reconstruction MSE: {mse:.6f}")