from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)

from src.data_loader import load_images
from src.data_loader import add_noise

from src.autoencoder import build_autoencoder

from src.utils import plot_loss


x_train = load_images("data/training")
x_test = load_images("data/testing")

print("Training Shape:", x_train.shape)
print("Testing Shape:", x_test.shape)

print("Train Min:", x_train.min())
print("Train Max:", x_train.max())
print("Train Mean:", x_train.mean())

x_train_noisy = add_noise(
    x_train,
    noise_factor=0.2
)

x_test_noisy = add_noise(
    x_test,
    noise_factor=0.2
)

autoencoder = build_autoencoder()

autoencoder.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["mse"]
)

autoencoder.summary()

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,
    verbose=1
)

checkpoint = ModelCheckpoint(
    "models/best_denoising_autoencoder.keras",
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=3,
    min_lr=1e-6,
    verbose=1
)

history = autoencoder.fit(
    x_train_noisy,
    x_train,
    epochs=50,
    batch_size=128,
    validation_data=(
        x_test_noisy,
        x_test
    ),
    shuffle=True,
    callbacks=[
        early_stopping,
        checkpoint,
        reduce_lr
    ]
)

autoencoder.save(
    "models/final_denoising_autoencoder.keras"
)

plot_loss(
    history,
    "outputs/plots/loss_curve.png"
)

print("\nTraining Complete")
print("Best model saved at:")
print("models/best_denoising_autoencoder.keras")