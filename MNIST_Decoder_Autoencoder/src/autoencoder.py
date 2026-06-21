from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    UpSampling2D
)

from tensorflow.keras.models import Model


def build_autoencoder():

    input_img = Input(
        shape=(28, 28, 1)
    )

    x = Conv2D(
        32,
        (3, 3),
        activation="relu",
        padding="same"
    )(input_img)

    x = MaxPooling2D(
        (2, 2),
        padding="same"
    )(x)

    x = Conv2D(
        64,
        (3, 3),
        activation="relu",
        padding="same"
    )(x)

    encoded = MaxPooling2D(
        (2, 2),
        padding="same"
    )(x)

    x = Conv2D(
        128,
        (3, 3),
        activation="relu",
        padding="same"
    )(encoded)

    x = Conv2D(
        128,
        (3, 3),
        activation="relu",
        padding="same"
    )(x)

    x = UpSampling2D(
        (2, 2)
    )(x)

    x = Conv2D(
        64,
        (3, 3),
        activation="relu",
        padding="same"
    )(x)

    x = UpSampling2D(
        (2, 2)
    )(x)

    x = Conv2D(
        32,
        (3, 3),
        activation="relu",
        padding="same"
    )(x)

    decoded = Conv2D(
        1,
        (3, 3),
        activation="sigmoid",
        padding="same"
    )(x)

    autoencoder = Model(
        input_img,
        decoded
    )

    return autoencoder