import matplotlib.pyplot as plt


def plot_loss(history, save_path):

    plt.figure(figsize=(8, 5))

    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])

    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.legend([
        "Train",
        "Validation"
    ])

    plt.savefig(save_path)

    plt.close()


def visualize_results(
    original,
    noisy,
    denoised,
    save_path
):

    n = 10

    plt.figure(figsize=(20, 6))

    for i in range(n):

        ax = plt.subplot(
            3,
            n,
            i + 1
        )

        plt.imshow(
            original[i].squeeze(),
            cmap="gray"
        )

        plt.axis("off")

        ax = plt.subplot(
            3,
            n,
            i + n + 1
        )

        plt.imshow(
            noisy[i].squeeze(),
            cmap="gray"
        )

        plt.axis("off")

        ax = plt.subplot(
            3,
            n,
            i + 2 * n + 1
        )

        plt.imshow(
            denoised[i].squeeze(),
            cmap="gray"
        )

        plt.axis("off")

    plt.savefig(save_path)
    plt.close()