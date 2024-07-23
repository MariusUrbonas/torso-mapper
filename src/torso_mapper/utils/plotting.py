import matplotlib.pyplot as plt


def plot_cross_sections(scan_np):
    """
    Plot cross-sections of a 3D scan.

    Parameters:
    - scan_np (numpy.ndarray): The 3D scan data.
    """

    # Plotting cross-sections of scan_np
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    # Plotting axial cross-section
    ax[0].imshow(scan_np[:, :, 32], cmap="gray")
    ax[0].set_title("Axial Cross-Section")

    # Plotting sagittal cross-section
    ax[1].imshow(scan_np[:, 32, :], cmap="gray")
    # Draw a red rectangle on the sagittal cross-section
    ax[1].set_title("Sagittal Cross-Section")

    # Plotting coronal cross-section
    ax[2].imshow(scan_np[32, :, :], cmap="gray")
    ax[2].set_title("Coronal Cross-Section")

    plt.show()
