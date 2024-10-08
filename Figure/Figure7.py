import matplotlib.pyplot as plt
import matplotlib.image as mpimg
def figure3(img1_path, img2_path, save_path, figsize=(6, 6), dpi=300):

    # Load the two images
    img1 = mpimg.imread(img1_path)
    img2 = mpimg.imread(img2_path)

    # Create a figure with two subplots
    fig, axs = plt.subplots(2, 1, figsize=figsize)  # Adjust figure size as needed

    # Display the images in the subplots
    axs[0].imshow(img1)
    axs[1].imshow(img2)

    # Hide axes for both subplots
    for ax in axs:
        ax.axis('off')

    # Add 'A' and 'B' labels to the top-left corner of each subplot
    axs[0].text(-0.1, 1.1, 'A', transform=axs[0].transAxes, fontsize=11, fontweight='bold', va='top', ha='right')
    axs[1].text(-0.1, 1.1, 'B', transform=axs[1].transAxes, fontsize=11, fontweight='bold', va='top', ha='right')

    # Save the final image
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi)
    plt.show()

img1_path = r"C:\Users\BSL\Desktop\Figures\Result\shap_values_EV6.png"
img2_path = r"C:\Users\BSL\Desktop\Figures\Result\shap_values_Ioniq5.png"
save_path = r'C:\Users\BSL\Desktop\Figures\figure7.png'

figure3(img1_path, img2_path, save_path)