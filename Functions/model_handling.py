# Required packages
import torch
import torch.nn as nn
from torchvision.transforms import ToPILImage
import numpy as np
import cv2
import matplotlib.pyplot as plt


class ConvAutoencoder(nn.Module):
    """
    This class defines the model used for reconstructing images. 
    It is a simple autoencoder structure enhanced by a U-Net-style skip connection that bypasses the most inner layer. 
    This skip connection helps retain spatial details during reconstruction.
    """
    def __init__(self):
        super(ConvAutoencoder, self).__init__()

        # Encoder
        self.enc_conv1 = nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1)   # 256x256 -> 128x128
        self.enc_relu1 = nn.ReLU(True)

        self.enc_conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)  # 128x128 -> 64x64
        self.enc_relu2 = nn.ReLU(True)

        self.enc_conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1) # 64x64 -> 32x32
        self.enc_relu3 = nn.ReLU(True)

        self.enc_conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1) # 32x32 -> 16x16
        self.enc_relu4 = nn.ReLU(True)

        # Decoder
        self.dec_conv1 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1) # 16x16 -> 32x32
        self.dec_relu1 = nn.ReLU(True)

        self.dec_conv2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1) # 32x32 -> 64x64
        self.dec_relu2 = nn.ReLU(True)

        self.dec_conv3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)  # 64x64 -> 128x128
        self.dec_relu3 = nn.ReLU(True)

        self.dec_conv4 = nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1)    # 128x128 -> 256x256
        self.dec_sigmoid = nn.Sigmoid()  # Use sigmoid for normalized RGB

    def forward(self, x):
        # Encoder path with saved activations for skipping connections
        x1 = self.enc_relu1(self.enc_conv1(x))
        x2 = self.enc_relu2(self.enc_conv2(x1))
        x3 = self.enc_relu3(self.enc_conv3(x2))
        x4 = self.enc_relu4(self.enc_conv4(x3))

        # Decoder path skipping one connection
        x = self.dec_relu1(self.dec_conv1(x4) + x3)  # Skip connection from x3
        x = self.dec_relu2(self.dec_conv2(x))
        x = self.dec_relu3(self.dec_conv3(x))
        x = self.dec_sigmoid(self.dec_conv4(x))

        return x



def train_ae(model, trainloader, criterion=nn.MSELoss(), num_epochs=10, device=torch.device("cpu")):
    """
    This function trains the autoencoder using predefined hyperparameters set as default values. 
    During training, it optimizes the model to improve image reconstruction quality. 
    After training, the function saves the model’s weights to the main folder.
    """
    # Define optimizer and send model to device (CPU/GPU)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    for epoch in range(num_epochs):
        for data in trainloader:  # Iterate over data
            img, _, _ = data  # Unpack the image, patient_id, and index
            img = img.to(device)  # Move image to device

            # Forward pass and loss
            output = model(img)
            loss = criterion(output, img)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Print epoch loss
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

    # Save model weights
    torch.save(model.state_dict(), '/kaggle/working/trained_model.pth')



def visualize(dataloader, model, patient_data, num_images=3, mode='simple', label=None, device=torch.device("cpu")):
    """
    This function allows for the visualization of images after they have been reconstructed by the autoencoder. 
    It supports two modes:

        - Simple Mode: Displays the original image alongside its reconstruction for easy comparison.
        - Complex Mode: Provides a more detailed visualization, 
                        showing for both the original and reconstructed images its HSV representation and the red pixels.

    Additionally, the `label` parameter can be used to filter and display images of a specific type.
    """
    # Define HSV masking ranges for red color
    lower_red = np.array([160, 25, 100])
    upper_red = np.array([179, 255, 255])
    model.to(device)

    with torch.no_grad():  # Disable gradient computation for inference
        num_printed = 0
        for batch in dataloader:
            if num_printed >= num_images:
                break  # Stop once we've processed `num_images`

            # Unpack the batch into patch, patient_id, and index
            patches, patient_ids, indexes = batch
            patches = patches.to(device)

            # Perform reconstruction
            inferences = model(patches)
            patches = patches.cpu()
            inferences = inferences.cpu()

            # Iterate patches
            for i in range(len(inferences)):
                if num_printed >= num_images:
                        break  # Stop once we've processed `num_images`
                if label is not None:
                    patient_label = patient_data[patient_ids[i]]['images'][indexes[i]]['label']
                    if (label == 1 and patient_label == -1) or (label == -1 and patient_label == 1):
                        continue # Process only the images of the indicated labels

                # Convert the image tensors to PIL images
                original = ToPILImage()(patches[i])
                reconstructed = ToPILImage()(inferences[i])

                if mode=='simple':
                    # Display
                    fig, axes = plt.subplots(1, 2, figsize=(6, 3))
                    axes[0].imshow(np.asarray(original))
                    axes[0].set_title(f"Original")
                    axes[0].axis("off")
                    axes[1].imshow(np.asarray(reconstructed))
                    axes[1].set_title(f"Reconstructed")
                    axes[1].axis("off")
                    plt.show()

                elif mode=='complex':
                    # Convert to BGR for OpenCV, then to HSV
                    hsv_original = cv2.cvtColor(np.asarray(original), cv2.COLOR_RGB2HSV)
                    hsv_reconstructed = cv2.cvtColor(np.asarray(reconstructed), cv2.COLOR_RGB2HSV)

                    # Define HSV ranges and create red mask
                    mask_original = cv2.inRange(hsv_original, lower_red, upper_red)
                    mask_reconstructed = cv2.inRange(hsv_reconstructed, lower_red, upper_red)

                    # Apply mask to HSV image
                    red_pixels_original = hsv_original.copy()
                    red_pixels_reconstructed = hsv_reconstructed.copy()

                    red_pixels_original[mask_original == 0] = 0
                    red_pixels_reconstructed[mask_reconstructed == 0] = 0

                    # Display
                    titles = ['Original', 'HSV Space', 'Red Pixels', 'Reconstructed', 'HSV Space', 'Red Pixels']
                    images = [original, hsv_original, red_pixels_original, reconstructed, hsv_reconstructed, red_pixels_reconstructed]
                    fig, axs = plt.subplots(2, 3, figsize=(12, 8))
                    for ax, img, title in zip(axs.flatten(), images, titles):
                        ax.imshow(img)
                        ax.set_title(title)
                        ax.axis('off')
                    plt.tight_layout()
                    plt.show()

                num_printed += 1 # Update counter



# Helper function for `patch_features`
def process_img(image, lower_red, upper_red):
    # Convert image to HSV
    hsv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2HSV)

    # Count non-zero pixels in the red mask (indicates red pixels)
    red_pixel_count = cv2.countNonZero(cv2.inRange(hsv_image, lower_red, upper_red))

    return red_pixel_count

def patch_features(dataloader, model, patient_data, device=torch.device("cpu")):
    """
    This function takes patches within a dataloader and passes them through an autoencoder for reconstruction. 
    After reconstructing each patch, it computes three features:

        - Red Pixel Count: The number of red pixels in the original patch.
        - Absolute Pixel Difference: The absolute difference in the number of red pixels between
                                     the original patch and its reconstruction.
        - Percentage Difference: The percentage difference in red pixels between the original and the reconstructed patch.

    Additionally, the helper function `process_img` is provided to count the number of red pixels in an image.
    """
    # Define HSV masking ranges for red color
    lower_red = np.array([160, 25, 100])
    upper_red = np.array([179, 255, 255])
    model.to(device)

    with torch.no_grad():  # Disable gradient computation for inference
        for batch in dataloader:
            # Unpack the batch into patch, patient_id, and index
            patches, patient_ids, indexes = batch
            patches = patches.to(device)

            # Perform reconstruction
            inferences = model(patches)
            patches = patches.cpu()
            inferences = inferences.cpu()

            # Collect reconstructed images along with their patient id and index
            for i in range(len(inferences)):
                # Convert the image tensors to PIL images
                original = ToPILImage()(patches[i])
                reconstructed = ToPILImage()(inferences[i])
                entry = patient_data[patient_ids[i]]['images'][indexes[i]]

                # Initialize 'features' key in entry if it doesn't exist
                entry['features'] = entry.get('features', [])

                # Calculate red pixels for both patches
                orcount = process_img(original, lower_red, upper_red)
                infcount = process_img(reconstructed, lower_red, upper_red)

                # Create patches' features
                entry['features'].append(orcount)                                   # Red pixels in the original image
                entry['features'].append(abs(orcount - infcount))                   # Absolute difference
                entry['features'].append(round((infcount + 1) / (orcount + 1), 4))  # Percentage difference