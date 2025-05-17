import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import lpips  # Import the lpips library for perceptual loss
import torch.nn.functional as F
import numpy as np  # Import NumPy for array operations

# Define the Dataset class
class ImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_paths = [os.path.join(image_dir, img) for img in os.listdir(image_dir) if img.endswith('.png')]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

# Define the Autoencoder architecture
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=False),  # Avoid in-place operation
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=False),  # Avoid in-place operation
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=False),  # Avoid in-place operation
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=False)   # Avoid in-place operation
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=False),  # Avoid in-place operation
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=False),  # Avoid in-place operation
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=False),  # Avoid in-place operation
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Define the UNet architecture
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        self.conv1 = self._create_conv_block(5, 64)
        self.conv2 = self._create_conv_block(64, 128)
        self.conv3 = self._create_conv_block(128, 256)
        self.conv4 = self._create_conv_block(256, 512)
        self.conv5 = self._create_conv_block(512, 1024)
        self.upconv1 = self._create_upsample_block(1024, 512)
        self.process1 = self._process(1024, 512)
        self.upconv2 = self._create_upsample_block(512, 256)
        self.process2 = self._process(512, 256)
        self.upconv3 = self._create_upsample_block(256, 128)
        self.process3 = self._process(256, 128)
        self.upconv4 = self._create_upsample_block(128, 64)
        self.process4 = self._process(128, 64)
        self.final_conv = nn.ConvTranspose2d(64, 3, kernel_size=2, stride=2)

    def _create_conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),  # Avoid in-place operation
            nn.MaxPool2d(2)
        )

    def _create_upsample_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2),
            nn.ReLU(inplace=False),  # Avoid in-place operation
            nn.Conv2d(out_channels, out_channels, kernel_size=3),
            nn.ReLU(inplace=False)   # Avoid in-place operation
        )

    def _process(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.ReLU(inplace=False)  # Avoid in-place operation
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x = self.upconv1(x5)
        x = torch.cat((x, x4), dim=1)
        x = self.process1(x)
        x = self.upconv2(x)
        x = torch.cat((x, x3), dim=1)
        x = self.process2(x)
        x = self.upconv3(x)
        x = torch.cat((x, x2), dim=1)
        x = self.process3(x)
        x = self.upconv4(x)
        x = torch.cat((x, x1), dim=1)
        x = self.process4(x)
        x = self.final_conv(x)
        return x

# Define the LatentDecoder architecture
class LatentDecoder(nn.Module):
    def __init__(self):
        super(LatentDecoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=False),  # Avoid in-place operation
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=False),  # Avoid in-place operation
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=False),  # Avoid in-place operation
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=False),  # Avoid in-place operation
            nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=False),  # Avoid in-place operation
            nn.Conv2d(1024, 2048, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=False)   # Avoid in-place operation
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(2048, 1024, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=False),  # Avoid in-place operation
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=False),  # Avoid in-place operation
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=False),  # Avoid in-place operation
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=False),  # Avoid in-place operation
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=False),  # Avoid in-place operation
            nn.ConvTranspose2d(64, 2, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Function to reshape the encoder output to have 2 channels
def reshape_encoder_output(encoder_output):
    reshaped_output = F.interpolate(encoder_output, size=(512, 512))  # Upsample to (512, 512)
    reshaped_output = reshaped_output[:, :2, :, :]  # Select only the first 2 channels
    return reshaped_output

# Custom LPIPS model that accepts 2-channel inputs
class CustomLPIPS(lpips.LPIPS):
    def forward(self, in0, in1, retPerLayer=False, normalize=False):
        # Adjust to make inputs compatible with 2-channel inputs
        if in0.shape[1] == 2:
            in0 = torch.cat((in0, torch.zeros_like(in0[:, :1, :, :])), dim=1)
        if in1.shape[1] == 2:
            in1 = torch.cat((in1, torch.zeros_like(in1[:, :1, :, :])), dim=1)
        return super().forward(in0, in1, retPerLayer, normalize)

# Example usage
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Initialize and load the pretrained autoencoder
autoencoder = Autoencoder().to(device)
autoencoder.load_state_dict(torch.load('autoencoder_epoch_300.pth'))
autoencoder.eval()  # Set the autoencoder to evaluation mode

# Extract the encoder part of the autoencoder
encoder = autoencoder.encoder

# Initialize models
unet = UNet(5, 3).to(device)
latent_decoder = LatentDecoder().to(device)

# Load Custom LPIPS model
lpips_loss_fn = CustomLPIPS(net='vgg').to(device)

# Load saved models and optimizers
checkpoint_unet = torch.load('unet_model6.pth')
checkpoint_latent_decoder = torch.load('latent_decoder_model6.pth')

unet.load_state_dict(checkpoint_unet['model_state_dict'])
latent_decoder.load_state_dict(checkpoint_latent_decoder['model_state_dict'])

optimizer_unet = optim.Adam(unet.parameters(), lr=0.0001)
optimizer_latent_decoder = optim.Adam(latent_decoder.parameters(), lr=0.0001, weight_decay=1e-5)

# Load the state_dict for optimizers
optimizer_unet.load_state_dict(checkpoint_unet['optimizer_state_dict'])
optimizer_latent_decoder.load_state_dict(checkpoint_latent_decoder['optimizer_state_dict'])

# Define the loss functions
criterion_mse = nn.MSELoss()

# Define the transformations
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor()
])

# Create the dataset and dataloader
dataset = ImageDataset('/home/himanshu/dataset/data/marksheet3', transform=transform)
data_loader = DataLoader(dataset, batch_size=1, shuffle=True)

# Training loop
num_epochs = 500
lpips_loss_weight = 5.0  # Reduce the weight of LPIPS loss
mse_loss_weight = 1.0  # Weight for MSE loss
lpips_loss_weight_latent_decoder = 1.0  # Weight for LPIPS loss in latent decoder
mse_loss_weight_latent_decoder = 1.0  # Weight for MSE loss in latent decoder

torch.autograd.set_detect_anomaly(True)


for epoch in range(num_epochs):
    unet.train()
    latent_decoder.train()
    total_loss_unet = 0.0
    total_loss_latent_decoder = 0.0

    for images in data_loader:
        images = images.to(device)
        # Forward pass through the encoder part of the autoencoder
        with torch.no_grad():
            encoder_output = encoder(images)  # Shape: (1, 512, 32, 32)

        # Reshape the encoder output
        reshaped_encoder_output = reshape_encoder_output(encoder_output)  # Shape: (1, 2, 512, 512)

        # Concatenate the reshaped encoder output with the original image
        combined_input = torch.cat((images, reshaped_encoder_output), dim=1)  # Shape: (1, 5, 512, 512)

        # Forward pass through UNet
        unet_output = unet(combined_input)
        mse_loss_unet = criterion_mse(unet_output, images)
        lpips_loss_unet = lpips_loss_fn(unet_output, images)
        total_loss_unet = mse_loss_weight * mse_loss_unet + lpips_loss_weight * lpips_loss_unet

        # Backward pass and optimization for UNet
        optimizer_unet.zero_grad()
        total_loss_unet.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(unet.parameters(), max_norm=1.0)
        

        # Forward pass through LatentDecoder
        latent_decoder_output = latent_decoder(unet_output)
        latent_decoder_output_resize = latent_decoder_output.reshape(1, 512, 32, 32)
        fout = autoencoder.decoder(latent_decoder_output_resize)
    
        # Compute the MSE loss with resized tensors
        mse_loss_latent_decoder = criterion_mse(fout, images)
        lpips_loss_latent_decoder = lpips_loss_fn(fout, images)
        mse_loss_encoder_decoder = criterion_mse(encoder_output, latent_decoder_output_resize)
        total_loss_latent_decoder = mse_loss_weight_latent_decoder * mse_loss_latent_decoder + \
                                    lpips_loss_weight_latent_decoder * lpips_loss_latent_decoder + \
                                    mse_loss_encoder_decoder

        # Backpropagation and optimization steps
        optimizer_latent_decoder.zero_grad()
        total_loss_latent_decoder.backward()
        torch.nn.utils.clip_grad_norm_(latent_decoder.parameters(), max_norm=1.0)
        optimizer_latent_decoder.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], UNet Total Loss: {total_loss_unet.item():.4f}, LatentDecoder Total Loss: {total_loss_latent_decoder.item():.4f}')

    # Save models and optimizers
    torch.save({
        'model_state_dict': unet.state_dict(),
        'optimizer_state_dict': optimizer_unet.state_dict()
    }, 'unet_modelst.pth')

    torch.save({
        'model_state_dict': latent_decoder.state_dict(),
        'optimizer_state_dict': optimizer_latent_decoder.state_dict()
    }, 'latent_decoder_modelst.pth')

print('Training complete!')
