import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import lpips
import os
import matplotlib.pyplot as plt
import torchvision

# Define the Autoencoder class
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),  # Output: 64x256x256
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # Output: 128x128x128
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # Output: 256x64x64
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),  # Output: 512x32x32
            nn.ReLU(inplace=True),
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # Output: 256x64x64
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # Output: 128x128x128
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # Output: 64x256x256
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),  # Output: 3x512x512
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model initialization
autoencoder = Autoencoder().to(device)

# LPIPS loss function
lpips_loss_fn = lpips.LPIPS(net='alex').to(device)

# MSE loss function
mse_criterion = nn.MSELoss()

# Optimizer
optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)

# Custom dataset class
class ImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_paths = [os.path.join(image_dir, img) for img in os.listdir(image_dir) if img.endswith('.png')]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

# Data loader
image_dir = '/home/himanshu/dataset/data/marksheet3'  # Change this to your image directory
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor()
])
dataset = ImageDataset(image_dir, transform=transform)
data_loader = DataLoader(dataset, batch_size=10, shuffle=True)


# Training loop
num_epochs = 300
lpips_weight = 5.0  # Weight for LPIPS loss
for epoch in range(num_epochs):
    for i, images in enumerate(data_loader):
        images = images.to(device)
        
        # Forward pass
        reconstructed_images = autoencoder(images)
        
        # Calculate MSE loss
        mse_loss = mse_criterion(reconstructed_images, images).mean()  # Ensure it's a scalar
        
        # Calculate LPIPS loss
        lpips_loss = lpips_loss_fn(reconstructed_images, images).mean()  # Ensure it's a scalar
        
        # Weighted total loss
        total_loss = mse_loss + lpips_weight * lpips_loss
        
        # Backpropagation and optimization
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # Plotting every 10 batches
        if (i + 1) % 10 == 0:
            # Move images to CPU for plotting
            images = images.cpu().clone().detach().numpy()
            reconstructed_images = reconstructed_images.cpu().clone().detach().numpy()
            
            # Denormalize images for plotting
            images = images.transpose(0, 2, 3, 1)
            reconstructed_images = reconstructed_images.transpose(0, 2, 3, 1)
            
            fig, ax = plt.subplots(2, 10, figsize=(20, 4))
            for j in range(10):
                ax[0, j].imshow((images[j] * 255).astype('uint8'))
                ax[0, j].axis('off')
                ax[1, j].imshow((reconstructed_images[j] * 255).astype('uint8'))
                ax[1, j].axis('off')
            plt.show()
        
    print(f"Epoch [{epoch+1}/{num_epochs}], MSE Loss: {mse_loss.item()}, LPIPS Loss: {lpips_loss.item()}, Total Loss: {total_loss.item()}")
    
    # Save the model
    torch.save(autoencoder.state_dict(), f'autoencoder_epoch_{epoch+1}.pth')

print("Training complete and model saved.")