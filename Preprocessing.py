import os  

from PIL import Image  
from torch.utils.data import Dataset  
from torchvision import transforms  

class ASLDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform  # Store the transform function
        self.image_paths = []
        self.labels = []

        # Create a mapping for labels: digits 0-9 and letters a-z
        folder_to_label = {str(i): i for i in range(10)}  # 0-9
        folder_to_label.update({letter: 10 + i for i, letter in enumerate("abcdefghijklmnopqrstuvwxyz")})  # a-z

        # Iterate through each folder in the dataset directory
        for letter_folder in os.listdir(root_dir):
            folder_path = os.path.join(root_dir, letter_folder)
            if os.path.isdir(folder_path):
                label = folder_to_label.get(letter_folder, -1)  # Assign label or default to -1 if invalid
                
                if label != -1:
                    # Collect all valid image files in the folder
                    for image_name in os.listdir(folder_path):
                        if image_name.endswith(('.jpeg', '.png')):  # Check for valid image extensions
                            image_path = os.path.join(folder_path, image_name)
                            self.image_paths.append(image_path)
                            self.labels.append(label)
                else:
                    print(f"Warning: Folder '{letter_folder}' does not have a valid label mapping.")

        # Filter out invalid labels (should be between 0 and 34)
        valid_indices = [i for i, label in enumerate(self.labels) if 0 <= label <= 34]
        self.image_paths = [self.image_paths[i] for i in valid_indices]
        self.labels = [self.labels[i] for i in valid_indices]

        # Debugging output to verify dataset loading
        print(f"Total images loaded: {len(self.image_paths)}")
        print(f"Total labels assigned: {len(self.labels)}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        "Loads an image and applies transformations."
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        # Open the image and convert it to RGB
        image = Image.open(image_path).convert('RGB')

        # Apply transformations if provided
        if self.transform:
            image = self.transform(image)

        return image, label

# Define a transformation pipeline for preprocessing images
transform = transforms.Compose([
    transforms.RandomResizedCrop(28, scale=(0.8, 1.0)),  # Random crop and resize to maintain variability
    transforms.Grayscale(num_output_channels=1),  # Convert image to grayscale
    transforms.Resize((28, 28)),  # Ensure uniform size across dataset
    transforms.ToTensor(),  # Convert image to tensor format
    transforms.Normalize((0.5,), (0.5,))  # Normalize pixel values for better model performance
])
