from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class EnsembleDataset(Dataset):
    def __init__(self, image_paths, radiomic_features_list, labels, device):
        """
        image_paths: List of file paths for mean images
        radiomic_features_list: List of radiomic feature vectors (N, R)
        labels: List of classification labels
        """
        self.image_paths = image_paths
        self.radiomic_features_list = radiomic_features_list
        self.labels = labels
        self.device = device

        # Define image preprocessing for ResNet
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),  # Convert grayscale to 3-channel
            transforms.Resize((224, 224)),  # ResNet requires 224x224 images
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
        ])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Load and preprocess image
        image = Image.open(self.image_paths[idx])
        image = self.transform(image).to(self.device)

        # Load radiomic features
        radiomic_features = torch.tensor(self.radiomic_features_list[idx], dtype=torch.float32).to(self.device)

        # Load label
        label = torch.tensor(self.labels[idx], dtype=torch.long).to(self.device)

        return image, radiomic_features, label
