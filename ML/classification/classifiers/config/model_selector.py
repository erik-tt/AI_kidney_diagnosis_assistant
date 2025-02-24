import torch
from monai.networks.nets import ResNet, EfficientNetBN, ViT, TorchVisionFCModel, resnet50
import torchvision.models as models
import torch.nn as nn
import monai
import torch.nn.functional as F
from monai.networks.nets import EfficientNetBN, ResNetFeatures
import torch
import torch.nn as nn
import torch.nn.functional as F

class ensembleModel(nn.Module):
    def __init__(self, device, in_channels=1, num_classes=5):
        super(ensembleModel, self).__init__()
        # Resnet18 feature extraction, Pytorch
        self.backbone = models.resnet152(pretrained=True)

        # Replace the fully connected (FC) layer with an identity layer
        self.backbone.fc = torch.nn.Identity()  # Removes classification head

        # Freeze feature extractor
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        self.fc = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.InstanceNorm1d(2048), ## Less overfit compared to batch norm
            nn.LeakyReLU(negative_slope=0.01, inplace=True),  # LeakyReLUs instead of ReLU
            nn.Dropout(0.5),  # Dropout for regularization
            nn.Linear(2048, num_classes)  # Output layer
        )

        self.to(device)

    def forward(self, x):
        # x is average image
        features = self.backbone(x)
        out = self.fc(features)
        return out


# CHAT <3
class Simple2DCNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=5, dropout_prob=0.2):
        super(Simple2DCNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(512)

        self.dropout = nn.Dropout2d(p=dropout_prob)  # Dropout for regularization

        # Global Average Pooling instead of Flattening
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.InstanceNorm1d(512),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),  # LeakyReLUs instead of ReLU
            nn.Dropout(0.3),  # Dropout for regularization
            nn.Linear(512, num_classes)  # Output layer
        )

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)  # Apply dropout after activation

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.dropout(x)

        x = F.relu(self.bn4(self.conv4(x)))
        x = self.dropout(x)

        x = F.relu(self.bn5(self.conv5(x)))
        x = self.dropout(x)

        x = self.global_pool(x)  # Shape: (B, 128, 1, 1, 1)
        x = torch.flatten(x, 1)  # Shape: (B, 128)

        out = self.fc(x)  # Shape: (B, num_classes)
        return out


class Simple3DCNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=5, dropout_prob=0.3):
        super(Simple3DCNN, self).__init__()

        self.conv1 = nn.Conv3d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm3d(32)
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm3d(64)
        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm3d(128)

        self.dropout = nn.Dropout3d(p=dropout_prob)  # Dropout for regularization

        # Global Average Pooling instead of Flattening
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(128, 128),
            nn.InstanceNorm1d(128),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),  # LeakyReLUs instead of ReLU
            nn.Dropout(0.4),  # Dropout for regularization
            nn.Linear(128, num_classes)  # Output layer
        )

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)  # Apply dropout after activation

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.dropout(x)

        x = self.global_pool(x)  # Shape: (B, 128, 1, 1, 1)
        x = torch.flatten(x, 1)  # Shape: (B, 128)

        out = self.fc(x)  # Shape: (B, num_classes)
        return out



class CNN_LSTM(nn.Module):
    def __init__(self, device, hidden_dim=128, num_layers=2, num_classes=5):
        super(CNN_LSTM, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        # Resnet 50 backbone

        self.cnn = EfficientNetBN(
            model_name="efficientnet-b0",  # Choose b0, b1, etc.
            in_channels=1,  # Grayscale input
            pretrained=True,
            num_classes=5
        ).to(device)


        self.cnn.fc = nn.Identity()

        self.lstm = nn.LSTM(input_size=1000, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, device=device, dropout=0.2)

        self.fc = nn.Linear(hidden_dim, num_classes)

    
    def forward(self, x):
        batch_size, seq_len, C, H, W = x.shape
        # https://github.com/ritchieng/deep-learning-wizard/blob/master/docs/deep_learning/practical_pytorch/pytorch_lstm_neuralnetwork.md

        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)

        features = []

        # Process som batch * time ... https://discuss.pytorch.org/t/how-to-input-image-sequences-to-a-cnn-lstm/89149
        for t in range(seq_len):
            feature_t = self.cnn(x[:, t, :, :, :])
            features.append(feature_t)  

        features = torch.stack(features, dim=1)

        lstm_out, (hn, cn) = self.lstm(features, (h0, c0))
        #print(lstm_out[:, -1, :].shape)
        out = self.fc(lstm_out[:, -1, :]) 

        #print(out)
        return out

# Define a 3D group convolution
class GroupConv3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, group_size=8):
        super().__init__()
        self.group_size = group_size  # Number of transformations
        self.conv3d = nn.Conv3d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)

    def forward(self, x):
        rotated_outputs = []
        
        # Apply different transformations for equivariance
        for i in range(self.group_size):
            if i % 2 == 0:  # Flip along spatial dimensions
                transformed_x = torch.flip(x, dims=[2])  # Flip along depth
            else:
                transformed_x = torch.flip(x, dims=[3])  # Flip along height
                
            rotated_outputs.append(self.conv3d(transformed_x))

        # Average over all transformations
        output = sum(rotated_outputs) / self.group_size
        return output


class GCNN3DResNet(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.gconv1 = GroupConv3D(in_channels=1, out_channels=64)
        self.gconv2 = GroupConv3D(in_channels=64, out_channels=128)
        self.gconv3 = GroupConv3D(in_channels=128, out_channels=256)
        self.gconv4 = GroupConv3D(in_channels=256, out_channels=512)

        # ResNet backbone (using MONAI)
        self.resnet = ResNet(
            block="basic",
            num_classes=num_classes,
            n_input_channels=512,  # Match last GCNN layer output
            layers=[2, 2, 2, 2],
            block_inplanes=[64, 128, 256, 512],  
            spatial_dims=3
        )

    def forward(self, x):
        x = F.relu(self.gconv1(x))
        x = F.relu(self.gconv2(x))
        x = F.relu(self.gconv3(x))
        x = F.relu(self.gconv4(x))

        # Pass through standard ResNet layers
        x = self.resnet(x)
        return x



def get_mobilenetv3():
    #Testing with small model first
    mobilenet_v3 = models.mobilenet_v3_small()

    #Grayscale config (prompted chatgpt)
    original_first_layer = mobilenet_v3.features[0][0]
    
    mobilenet_v3.features[0][0] = nn.Conv2d(
        in_channels=1,
        out_channels=original_first_layer.out_channels,
        kernel_size=original_first_layer.kernel_size,
        stride=original_first_layer.stride,
        padding=original_first_layer.padding,
        bias=False
    ).to(device)
    return mobilenet_v3

def model_selector(model_name :str, device: torch.device):


    #TODO: make this 3D with dicom
    if model_name.lower() == "3dresnet":
        #Resnet18 config
        model = ResNet(
            block="basic",
            num_classes = 5,
            n_input_channels = 1,
            layers = [2, 2, 2, 2],
            block_inplanes=[64, 128, 256, 512],
            spatial_dims=3
        ).to(device)
        return model
    
    elif model_name.lower() == "resnet18":
        model = TorchVisionFCModel(
            model_name='resnet18',
            num_classes=5,
            pretrained=True
        ).to(device)

        return model
    
    elif model_name.lower() == "resnet50":
        model = TorchVisionFCModel(
            model_name='resnet50',
            num_classes=5,
            pretrained=True
        ).to(device)
        return model
    
    elif model_name.lower() == "efficientnet":
        model = EfficientNetBN(
            model_name="efficientnet-b0",
            in_channels=1,
            pretrained=False
        ).to(device)
        return model
    elif model_name.lower() == "gcnn":
        model = GCNN3DResNet().to(device)
        return model
        
    elif model_name.lower() == "lstm":
        model = CNN_LSTM(device=device).to(device)
        return model

    elif model_name.lower() == "simple3d":
        model = Simple3DCNN().to(device)
        return model
    elif model_name.lower() == "simple2d":
        model = Simple2DCNN().to(device)
        return model
    elif model_name.lower() == "mobilenetv3":
        return get_mobilenetv3()
    elif model_name.lower() == "ensemble":
        return ensembleModel(device=device).to(device)
    
    #Improve the ViT before further use to make sure it is configured properly
    #Can be configured to 3D for temporal dimension (but should look into ViViT)
    elif (model_name.lower()) == "vit" or (model_name.lower() == "vision_transformer"):
        model = ViT(
            in_channels=1,
            img_size= (96, 96, 180),
            patch_size= (8,8,8),
            spatial_dims=3,
            hidden_size=768,
            num_classes=5
        ).to(device)
        return model
    
    else:
        raise ValueError(f"Unkown classifier, check spelling of model in model_selector.py")