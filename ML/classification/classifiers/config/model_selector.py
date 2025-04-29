import torch
from monai.networks.nets import ResNet, EfficientNetBN, ViT, TorchVisionFCModel, resnet50, ResNetFeatures
import torchvision.models as models
import torch.nn as nn
import monai
import torch.nn.functional as F
from monai.networks.nets import EfficientNetBN, ResNetFeatures
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms


preprocess = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to the input size expected by ResNet
])

class static(nn.Module):
    def __init__(self, num_classes=5):
        super(static, self).__init__()
        # Resnet18 feature extraction, Pytorch
        
        self.fc = nn.Linear(8, num_classes)
        
    def forward(self, x):
        return self.fc(x)

class CNNWeakModel(nn.Module):
    def __init__(self, device, in_channels=1, num_classes=5):
        super(CNNWeakModel, self).__init__()
        # Resnet18 feature extraction, Pytorch
        self.backbone = ResNetFeatures(model_name="resnet18", pretrained=True)
        self.num_layers = 4
        self.hidden_dim = 512
        
        
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        #for param in self.backbone.layer4.parameters():  # Unfreezing Layer4 (ResNet-50/34)
        #    param.requires_grad = True
        
        #for param in self.backbone.layer3.parameters():  # Unfreezing Layer4 (ResNet-50/34)
        #    param.requires_grad = True
        self.fc = nn.Sequential(
            nn.Linear(3072, num_classes),
            #nn.ReLU(),
            #nn.Linear(128, 1)
        )
        self.temporal_pool = nn.AvgPool3d(kernel_size=(3, 1, 1), stride=(1, 1, 1))

        self.to(device)

    def forward(self, images):
        # x is average image'
        batch_size = images.shape[0]
        #https://www.kaggle.com/code/kanncaa1/long-short-term-memory-with-pytorch
        images = images.permute(0,1,3,4,2) #Correct?
        #images = self.temporal_pool(images)
        features = self.backbone(images)
        features = features[-1]
        B, C, D, H, W = features.shape
        features = features.mean(dim=[-1, -2])
        features = features.reshape(B, -1)
        #features = features.reshape(B, -1)
        out = self.fc(features).squeeze(-1) # SQUEEZE regression
        return out, features

class CNNModel(nn.Module):
    def __init__(self, in_channels=1, num_classes=5):
        super(CNNModel, self).__init__()
        # Resnet18 feature extraction, Pytorch
        self.backbone = models.resnet18(pretrained=True)
        # Replace the fully connected (FC) layer with an identity layer
        self.backbone.fc = torch.nn.Identity()  # Removes classification head

        for param in self.backbone.parameters():
            param.requires_grad = False
        
        #for param in self.backbone.layer3.parameters():  # Unfreezing Layer4 (ResNet-50/34)
            #param.requires_grad = True
        
        for param in self.backbone.layer4.parameters():  # Unfreezing Layer4 (ResNet-50/34)
            param.requires_grad = True

        self.fc = nn.Sequential(
            nn.Linear(92160, 64),  # Reduce feature size
            nn.ReLU(),
            nn.Linear(64, num_classes)  # Final classification
        )

    def forward(self, images):
        # x is average image'
        feature_list = []
        images = images.permute(0,1,4,3,2)
        B, C, T, H, W = images.shape
        for i in range(T):
            image = images[:,:,i,:,:]
            image = preprocess(image)
            features = self.backbone(image)
            feature_list.append(features)

        features = torch.stack(feature_list, dim=1)
        features = features.reshape(B, -1)
        #features = self.backbone(images_reshaped)
        #features = features.reshape(B, -1)
        #features = features.view(B, T, -1)  # Shape: (B, T, 512)
        #pooled_features = F.adaptive_avg_pool1d(features.permute(0, 2, 1), 4)  # (B, 512, 30)
        #pooled_features = pooled_features.reshape(pooled_features.shape[0], -1)
        out = self.fc(features)
        return out, features

class CNNTransformerModel(nn.Module):
    def __init__(self, device, in_channels=1, num_classes=5):
        super(CNNTransformerModel, self).__init__()
        # Resnet18 feature extraction, Pytorch
        self.backbone = models.resnet18(pretrained=True)
        # Replace the fully connected (FC) layer with an identity layer
        self.backbone.fc = torch.nn.Identity()  # Removes classification head

        # Freeze feature extractor
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        self.attention = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)

        self.fc = nn.Sequential(
            nn.Linear(30720, 4096),  # Reduce feature size
            #nn.InstanceNorm1d(4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(),
            nn.Linear(4096, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            #nn.LeakyReLU(negative_slope=0.01, inplace=True),
            #nn.Dropout(0.3),
            nn.Linear(256, num_classes)  # Final classification
        )

        self.to(device)

    def forward(self, images):
        # x is average image'
        feature_list = []
        B, C, T, H, W = images.shape
        images_reshaped = images.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)

        features = self.backbone(images_reshaped)
        # Reshape back: (B*T, 512) → (B, T, 512)
        features = features.view(B, T, -1)  # Shape: (B, T, 512)
        pooled_features = F.adaptive_avg_pool1d(features.permute(0, 2, 1), 30)
        pooled_features = pooled_features.reshape(B, 30, 512)
        attn_output, _ = self.attention(pooled_features, pooled_features, pooled_features)
        print(attn_output.shape)
        attn_output = attn_output.reshape(B, -1)
        features = pooled_features.reshape(B, -1)
        print(attn_output.shape)
        print(features.shape)
        features = torch.cat((features, attn_output), dim=1) 
        #features = torch.stack(features, dim=1)  # Shape: [6, iterations * 512]
        #print(features.shape)
        out = self.fc(features)
        return out

class CNNLSTMModel(nn.Module):
    def __init__(self, device, in_channels=1, num_classes=5):
        super(CNNLSTMModel, self).__init__()
        # Resnet18 feature extraction, Pytorch
        # Replace the fully connected (FC) layer with an identity layer
        #self.backbone.fc = torch.nn.Identity()  # Removes classification head

        self.backbone = ResNetFeatures(model_name="resnet18", pretrained=True)
        self.num_layers = 2
        self.hidden_dim = 128
        
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        self.lstm = nn.LSTM(input_size=512, hidden_size=self.hidden_dim, num_layers=self.num_layers, batch_first=True, device=device, dropout=0.1)
    
        self.fc = nn.Sequential(
            nn.Linear(768, 1),  # Reduce feature size
        )

        self.to(device)

    def forward(self, images):
        # x is average image'
        #https://www.kaggle.com/code/kanncaa1/long-short-term-memory-with-pytorch
        batch_size = images.shape[0]
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=images.device).requires_grad_()
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=images.device).requires_grad_()
        
        images = images.permute(0,1,4,3,2)
        #images = self.temporal_pool(images)
        features = self.backbone(images)
        features = features[-1]
        B, C, D, H, W = features.shape
        features = features.mean(dim=[-1, -2])
        features = features.view(B, D, -1)  # Shape: (B, T, 512)
        lstm_out, (hn, cn) = self.lstm(features, (h0.detach(), c0.detach()))
        lstm_out = lstm_out.reshape(B, -1)
        #lstm_out = lstm_out[:, -1, :]
        
        #pooled_features = F.adaptive_avg_pool1d(features.permute(0, 2, 1), 30)
        #features = torch.cat((pooled_features.reshape(B, -1), lstm_out), dim=1)
        #features = torch.cat(feature_list, dim=1)  # Shape: [6, iterations * 512]
        #features = torch.cat((features, lstm_out), dim=1) 
        out = self.fc(lstm_out).squeeze(-1)
        return out, lstm_out

class CNNLSTMModel2(nn.Module):
    def __init__(self, device, in_channels=1, num_classes=5):
        super(CNNLSTMModel2, self).__init__()
        # Resnet18 feature extraction, Pytorch
        self.backbone = ResNetFeatures(model_name="resnet18", pretrained=True)
        self.num_layers = 4
        self.hidden_dim = 512
        
        # Replace the fully connected (FC) layer with an identity layer
        #self.backbone.fc = torch.nn.Identity()  # Removes classification head
        self.mha = nn.MultiheadAttention(embed_dim=25088, num_heads=8, batch_first=True)

        # Freeze feature extractor
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        for param in self.backbone.layer4.parameters():  # Unfreezing Layer4 (ResNet-50/34)
            param.requires_grad = True
        
        for param in self.backbone.layer3.parameters():  # Unfreezing Layer4 (ResNet-50/34)
            param.requires_grad = True

        self.lstm = nn.LSTM(input_size=512, hidden_size=self.hidden_dim, num_layers=self.num_layers, batch_first=True, device=device, dropout=0.2)
        self.attention = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)
        #self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])  # Keep up to last conv layer

        self.fc = nn.Sequential(
            nn.Linear(49152, 2048),  # Reduce feature size
            #nn.InstanceNorm1d(4096),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            #nn.LeakyReLU(negative_slope=0.01, inplace=True),
            #nn.Dropout(0.3),
            nn.Linear(512, num_classes)  # Final classification
        )

        self.to(device)

    def forward(self, images, features):
        # x is average image'
        batch_size = images.shape[0]
        #https://www.kaggle.com/code/kanncaa1/long-short-term-memory-with-pytorch

        batch_size = images.shape[0]
        #https://www.kaggle.com/code/kanncaa1/long-short-term-memory-with-pytorch

        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=images.device).requires_grad_()
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=images.device).requires_grad_()

        features = self.backbone(images)
        features = features[-1]
        batch, channels, depth, height, width = features.shape

        features = features.reshape(batch_size, 512, -1).permute(0,2,1)

        #pooled_features = features.mean(dim=[-1, -2])
        #pooled_features = pooled_features.permute(0, 2, 1)
        #lstm_out, (hn, cn) = self.lstm(pooled_features, (h0.detach(), c0.detach()))
        #lstm_out = torch.mean(lstm_out, dim=1)
        #lstm_out = lstm_out[:,-1,:]
        #pooled_features = pooled_features.reshape(batch_size, -1)
        
        #attn_output, attn_weights = self.mha(features, features, features)
        #attn_output = attn_output.reshape(B, T, F, H, W)
        #attn_output = attn_output.mean(dim=[-1, -2])
        #pooled_features = attn_output.mean(dim=1)
        #B, C, T, H, W = images.shape
        #images_reshaped = images.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)

        #features = self.backbone(images_reshaped)
        #features = features.view(B, T, -1)  # Shape: (B, T, 512)
        #pooled_features = F.adaptive_avg_pool1d(features.permute(0, 2, 1), 30)
        #features = torch.cat((pooled_features.reshape(B, -1), lstm_out), dim=1)
        #features = torch.cat(feature_list, dim=1)  # Shape: [6, iterations * 512]
        #features = torch.cat((features, lstm_out), dim=1) 
        attn_output, _ = self.attention(features, features, features)  # (B, D, embed_dim)
        attn_output = attn_output.permute(0, 2, 1).view(batch, 512, depth, height, width)
        #attn_output = attn_output.mean(dim=[-1,-2])
        attn_output = attn_output.reshape(batch, -1)
        out = self.fc(attn_output)
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
    elif model_name.lower() == "static":
        return static().to(device)
    elif model_name.lower() == "cnnlstm":
        return CNNLSTMModel(device=device).to(device)
    elif model_name.lower() == "cnn":
        return CNNModel().to(device)
    elif model_name.lower() == "cnntransformer":
        return CNNTransformerModel(device=device).to(device)
    elif model_name.lower() == "cnnlstm2":
        return CNNLSTMModel2(device=device).to(device)
    elif model_name.lower() == "cnnweak":
        return CNNWeakModel(device=device).to(device)
    
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