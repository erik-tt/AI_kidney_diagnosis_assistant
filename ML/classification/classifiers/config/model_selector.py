import torch
from monai.networks.nets import ResNet, EfficientNetBN, ViT, TorchVisionFCModel, resnet50, ResNetFeatures
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms


class CNNWeakRadiomics(nn.Module):
    def __init__(self, num_classes=5):
        super(CNNWeakRadiomics, self).__init__()
        self.backbone = ResNetFeatures(model_name="resnet18", pretrained=True)
        
        # FREEZE LAYERS
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        ## UNFREEZE LAYER 3 & 4
        for layer in [self.backbone.layer3, self.backbone.layer4]:
            layer.train()  # Important: puts BatchNorm in training mode
            for param in layer.parameters():
                param.requires_grad = True
        
        self.fc = nn.Sequential(
            nn.Linear(3918, num_classes),
            #nn.ReLU(),
            #nn.Linear(128, 1)
        )


    def forward(self, image, radiomics):
        # x is average image'
        batch_size = image.shape[0]
        #https://www.kaggle.com/code/kanncaa1/long-short-term-memory-with-pytorch
        images = image.permute(0,1,3,2,4)
        
        features = self.backbone(images)
        features = features[-1]
        B, C, D, H, W = features.shape
        features = features.mean(dim=[-1, -2])
        features = features.reshape(B, -1)
        features = features.float()
        radiomics = radiomics.float()

        all_feats = torch.cat([features, radiomics], dim=1)
        out = self.fc(all_feats).squeeze(-1) # SQUEEZE regression
        # RETURNERE RADIOMIC FEATURES HERE?
        return out, features
    
class CNNWeakModel(nn.Module):
    def __init__(self, num_classes=5):
        super(CNNWeakModel, self).__init__()
        # Resnet18 feature extraction, Pytorch
        self.backbone = ResNetFeatures(model_name="resnet18", pretrained=True)
        
        # FREEZE LAYERS
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        ## UNFREEZE LAYER 3 & 4
        for layer in [self.backbone.layer3, self.backbone.layer4]:
            layer.train()  # Important: puts BatchNorm in training mode
            for param in layer.parameters():
                param.requires_grad = True
        
        self.fc = nn.Sequential(
            nn.Linear(3072, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, images):
        #https://www.kaggle.com/code/kanncaa1/long-short-term-memory-with-pytorch
        images = images.permute(0,1,3,2,4)

        
        #images = self.temporal_pool(images)
        features = self.backbone(images)
        features = features[-1]
        B, C, D, H, W = features.shape
        features = features.mean(dim=[-1, -2])
        features = features.reshape(B, -1)
        #features = features.reshape(B, -1)
        out = self.fc(features).squeeze(-1) # SQUEEZE regression
        return out, features

class Resnet18Radiomics(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        
        self.backbone = TorchVisionFCModel(
            model_name='resnet18',
            num_classes=num_classes,
            pretrained=True
        )

        self.backbone_fc = self.backbone.fc  
        self.backbone.fc = nn.Identity()     

        self.fc = nn.Linear(
            self.backbone_fc.in_features + 846,  # 846 is RADIOMICS DIM
            num_classes
        )

    def forward(self, image, radiomics):

        image_features = self.backbone(image)

        radiomics = radiomics.float()
        image_features = image_features.float()

        combined = torch.cat((image_features, radiomics), dim=1) 

        return self.fc(combined)
    
class CNNTransformerModel(nn.Module):
    def __init__(self, device, in_channels=1, num_classes=5):
        super(CNNTransformerModel, self).__init__()

        self.backbone = models.resnet18(pretrained=True)
        self.backbone.fc = torch.nn.Identity()  # Removes classification head

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
    )
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
    elif model_name.lower() == "lstm":
        model = CNN_LSTM(device=device).to(device)
        return model

    elif model_name.lower() == "mobilenetv3":
        return get_mobilenetv3()
    elif model_name.lower() == "resnet18radiomics":
        return Resnet18Radiomics().to(device)
    elif model_name.lower() == "cnnlstm":
        return CNNLSTMModel(device=device).to(device)
    elif model_name.lower() == "cnntransformer":
        return CNNTransformerModel(device=device).to(device)
    elif model_name.lower() == "cnnweak":
        return CNNWeakModel().to(device)
    elif model_name.lower() == "cnnweakradiomics":
        return CNNWeakRadiomics().to(device)
    
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