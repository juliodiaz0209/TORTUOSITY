import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2  # Para CLAHE

def resize_to_target_size(img, target_size=(512, 512)):
    """
    Redimensiona la imagen al tamaño objetivo usado en entrenamiento.
    """
    return img.resize((target_size[1], target_size[0]), Image.BILINEAR)

def apply_clahe_preprocessing(img_pil):
    """
    Aplica CLAHE igual que en entrenamiento
    """
    # Convertir a escala de grises
    img_gray = img_pil.convert("L")
    img_np_gray = np.array(img_gray)
    
    # Aplicar CLAHE con parámetros similares al entrenamiento
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_eq = clahe.apply(img_np_gray)
    
    # Replicar a 3 canales como en entrenamiento
    img_np_rgb = np.stack([img_eq]*3, axis=-1)
    return Image.fromarray(img_np_rgb, mode='RGB')

# Definición de Attention Gate (igual que en script.py)
class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid()
        )
        self.act = nn.ReLU(inplace=True)
    
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.act(g1 + x1)
        psi = self.psi(psi)
        return x * psi

# Definición del modelo (igual que en el entrenamiento con Attention Gates)
class UNetWithPretrainedEncoder(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, use_attention=True):
        super(UNetWithPretrainedEncoder, self).__init__()
        self.use_attention = use_attention
        # Usar los mismos pesos preentrenados que en entrenamiento (script.py)
        try:
            self.encoder = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        except AttributeError:
            # Compatibilidad con versiones anteriores de torchvision
            self.encoder = models.resnet34(pretrained=True)
        self.encoder_layers = list(self.encoder.children())
        self.layer0 = nn.Sequential(*self.encoder_layers[:3])   # Conv1, BN1, ReLU
        self.layer1 = nn.Sequential(*self.encoder_layers[3:5])    # MaxPool + Layer1
        self.layer2 = self.encoder_layers[5]                      # Layer2
        self.layer3 = self.encoder_layers[6]                      # Layer3
        self.layer4 = self.encoder_layers[7]                      # Layer4

        # Decoder
        self.upconv4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv4 = self._double_conv(512, 256)
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv3 = self._double_conv(256, 128)
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv2 = self._double_conv(128, 64)
        self.upconv1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.conv1 = self._double_conv(128, 64)
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        
        # Attention Gates (igual que en script.py)
        if self.use_attention:
            self.att3 = AttentionGate(256, 256, 128)
            self.att2 = AttentionGate(128, 128, 64)
            self.att1 = AttentionGate(64, 64, 32)
            self.att0 = AttentionGate(64, 64, 32)
    
    def forward(self, x):
        x0 = self.layer0(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        
        d4 = self.upconv4(x4)
        skip3 = self.att3(d4, x3) if self.use_attention else x3
        d4 = torch.cat((d4, skip3), dim=1)
        d4 = self.conv4(d4)
        
        d3 = self.upconv3(d4)
        skip2 = self.att2(d3, x2) if self.use_attention else x2
        d3 = torch.cat((d3, skip2), dim=1)
        d3 = self.conv3(d3)
        
        d2 = self.upconv2(d3)
        skip1 = self.att1(d2, x1) if self.use_attention else x1
        d2 = torch.cat((d2, skip1), dim=1)
        d2 = self.conv2(d2)
        
        d1 = self.upconv1(d2)
        skip0 = self.att0(d1, x0) if self.use_attention else x0
        d1 = torch.cat((d1, skip0), dim=1)
        d1 = self.conv1(d1)
        
        out = self.final_conv(d1)
        out = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=True)
        return out
    
    def _double_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

# Pipeline de transformaciones: igual que en entrenamiento
transform = transforms.Compose([
    transforms.Lambda(lambda img: resize_to_target_size(img, (512, 512))),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def load_model(model_path, device):
    model = UNetWithPretrainedEncoder(in_channels=3, out_channels=1, use_attention=True)
    ckpt = torch.load(model_path, map_location=device)
    # Aceptar formatos: {'model': state_dict, ...} o state_dict plano
    if isinstance(ckpt, dict) and 'model' in ckpt:
        state_dict = ckpt['model']
    else:
        state_dict = ckpt
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[WARN] Missing keys: {len(missing)} (primeros 5) {missing[:5]}")
    if unexpected:
        print(f"[WARN] Unexpected keys: {len(unexpected)} (primeros 5) {unexpected[:5]}")
    model.to(device)
    model.eval()
    return model

def predict_image(model, image_path, device, use_clahe=True, use_tta=True):
    # Cargar imagen original
    original_image = Image.open(image_path).convert("RGB")
    
    # Aplicar CLAHE preprocessing igual que en entrenamiento
    if use_clahe:
        processed_image = apply_clahe_preprocessing(original_image)
    else:
        processed_image = original_image
    
    # Aplicar transformaciones
    input_tensor = transform(processed_image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        if use_tta:
            # Test-Time Augmentation igual que en script.py
            output = model(input_tensor)
            # Flip horizontal
            img_flip = torch.flip(input_tensor, dims=[3])
            output_flip = model(img_flip)
            output_flip = torch.flip(output_flip, dims=[3])
            # Promedio
            output = (output + output_flip) / 2.0
        else:
            output = model(input_tensor)
        
        output = torch.sigmoid(output)  # Convertir logits a probabilidades
        prediction = (output > 0.5).float()  # Umbralización
    
    return original_image, prediction.cpu().squeeze(0)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "final_model_tarsus_improved (6).pth"  # Usar el mejor modelo
    model = load_model(model_path, device)
    
    image_path = "tests\clahe_optimized_4_times (1).jpg"
    original_image, pred_mask = predict_image(model, image_path, device, use_clahe=True, use_tta=True)
    
    # Redimensionar imagen original para visualización
    resized_image = resize_to_target_size(original_image, (512, 512))
    image_np = np.array(resized_image).astype(np.float32) / 255.0
    
    # Convertir la máscara predicha a numpy
    mask_np = pred_mask.squeeze(0).numpy()
    
    # Crear visualización mejorada
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Imagen original
    axes[0].imshow(image_np)
    axes[0].set_title("Imagen Original (CLAHE aplicado)")
    axes[0].axis("off")
    
    # Máscara predicha
    axes[1].imshow(mask_np, cmap='gray')
    axes[1].set_title("Máscara Predicha (Tarso)")
    axes[1].axis("off")
    
    # Superposición
    axes[2].imshow(image_np)
    axes[2].imshow(mask_np, cmap="jet", alpha=0.5)
    axes[2].set_title("Superposición")
    axes[2].axis("off")
    
    plt.tight_layout()
    plt.show()
    
    print(f"Proporción de píxeles segmentados: {mask_np.mean():.3f}")
