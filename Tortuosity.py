import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision import transforms, models
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch.nn.functional as F
import random
import cv2  # Needed for contour operations in tortuosity calculation

# Definir dispositivo (GPU o CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------------------------------
# Función para cargar el modelo Mask R-CNN y sus pesos
# ------------------------------------------------------
def load_maskrcnn_model(model_path="final_model.pth"):
    # Nota: el parámetro 'pretrained' está deprecado, se recomienda usar 'weights'
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=False)
    num_classes = 2  # fondo y glándula
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

# ------------------------------------------------------
# Definición del modelo UNet con encoder preentrenado
# ------------------------------------------------------
class UNetWithPretrainedEncoder(torch.nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNetWithPretrainedEncoder, self).__init__()
        self.encoder = models.resnet34(pretrained=False)
        self.encoder_layers = list(self.encoder.children())
        self.layer0 = torch.nn.Sequential(*self.encoder_layers[:3])
        self.layer1 = torch.nn.Sequential(*self.encoder_layers[3:5])
        self.layer2 = self.encoder_layers[5]
        self.layer3 = self.encoder_layers[6]
        self.layer4 = self.encoder_layers[7]

        self.upconv4 = torch.nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv4 = self.double_conv(512, 256)
        self.upconv3 = torch.nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv3 = self.double_conv(256, 128)
        self.upconv2 = torch.nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv2 = self.double_conv(128, 64)
        self.upconv1 = torch.nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.conv1 = self.double_conv(128, 64)
        self.final_conv = torch.nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        x0 = self.layer0(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        d4 = self.upconv4(x4)
        d4 = torch.cat((d4, x3), dim=1)
        d4 = self.conv4(d4)

        d3 = self.upconv3(d4)
        d3 = torch.cat((d3, x2), dim=1)
        d3 = self.conv3(d3)

        d2 = self.upconv2(d3)
        d2 = torch.cat((d2, x1), dim=1)
        d2 = self.conv2(d2)

        d1 = self.upconv1(d2)
        d1 = torch.cat((d1, x0), dim=1)
        d1 = self.conv1(d1)

        out = self.final_conv(d1)
        out = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=True)
        return out

    def double_conv(self, in_channels, out_channels):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True)
        )

# ------------------------------------------------------
# Funciones para cargar y predecir con UNet
# ------------------------------------------------------
def load_unet_model(model_path, device):
    model = UNetWithPretrainedEncoder(in_channels=3, out_channels=1)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

def predict_unet_model(model, image_path, device):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Lambda(resize_to_previous_multiple_of_32),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    input_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        output = torch.sigmoid(output)
        prediction = (output > 0.5).float()  # Umbralización para obtener máscara binaria
    return prediction.cpu().squeeze(0)

# ------------------------------------------------------
# Función de predicción de instancias con Mask-RCNN
# ------------------------------------------------------
def predict_maskrcnn_model(model, image_path, device):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        prediction = model(image_tensor)[0]
    masks_np = prediction["masks"].cpu().numpy()
    scores = prediction["scores"].cpu().numpy()
    selected_masks = (masks_np[scores > 0.5] > 0.5).astype(np.uint8)
    if selected_masks.size == 0:
        return np.zeros(image_tensor.shape[2:], dtype=np.int32)
    pred_instance = np.zeros(selected_masks.shape[2:], dtype=np.int32)
    for i, mask in enumerate(selected_masks, start=1):
        mask_squeezed = mask.squeeze(0)
        pred_instance[mask_squeezed > 0.5] = i
    return pred_instance

# ------------------------------------------------------
# Función para generar un color aleatorio (RGB)
# ------------------------------------------------------
def generate_random_color():
    return [random.randint(0, 255) for _ in range(3)]

# ------------------------------------------------------
# Función para calcular la tortuosidad de una glándula de Meibomio
# ------------------------------------------------------
def calculate_gland_tortuosity(mask):
    """
    Calcula la tortuosidad de una glándula de Meibomio según la fórmula:
    Tortuosidad = (Perímetro / (2 × Altura del rectángulo mínimo externo)) - 1

    Args:
        mask: Máscara binaria de la glándula

    Returns:
        Valor de tortuosidad
    """
    # Encontrar contornos en la máscara
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return 0.0

    # Usar el contorno más grande (debería ser la glándula)
    contour = max(contours, key=cv2.contourArea)

    # Calcular el perímetro
    perimeter = cv2.arcLength(contour, True)

    # Obtener el rectángulo mínimo externo
    rect = cv2.minAreaRect(contour)
    (_, (width, height), _) = rect

    # Usar la dimensión más larga como altura
    max_dim = max(width, height) if max(width, height) > 0 else 1

    # Calcular tortuosidad
    tortuosity = (perimeter / (2 * max_dim)) - 1

    # Limitar el valor máximo de tortuosidad a 1.0 para evitar valores extremos
    tortuosity = min(tortuosity, 1.0)

    return tortuosity

# ------------------------------------------------------
# Función de visualización y combinación final
# ------------------------------------------------------
def show_combined_result_with_models(image_path, maskrcnn_model, unet_model, device):
    # Usar modelos ya cargados (no cargar de nuevo)
    
    # Predicción de la máscara de instancias (glándulas de Meibomio)
    pred_instance = predict_maskrcnn_model(maskrcnn_model, image_path, device)

    # Predicción de la máscara del contorno del párpado (Tarsus)
    mask_tarsus = predict_unet_model(unet_model, image_path, device)

    # Redimensionar la máscara de Tarsus a las dimensiones de la máscara de instancias
    mask_tarsus_resized = F.interpolate(mask_tarsus.unsqueeze(0), size=pred_instance.shape, mode='bilinear', align_corners=True)
    mask_tarsus_resized = mask_tarsus_resized.squeeze(0).cpu().numpy()

    # Aquí, aún tenemos forma (1, H, W); eliminamos la dimensión extra:
    mask_tarsus_resized = np.squeeze(mask_tarsus_resized, axis=0)

    # Aplicar la máscara de Tarsus a la máscara de instancias (conservar solo las instancias dentro del contorno)
    pred_instance_cleaned = pred_instance * mask_tarsus_resized.astype(np.int32)
    pred_instance_cleaned = np.squeeze(pred_instance_cleaned)

    # Crear imagen de instancias: asignar un color aleatorio único a cada instancia
    colored_instance_image = np.zeros((pred_instance_cleaned.shape[0], pred_instance_cleaned.shape[1], 3), dtype=np.uint8)

    # Calcular la tortuosidad para cada glándula
    gland_tortuosities = []
    gland_ids = np.unique(pred_instance_cleaned)
    gland_ids = gland_ids[gland_ids > 0]  # Excluir el fondo (0)

    for i in gland_ids:
        # Crear una máscara para esta glándula específica
        gland_mask = (pred_instance_cleaned == i).astype(np.uint8)

        # Calcular la tortuosidad
        tortuosity = calculate_gland_tortuosity(gland_mask)
        gland_tortuosities.append(tortuosity)

        # Asignar color basado en la tortuosidad (opcional: usar un mapa de colores)
        color = generate_random_color()
        colored_instance_image[pred_instance_cleaned == i] = color

    # Calcular la tortuosidad promedio
    avg_tortuosity = np.mean(gland_tortuosities) if gland_tortuosities else 0.0

    # Abrir la imagen original
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)

    # Superponer las instancias (colores sólidos) sobre la imagen original
    result_image = image_np.copy()
    mask_inst = (colored_instance_image.sum(axis=-1) > 0)
    result_image[mask_inst] = colored_instance_image[mask_inst]

    # Redimensionar la máscara de Tarsus para que tenga el mismo tamaño que la imagen original para su visualización
    tarsus_mask_overlay = F.interpolate(mask_tarsus.unsqueeze(0), size=(image_np.shape[0], image_np.shape[1]), mode='bilinear', align_corners=True)
    tarsus_mask_overlay = tarsus_mask_overlay.squeeze(0).cpu().numpy()
    tarsus_mask_overlay = np.squeeze(tarsus_mask_overlay, axis=0)  # Eliminar la dimensión extra
    tarsus_mask_overlay = (tarsus_mask_overlay > 0.5).astype(np.uint8)

    # Visualizar la imagen final: instancias en colores únicos y el contorno del párpado con transparencia
    plt.figure(figsize=(10, 10))
    plt.imshow(result_image)
    plt.imshow(tarsus_mask_overlay, cmap="jet", alpha=0.5)  # Overlay del contorno

    # Añadir información de tortuosidad al título
    plt.title(f"Instancias (colores únicos) y contorno del párpado\nTortuosidad promedio: {avg_tortuosity:.3f}")
    plt.axis("off")
    # plt.show() # Commented out for Streamlit integration

    # Return the final image array and tortuosity data
    import io
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    img_arr = Image.open(buf)
    plt.close() # Close the plot to free memory

    # Return both the image and tortuosity data
    return img_arr, {
        'avg_tortuosity': avg_tortuosity,
        'individual_tortuosities': gland_tortuosities,
        'num_glands': len(gland_ids)
    }

def show_combined_result(image_path, maskrcnn_model_path, unet_model_path, device):
    # Cargar modelos
    maskrcnn_model = load_maskrcnn_model(maskrcnn_model_path)
    unet_model = load_unet_model(unet_model_path, device)

    # Predicción de la máscara de instancias (glándulas de Meibomio)
    pred_instance = predict_maskrcnn_model(maskrcnn_model, image_path, device)

    # Predicción de la máscara del contorno del párpado (Tarsus)
    mask_tarsus = predict_unet_model(unet_model, image_path, device)

    # Redimensionar la máscara de Tarsus a las dimensiones de la máscara de instancias
    mask_tarsus_resized = F.interpolate(mask_tarsus.unsqueeze(0), size=pred_instance.shape, mode='bilinear', align_corners=True)
    mask_tarsus_resized = mask_tarsus_resized.squeeze(0).cpu().numpy()

    # Aquí, aún tenemos forma (1, H, W); eliminamos la dimensión extra:
    mask_tarsus_resized = np.squeeze(mask_tarsus_resized, axis=0)

    # Aplicar la máscara de Tarsus a la máscara de instancias (conservar solo las instancias dentro del contorno)
    pred_instance_cleaned = pred_instance * mask_tarsus_resized.astype(np.int32)
    pred_instance_cleaned = np.squeeze(pred_instance_cleaned)

    # Crear imagen de instancias: asignar un color aleatorio único a cada instancia
    colored_instance_image = np.zeros((pred_instance_cleaned.shape[0], pred_instance_cleaned.shape[1], 3), dtype=np.uint8)

    # Calcular la tortuosidad para cada glándula
    gland_tortuosities = []
    gland_ids = np.unique(pred_instance_cleaned)
    gland_ids = gland_ids[gland_ids > 0]  # Excluir el fondo (0)

    for i in gland_ids:
        # Crear una máscara para esta glándula específica
        gland_mask = (pred_instance_cleaned == i).astype(np.uint8)

        # Calcular la tortuosidad
        tortuosity = calculate_gland_tortuosity(gland_mask)
        gland_tortuosities.append(tortuosity)

        # Asignar color basado en la tortuosidad (opcional: usar un mapa de colores)
        color = generate_random_color()
        colored_instance_image[pred_instance_cleaned == i] = color

    # Calcular la tortuosidad promedio
    avg_tortuosity = np.mean(gland_tortuosities) if gland_tortuosities else 0.0

    # Abrir la imagen original
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)

    # Superponer las instancias (colores sólidos) sobre la imagen original
    result_image = image_np.copy()
    mask_inst = (colored_instance_image.sum(axis=-1) > 0)
    result_image[mask_inst] = colored_instance_image[mask_inst]

    # Redimensionar la máscara de Tarsus para que tenga el mismo tamaño que la imagen original para su visualización
    tarsus_mask_overlay = F.interpolate(mask_tarsus.unsqueeze(0), size=(image_np.shape[0], image_np.shape[1]), mode='bilinear', align_corners=True)
    tarsus_mask_overlay = tarsus_mask_overlay.squeeze(0).cpu().numpy()
    tarsus_mask_overlay = np.squeeze(tarsus_mask_overlay, axis=0)  # Eliminar la dimensión extra
    tarsus_mask_overlay = (tarsus_mask_overlay > 0.5).astype(np.uint8)

    # Visualizar la imagen final: instancias en colores únicos y el contorno del párpado con transparencia
    plt.figure(figsize=(10, 10))
    plt.imshow(result_image)
    plt.imshow(tarsus_mask_overlay, cmap="jet", alpha=0.5)  # Overlay del contorno

    # Añadir información de tortuosidad al título
    plt.title(f"Instancias (colores únicos) y contorno del párpado\nTortuosidad promedio: {avg_tortuosity:.3f}")
    plt.axis("off")
    # plt.show() # Commented out for Streamlit integration

    # Return the final image array and tortuosity data
    import io
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    img_arr = Image.open(buf)
    plt.close() # Close the plot to free memory

    # Return both the image and tortuosity data
    return img_arr, {
        'avg_tortuosity': avg_tortuosity,
        'individual_tortuosities': gland_tortuosities,
        'num_glands': len(gland_ids)
    }


# ------------------------------------------------------
# Función para redimensionar la imagen a múltiplos de 32
# ------------------------------------------------------
def resize_to_previous_multiple_of_32(img):
    """
    Redimensiona la imagen a dimensiones que sean múltiplos de 32, redondeando hacia abajo.
    Esto preserva la relación de aspecto, pero puede recortar algunos píxeles.
    """
    w, h = img.size
    new_w = (w // 32) * 32
    new_h = (h // 32) * 32
    return img.resize((new_w, new_h), Image.BILINEAR)

# ------------------------------------------------------
# Bloque principal (Commented out for Streamlit integration)
# ------------------------------------------------------
# if __name__ == "__main__":
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     maskrcnn_model_path = "final_model (11).pth"
#     unet_model_path = "final_model_tarsus.pth"
#     image_path = "meibomio.jpg"

#     show_combined_result(image_path, maskrcnn_model_path, unet_model_path, device)
