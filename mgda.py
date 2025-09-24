import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os

from skimage.morphology import binary_closing, disk, binary_erosion
from skimage.measure import find_contours, label
from scipy.ndimage import binary_fill_holes
import inferencia_tarso as tarso_module
from io import BytesIO

# (Padding con gradiente eliminado)

# -----------------------------------------------------------------------------------
# UNet and utilities (igual que antes)
# -----------------------------------------------------------------------------------
def resize_to_multiple(image, base_size=32):
    width, height = image.size
    new_width = (width // base_size) * base_size
    new_height = (height // base_size) * base_size
    try:
        resample = Image.Resampling.LANCZOS
    except AttributeError:
        resample = Image.LANCZOS
    return image.resize((new_width, new_height), resample)

class UNetWithPretrainedEncoder(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, encoder_pretrained=True):
        super(UNetWithPretrainedEncoder, self).__init__()
        self.encoder = models.resnet34(pretrained=encoder_pretrained)
        self.encoder_layers = list(self.encoder.children())

        self.layer0 = nn.Sequential(*self.encoder_layers[:3])
        self.layer1 = nn.Sequential(*self.encoder_layers[3:5])
        self.layer2 = self.encoder_layers[5]
        self.layer3 = self.encoder_layers[6]
        self.layer4 = self.encoder_layers[7]

        # Decoder
        self.upconv4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv4 = self.double_conv(512, 256)
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv3 = self.double_conv(256, 128)
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv2 = self.double_conv(128, 64)
        self.upconv1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.conv1 = self.double_conv(128, 64)

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

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
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

transform = transforms.Compose([
    transforms.Lambda(lambda img: resize_to_multiple(img, base_size=32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def load_model(model_path, device, encoder_pretrained=True):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Archivo del modelo no encontrado: {model_path}")
    model = UNetWithPretrainedEncoder(in_channels=3, out_channels=1, encoder_pretrained=encoder_pretrained)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print(f"Modelo cargado exitosamente desde: {model_path}")
    return model

def predict_image(model, image_path, device):
    # Abrir la imagen sin padding adicional
    image = Image.open(image_path).convert("RGB")
    resized_image = resize_to_multiple(image, base_size=32)
    input_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        output = torch.sigmoid(output)
        prediction = (output > 0.5).float()
    return resized_image, prediction.cpu().squeeze(0)

def display_and_save_results(image, gland_mask, tarsus_mask, expansion_mode, save_path="resultado_final.png"):
    """
    Muestra y guarda la imagen con un contorno unificado (cóncavo) alrededor de las glándulas.
    
    Según expansion_mode:
      - "inferior": Expansión desde el borde inferior de la máscara unificada hasta el borde inferior del tarso.
      - "superior": Expansión desde el borde superior de la máscara unificada hasta el borde superior del tarso.
    
    Luego se calcula el área de disfunción glandular (área del tarso menos área de glándulas) y 
    el porcentaje de disfunción (esta diferencia sobre el área total del tarso).
    """
    # 1) Intersección de máscaras: glándulas * tarso
    gland_inside_mask = gland_mask * tarsus_mask

    # Convertir a numpy
    gland_np = gland_inside_mask.numpy() if gland_inside_mask.dim() == 2 else gland_inside_mask.squeeze(0).numpy()
    tarsus_np = tarsus_mask.numpy() if tarsus_mask.dim() == 2 else tarsus_mask.squeeze(0).numpy()

    # 2) Cálculo de área del tarso (una sola vez)
    area_tarsus = np.sum(tarsus_np)
    
    # --------------------------------------------------------------------------------
    # 3) Unificar las glándulas (closing y refinamiento del contorno) - OPTIMIZADO
    # --------------------------------------------------------------------------------
    gland_bool = gland_np.astype(bool)
    tarsus_bool = tarsus_np.astype(bool)

    # Optimización: usar disco más pequeño para mejor rendimiento
    closing_disk_size = 12  # Reducido de 18 a 12
    closed = binary_closing(gland_bool, disk(closing_disk_size))
    closed = closed & tarsus_bool

    # Optimización: simplificar el refinamiento
    labeled_glands = label(closed)
    if labeled_glands.max() > 0:
        # Solo procesar si hay glándulas detectadas
        refined_closed = np.zeros_like(closed)
        for gland_label in range(1, labeled_glands.max() + 1):
            gland_region = labeled_glands == gland_label
            # Usar disco más pequeño para erosión
            refined_region = binary_closing(gland_region, disk(closing_disk_size // 4))
            refined_region = binary_erosion(refined_region, disk(1))  # Reducido de 2 a 1
            refined_closed = refined_closed | refined_region
        
        # Rellenar huecos solo si es necesario
        final_closed = binary_fill_holes(refined_closed)
    else:
        final_closed = closed

    # --------------------------------------------------------------------------------
    # 4) Expansión automática según el modo ("inferior" o "superior") - OPTIMIZADO
    # --------------------------------------------------------------------------------
    rows, cols = final_closed.shape
    filled = final_closed.copy()
    indices = np.argwhere(final_closed)
    
    if indices.shape[0] > 0:
        y_min, x_min = indices.min(axis=0)
        y_max, x_max = indices.max(axis=0)
        
        # Optimización: vectorizar la expansión usando operaciones de numpy
        if expansion_mode == "inferior":
            # Encontrar el límite inferior de glándulas por columna
            gland_bottom = np.argmax(final_closed[::-1, :], axis=0)  # Desde abajo hacia arriba
            gland_bottom = (rows - 1) - gland_bottom  # Convertir a coordenadas reales
            
            # Encontrar el límite inferior del tarso por columna
            tarsus_bottom = np.argmax(tarsus_np[::-1, :], axis=0)
            tarsus_bottom = (rows - 1) - tarsus_bottom
            
            # Expandir solo donde el tarso está más abajo que las glándulas
            for x in range(x_min, x_max + 1):
                if tarsus_bottom[x] > gland_bottom[x] and tarsus_bottom[x] < rows:
                    filled[gland_bottom[x]:tarsus_bottom[x] + 1, x] = True
                    
        elif expansion_mode == "superior":
            # Encontrar el límite superior de glándulas por columna
            gland_top = np.argmax(final_closed, axis=0)
            
            # Encontrar el límite superior del tarso por columna
            tarsus_top = np.argmax(tarsus_np, axis=0)
            
            # Expandir solo donde el tarso está más arriba que las glándulas
            for x in range(x_min, x_max + 1):
                if tarsus_top[x] < gland_top[x] and tarsus_top[x] >= 0:
                    filled[tarsus_top[x]:gland_top[x] + 1, x] = True
        
        final_closed = filled

    # Extraer contornos de la máscara unificada (cóncava)
    contours_unified = find_contours(final_closed, 0.5)

    # --------------------------------------------------------------------------------
    # 5) Cálculo optimizado del área y porcentaje de disfunción glandular
    # --------------------------------------------------------------------------------
    area_gland_unified = np.sum(final_closed)
    
    # Cálculo directo del porcentaje de disfunción (más eficiente)
    if area_tarsus > 0:
        mg_ratio_unified = area_gland_unified / area_tarsus
        dysfunction_percentage = (1.0 - mg_ratio_unified) * 100  # Más directo que calcular área intermedia
        dysfunction_area = area_tarsus - area_gland_unified
    else:
        mg_ratio_unified = 0.0
        dysfunction_percentage = 0.0
        dysfunction_area = 0

    # --------------------------------------------------------------------------------
    # 6) Visualización y guardado
    # --------------------------------------------------------------------------------
    image_np = np.array(image).astype(np.float32) / 255.0

    plt.figure(figsize=(10, 10))
    plt.imshow(image_np)
    plt.imshow(tarsus_np, cmap="jet", alpha=0.3)
    plt.imshow(gland_np, cmap="jet", alpha=0.5)

    for contour in contours_unified:
        plt.plot(contour[:, 1], contour[:, 0], linewidth=2, color='yellow')

    plt.text(10, 20, f"MG Ratio: {mg_ratio_unified:.4f}", color="white", fontsize=12,
             bbox=dict(facecolor="black", alpha=0.5))
    plt.text(10, 50, f"Área disfuncional: {dysfunction_area:.0f} pixeles", color="white", fontsize=12,
             bbox=dict(facecolor="black", alpha=0.5))
    plt.text(10, 80, f"Disfunción glandular: {dysfunction_percentage:.2f}%", color="white", fontsize=12,
             bbox=dict(facecolor="black", alpha=0.5))

    plt.title(f"Glándulas unificadas y disfunción ({expansion_mode.capitalize()})")
    plt.axis("off")
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
    plt.show()

    print(f"MG Ratio: {mg_ratio_unified:.4f}")
    print(f"Área disfuncional: {dysfunction_area:.0f} pixeles")
    print(f"Disfunción glandular: {dysfunction_percentage:.2f}%")
    print(f"Imagen final guardada en: {save_path}")

    return dysfunction_percentage

# -----------------------------------------------------------------------------------
# API-friendly reusable pipeline (no plt.show, returns PIL image and metrics)
# -----------------------------------------------------------------------------------
def analyze_mgda_with_models(
    image_path: str,
    model_meibomio: nn.Module,
    model_tarsus: nn.Module,
    device: torch.device,
    expansion_mode: str = "inferior"
):
    """
    Ejecuta el análisis MGDA usando modelos ya cargados.

    Returns: (PIL.Image, dict)
    dict keys: mg_ratio, dysfunction_percentage, dysfunction_area
    """
    # 1) Inferencia - similar al código original
    resized_image, mask_meibomio = predict_image(model_meibomio, image_path, device)
    _, mask_tarsus = tarso_module.predict_image(model_tarsus, image_path, device, use_clahe=True, use_tta=True)

    # 2) Post-procesamiento del tarso - similar al código original
    mask_tarsus_np = mask_tarsus.numpy() if mask_tarsus.dim() == 2 else mask_tarsus.squeeze(0).numpy()
    labeled_mask, num_components = label(mask_tarsus_np.astype(bool), connectivity=2, return_num=True)

    if num_components > 1:
        # Calcular áreas de cada componente
        component_areas = []
        for comp_id in range(1, num_components + 1):
            area = np.sum(labeled_mask == comp_id)
            component_areas.append((comp_id, area))

        # Ordenar por área descendente y mantener solo la componente más grande
        component_areas.sort(key=lambda x: x[1], reverse=True)
        largest_component_id = component_areas[0][0]

        # Crear máscara limpia con solo la componente principal
        cleaned_mask = (labeled_mask == largest_component_id).astype(np.uint8)
        print(f"Post-procesamiento tarso: eliminadas {num_components - 1} componentes pequeñas")
    else:
        cleaned_mask = mask_tarsus_np.astype(np.uint8)

    # Alinear la máscara de tarso al tamaño de la imagen/máscara de glándulas
    target_w, target_h = resized_image.size
    mask_tarsus_resized = Image.fromarray(cleaned_mask * 255, mode='L').resize((target_w, target_h), Image.NEAREST)
    tarsus_np = (np.array(mask_tarsus_resized) > 127).astype(np.uint8)

    # 3) Intersección y preparación - similar al código original
    gland_np = mask_meibomio.numpy() if mask_meibomio.dim() == 2 else mask_meibomio.squeeze(0).numpy()

    area_tarsus = np.sum(tarsus_np)

    gland_bool = gland_np.astype(bool)
    tarsus_bool = tarsus_np.astype(bool)

    # Optimización: usar disco más pequeño para mejor rendimiento
    closing_disk_size = 12  # Reducido de 18 a 12
    closed = binary_closing(gland_bool, disk(closing_disk_size))
    closed = closed & tarsus_bool

    # Optimización: simplificar el refinamiento
    labeled_glands = label(closed)
    if labeled_glands.max() > 0:
        # Solo procesar si hay glándulas detectadas
        refined_closed = np.zeros_like(closed)
        for gland_label in range(1, labeled_glands.max() + 1):
            gland_region = labeled_glands == gland_label
            # Usar disco más pequeño para erosión
            refined_region = binary_closing(gland_region, disk(closing_disk_size // 4))
            refined_region = binary_erosion(refined_region, disk(1))  # Reducido de 2 a 1
            refined_closed = refined_closed | refined_region

        # Rellenar huecos solo si es necesario
        final_closed = binary_fill_holes(refined_closed)
    else:
        final_closed = closed

    # --------------------------------------------------------------------------------
    # 4) Expansión automática según el modo ("inferior" o "superior") - OPTIMIZADO
    # --------------------------------------------------------------------------------
    rows, cols = final_closed.shape
    filled = final_closed.copy()
    indices = np.argwhere(final_closed)

    if indices.shape[0] > 0:
        y_min, x_min = indices.min(axis=0)
        y_max, x_max = indices.max(axis=0)

        # Optimización: vectorizar la expansión usando operaciones de numpy
        if expansion_mode == "inferior":
            # Encontrar el límite inferior de glándulas por columna
            gland_bottom = np.argmax(final_closed[::-1, :], axis=0)  # Desde abajo hacia arriba
            gland_bottom = (rows - 1) - gland_bottom  # Convertir a coordenadas reales

            # Encontrar el límite inferior del tarso por columna
            tarsus_bottom = np.argmax(tarsus_np[::-1, :], axis=0)
            tarsus_bottom = (rows - 1) - tarsus_bottom

            # Expandir solo donde el tarso está más abajo que las glándulas
            for x in range(x_min, x_max + 1):
                if tarsus_bottom[x] > gland_bottom[x] and tarsus_bottom[x] < rows:
                    filled[gland_bottom[x]:tarsus_bottom[x] + 1, x] = True

        elif expansion_mode == "superior":
            # Encontrar el límite superior de glándulas por columna
            gland_top = np.argmax(final_closed, axis=0)

            # Encontrar el límite superior del tarso por columna
            tarsus_top = np.argmax(tarsus_np, axis=0)

            # Expandir solo donde el tarso está más arriba que las glándulas
            for x in range(x_min, x_max + 1):
                if tarsus_top[x] < gland_top[x] and tarsus_top[x] >= 0:
                    filled[tarsus_top[x]:gland_top[x] + 1, x] = True

        final_closed = filled

    # Extraer contornos de la máscara unificada (cóncava)
    contours_unified = find_contours(final_closed, 0.5)

    # --------------------------------------------------------------------------------
    # 5) Cálculo optimizado del área y porcentaje de disfunción glandular
    # --------------------------------------------------------------------------------
    area_gland_unified = np.sum(final_closed)

    # Cálculo directo del porcentaje de disfunción (más eficiente)
    if area_tarsus > 0:
        mg_ratio_unified = area_gland_unified / area_tarsus
        dysfunction_percentage = (1.0 - mg_ratio_unified) * 100  # Más directo que calcular área intermedia
        dysfunction_area = area_tarsus - area_gland_unified
    else:
        mg_ratio_unified = 0.0
        dysfunction_percentage = 0.0
        dysfunction_area = 0

    # --------------------------------------------------------------------------------
    # 6) Visualización y guardado (sin plt.show para API)
    # --------------------------------------------------------------------------------
    image_np = np.array(resized_image).astype(np.float32) / 255.0

    plt.figure(figsize=(10, 10))
    plt.imshow(image_np)
    plt.imshow(tarsus_np, cmap="jet", alpha=0.3)
    plt.imshow(gland_np, cmap="jet", alpha=0.5)

    for contour in contours_unified:
        plt.plot(contour[:, 1], contour[:, 0], linewidth=2, color='yellow')

    plt.text(10, 20, f"MG Ratio: {mg_ratio_unified:.4f}", color="white", fontsize=12,
             bbox=dict(facecolor="black", alpha=0.5))
    plt.text(10, 50, f"Área disfuncional: {dysfunction_area:.0f} pixeles", color="white", fontsize=12,
             bbox=dict(facecolor="black", alpha=0.5))
    plt.text(10, 80, f"Disfunción glandular: {dysfunction_percentage:.2f}%", color="white", fontsize=12,
             bbox=dict(facecolor="black", alpha=0.5))

    plt.title(f"Glándulas unificadas y disfunción ({expansion_mode.capitalize()})")
    plt.axis("off")

    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    img_pil = Image.open(buf)
    plt.close()

    return img_pil, {
        "mg_ratio": float(mg_ratio_unified),
        "dysfunction_percentage": float(dysfunction_percentage),
        "dysfunction_area": int(dysfunction_area)
    }

# -----------------------------------------------------------------------------------
# Ejecución principal
# -----------------------------------------------------------------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Rutas de los modelos
    model_path_meibomio = "final_model_improved_fixed.pth"  # Modelo para glándulas de meibomio
    model_path_tarsus = "final_model_tarsus_improved (6).pth"     # Modelo para contorno del párpado (tarso)

    # Cargar ambos modelos
    model_meibomio = load_model(model_path_meibomio, device, encoder_pretrained=True)
    # Usar el cargador del módulo de tarso entrenado con Attention Gates y CLAHE
    model_tarsus = tarso_module.load_model(model_path_tarsus, device)

    # Paths separados de entrada para glándulas y tarso
    image_path_gland = "tests\clahe_optimized_3_times (33).png"
    image_path_tarsus = "tests\clahe_optimized_3_times (35).png"

    # Determinar el modo de expansión según el nombre del archivo de glándulas
    filename = os.path.basename(image_path_gland)
    if "_I" in filename:
        expansion_mode = "inferior"
    elif "_S" in filename:
        expansion_mode = "superior"
    else:
        expansion_mode = "inferior"  # Por defecto

    # Realizar la inferencia con paths separados
    resized_image, mask_meibomio = predict_image(model_meibomio, image_path_gland, device)
    # Usar la inferencia especializada del módulo de tarso (con CLAHE y TTA)
    _, mask_tarsus = tarso_module.predict_image(model_tarsus, image_path_tarsus, device, use_clahe=True, use_tta=True)

    # Post-procesamiento: eliminar componentes pequeñas, mantener solo el tarso principal
    mask_tarsus_np = mask_tarsus.squeeze(0).numpy().astype(bool) if mask_tarsus.dim() == 3 else mask_tarsus.numpy().astype(bool)
    labeled_mask, num_components = label(mask_tarsus_np, connectivity=2, return_num=True)

    if num_components > 1:
        # Calcular áreas de cada componente
        component_areas = []
        for comp_id in range(1, num_components + 1):
            area = np.sum(labeled_mask == comp_id)
            component_areas.append((comp_id, area))

        # Ordenar por área descendente y mantener solo la componente más grande
        component_areas.sort(key=lambda x: x[1], reverse=True)
        largest_component_id = component_areas[0][0]

        # Crear máscara limpia con solo la componente principal
        cleaned_mask = (labeled_mask == largest_component_id).astype(np.uint8)
        print(f"Post-procesamiento tarso: eliminadas {num_components - 1} componentes pequeñas")
    else:
        cleaned_mask = mask_tarsus_np.astype(np.uint8)

    # Alinear la máscara de tarso al tamaño de la imagen/máscara de glándulas
    target_w, target_h = resized_image.size
    mask_tarsus_resized = Image.fromarray(cleaned_mask * 255, mode='L').resize((target_w, target_h), Image.NEAREST)
    mask_tarsus = torch.from_numpy((np.array(mask_tarsus_resized) > 127).astype(np.float32))

    # Visualizar, guardar y calcular indicadores
    display_and_save_results(resized_image, mask_meibomio, mask_tarsus, expansion_mode,
                             save_path="resultado_final.png")
