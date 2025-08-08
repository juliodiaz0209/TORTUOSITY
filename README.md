# Análisis de Tortuosidad Avanzado - FastAPI Backend

Este proyecto ha sido migrado desde Streamlit a FastAPI para proporcionar una API REST robusta y escalable para el análisis de tortuosidad de glándulas de Meibomio.

## 🚀 Características

- **API REST completa** con FastAPI
- **Interfaz web moderna** con HTML/CSS/JavaScript
- **Análisis de imágenes biomédicas** usando PyTorch
- **Modelos de IA**: Mask R-CNN y UNet
- **Documentación automática** con Swagger UI
- **CORS habilitado** para integración frontend
- **Manejo de errores robusto**
- **Procesamiento asíncrono**

## 📁 Estructura del Proyecto

```
TORTUOSITY/
├── main.py                 # Aplicación principal FastAPI
├── Tortuosity.py          # Funciones de análisis (sin cambios)
├── start_server.py        # Script de inicio
├── requirements.txt       # Dependencias Python
├── README.md             # Este archivo
├── static/
│   └── index.html        # Interfaz web
├── temp/                 # Archivos temporales (se crea automáticamente)
├── results/              # Resultados (se crea automáticamente)
├── final_model (11).pth  # Modelo Mask R-CNN
└── final_model_tarsus_improved.pth  # Modelo UNet
```

## 🛠️ Instalación

1. **Instalar dependencias**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Verificar archivos de modelo**:
   Asegúrate de que los archivos de modelo estén en el directorio raíz:
   - `final_model (11).pth`
   - `final_model_tarsus_improved.pth`

## 🚀 Ejecución

### Opción 1: Usar el script de inicio (Recomendado)
```bash
python start_server.py
```

### Opción 2: Ejecutar directamente
```bash
python main.py
```

### Opción 3: Usar uvicorn directamente
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## 🌐 Acceso a la Aplicación

Una vez iniciado el servidor, puedes acceder a:

- **Interfaz Web**: http://localhost:8000
- **Documentación API**: http://localhost:8000/docs
- **Información API**: http://localhost:8000/api
- **Health Check**: http://localhost:8000/health

## 📡 Endpoints de la API

### GET /
- **Descripción**: Interfaz web principal
- **Respuesta**: HTML con la interfaz de usuario

### GET /api
- **Descripción**: Información general de la API
- **Respuesta**: JSON con detalles de endpoints

### GET /health
- **Descripción**: Verificación del estado del servidor
- **Respuesta**: JSON con estado de modelos y dispositivo

### POST /analyze
- **Descripción**: Analizar imagen para tortuosidad
- **Parámetros**: `file` (imagen: jpg, jpeg, png)
- **Respuesta**: JSON con resultados del análisis

### GET /info
- **Descripción**: Información sobre la metodología de análisis
- **Respuesta**: JSON con detalles técnicos

## 📊 Ejemplo de Uso

### Usando curl
```bash
curl -X POST "http://localhost:8000/analyze" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@meibomio.jpg"
```

### Usando Python requests
```python
import requests

url = "http://localhost:8000/analyze"
files = {"file": open("meibomio.jpg", "rb")}
response = requests.post(url, files=files)
result = response.json()
print(result)
```

## 🔧 Configuración

### Variables de Entorno (Opcional)
```bash
export MODEL_MASK_RCNN_PATH="final_model (11).pth"
export MODEL_UNET_PATH="final_model_tarsus_improved.pth"
export API_HOST="0.0.0.0"
export API_PORT="8000"
```

### CORS Configuration
El servidor está configurado para permitir todas las origenes (`*`). Para producción, modifica en `main.py`:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://tu-dominio.com"],  # Especifica dominios
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## 📈 Diferencias con Streamlit

| Característica | Streamlit | FastAPI |
|----------------|-----------|---------|
| **Tipo** | Aplicación web monolítica | API REST + Frontend separado |
| **Escalabilidad** | Limitada | Alta (microservicios) |
| **Integración** | Difícil | Fácil (REST API) |
| **Performance** | Moderada | Alta |
| **Deployment** | Streamlit Cloud | Cualquier servidor |
| **Documentación** | Manual | Automática (Swagger) |

## 🚀 Deployment

### Docker (Recomendado)
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Heroku
```bash
# Crear Procfile
echo "web: uvicorn main:app --host 0.0.0.0 --port \$PORT" > Procfile

# Deploy
heroku create tu-app-name
git push heroku main
```

### VPS/Cloud
```bash
# Instalar dependencias del sistema
sudo apt update
sudo apt install python3-pip python3-venv

# Configurar aplicación
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Usar systemd para servicio
sudo systemctl enable tortuosity-api
sudo systemctl start tortuosity-api
```

## 🔍 Troubleshooting

### Error: "Models not loaded"
- Verifica que los archivos `.pth` estén en el directorio correcto
- Revisa los logs del servidor para errores de carga

### Error: "CUDA out of memory"
- Reduce el tamaño de las imágenes de entrada
- Usa CPU en lugar de GPU modificando `device` en `Tortuosity.py`

### Error: "File not found"
- Asegúrate de que los archivos de modelo tengan los nombres correctos
- Verifica permisos de lectura

## 📝 Logs

Los logs del servidor incluyen:
- Carga de modelos al inicio
- Procesamiento de imágenes
- Errores y excepciones
- Métricas de performance

## 🤝 Contribución

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo `LICENSE` para más detalles.

## 📞 Soporte

Para soporte técnico o preguntas:
- Abre un issue en GitHub
- Revisa la documentación en `/docs`
- Consulta los logs del servidor

---

**Desarrollado con ❤️ usando FastAPI, PyTorch y Streamlit** 