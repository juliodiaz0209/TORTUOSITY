# Instrucciones para Ejecutar la Aplicación

## Requisitos previos

1. **Python 3.8+** instalado
2. **Node.js 18+** instalado
3. **Git** instalado
4. **Módulo IR USB** (opcional, para captura de imágenes)

## Instalación y ejecución

### 1. Clonar el repositorio (si no lo tienes ya)

```bash
git clone <tu-repositorio>
cd TORTUOSITY
```

### 2. Instalar dependencias del backend

```bash
pip install -r requirements.txt
```

### 3. Instalar dependencias del frontend

```bash
cd frontend
npm install
cd ..
```

### 4. Ejecutar el backend

```bash
python main.py
```

El backend estará disponible en: http://localhost:8000

### 5. Ejecutar el frontend (en otra terminal)

```bash
cd frontend
npm run dev
```

El frontend estará disponible en: http://localhost:3000

## Verificar que todo funciona

1. Abre http://localhost:3000 en tu navegador
2. Ve a la pestaña **"Capturar"**
3. Conecta tu módulo IR USB
4. Haz clic en **"Iniciar Cámara"**
5. Deberías ver el feed de video
6. Captura algunas fotos
7. Ve a **"Galería"** para ver las fotos guardadas
8. Selecciona una foto y haz clic en **"Analizar Esta Foto"**

## Estructura de archivos

```
TORTUOSITY/
├── frontend/                 # Aplicación Next.js
│   ├── src/
│   │   ├── components/
│   │   │   ├── camera-capture.tsx    # Componente de captura
│   │   │   ├── photo-manager.tsx     # Gestión de fotos
│   │   │   └── ...
│   │   └── lib/
│   │       └── photo-storage.ts      # Almacenamiento local
│   └── ...
├── main.py                   # Backend FastAPI
├── requirements.txt          # Dependencias Python
├── Tortuosity.py            # Modelos de IA
├── final_model*.pth         # Modelos entrenados
└── README-CAPTURE.md        # Documentación completa
```

## Resolución de problemas comunes

### Error de modelos no encontrados
```bash
# Verifica que los archivos de modelo existan
ls -la *.pth
```

### Error de dependencias
```bash
# Reinstalar dependencias del backend
pip install --upgrade -r requirements.txt

# Reinstalar dependencias del frontend
cd frontend && npm install && cd ..
```

### Error de puerto ocupado
```bash
# Cambiar puerto del backend
python main.py --port 8001

# Cambiar puerto del frontend
cd frontend && npm run dev -- -p 3001
```

### Error de permisos de cámara
- Asegúrate de usar HTTPS o localhost
- Otorga permisos cuando el navegador lo pida
- Verifica que no haya otras aplicaciones usando la cámara

## Funcionalidades implementadas

### ✅ Funcionalidades principales
- Captura de imágenes desde módulo IR USB
- Almacenamiento local persistente con IndexedDB
- Análisis de tortuosidad con modelos de IA
- Interfaz moderna con Next.js y Tailwind CSS
- Galería de fotos con funciones CRUD

### ✅ Características técnicas
- WebRTC para acceso a dispositivos USB
- Canvas API para captura de frames
- Integración completa con backend Python
- Soporte para múltiples dispositivos
- Exportación/importación de datos

## Próximos pasos

1. **Probar con módulo IR real**: Conecta tu dispositivo y verifica funcionamiento
2. **Personalizar configuración**: Ajusta resoluciones y parámetros según necesidades
3. **Agregar funcionalidades**: Implementa features adicionales según requerimientos
4. **Despliegue**: Configura para producción con HTTPS y base de datos

## Contacto

Para soporte técnico o preguntas sobre la implementación, consulta la documentación en `README-CAPTURE.md` o contacta al desarrollador.
