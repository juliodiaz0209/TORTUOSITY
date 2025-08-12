# Frontend - Análisis de Tortuosidad Avanzado

Frontend moderno para la aplicación de análisis de tortuosidad glandular construido con Next.js, TypeScript, Tailwind CSS y Shadcn/ui.

## 🚀 Características

- **UI Moderna**: Interfaz elegante y profesional con Shadcn/ui
- **Drag & Drop**: Carga de archivos intuitiva con react-dropzone
- **Gráficos Interactivos**: Visualizaciones con Recharts
- **Responsive Design**: Optimizado para todos los dispositivos
- **TypeScript**: Tipado completo para mejor desarrollo
- **Dark Mode**: Soporte para tema oscuro/claro

## 🛠️ Tecnologías

- **Next.js 15**: Framework React con App Router
- **TypeScript**: Tipado estático
- **Tailwind CSS**: Framework de CSS utility-first
- **Shadcn/ui**: Componentes de UI modernos
- **Recharts**: Librería de gráficos
- **Lucide React**: Iconos modernos
- **React Dropzone**: Manejo de archivos

## 📦 Instalación

```bash
# Instalar dependencias
npm install

# Ejecutar en modo desarrollo
npm run dev
```

## 🔧 Configuración

### Variables de Entorno

Crea un archivo `.env.local` en la raíz del proyecto:

```env
NEXT_PUBLIC_API_URL=https://tortuosity-backend-488176611125.us-central1.run.app
```

### Proxy API

El frontend está configurado para redirigir las llamadas a `/api/*` hacia el backend FastAPI en `https://tortuosity-backend-488176611125.us-central1.run.app`.

## 🎨 Componentes Principales

### UploadZone
- Zona de carga de archivos con drag & drop
- Validación de tipos de archivo
- Preview del archivo seleccionado

### ResultsDisplay
- Visualización de resultados del análisis
- Métricas principales en cards
- Tabla de tortuosidad individual
- Gráfico de barras interactivo
- Información detallada

## 📱 Uso

1. **Cargar Imagen**: Arrastra y suelta una imagen o haz clic para seleccionar
2. **Analizar**: Haz clic en "Analizar Imagen" para procesar
3. **Ver Resultados**: Revisa las métricas, tabla y gráfico
4. **Interpretar**: Usa la información detallada para interpretar los resultados

## 🎯 Funcionalidades

- ✅ Carga de archivos con drag & drop
- ✅ Validación de tipos de archivo
- ✅ Barra de progreso durante el análisis
- ✅ Manejo de errores con alertas
- ✅ Métricas principales en tiempo real
- ✅ Tabla de tortuosidad individual con badges
- ✅ Gráfico de barras interactivo
- ✅ Información detallada del análisis
- ✅ Diseño responsive
- ✅ Tema oscuro/claro

## 🚀 Scripts Disponibles

```bash
npm run dev          # Ejecutar en desarrollo
npm run build        # Construir para producción
npm run start        # Ejecutar en producción
npm run lint         # Ejecutar ESLint
```

## 📁 Estructura del Proyecto

```
src/
├── app/                 # App Router de Next.js
│   ├── globals.css     # Estilos globales
│   ├── layout.tsx      # Layout principal
│   └── page.tsx        # Página principal
├── components/         # Componentes React
│   ├── ui/            # Componentes de Shadcn/ui
│   ├── upload-zone.tsx # Componente de carga
│   └── results-display.tsx # Componente de resultados
└── lib/               # Utilidades
    └── utils.ts       # Funciones utilitarias
```

## 🔗 Integración con Backend

El frontend se comunica con el backend FastAPI a través de:

- **Endpoint**: `/api/analyze`
- **Método**: POST
- **Formato**: FormData con archivo de imagen
- **Respuesta**: JSON con resultados del análisis

## 🎨 Personalización

### Temas
Los temas se pueden personalizar en `src/app/globals.css` modificando las variables CSS de Shadcn/ui.

### Componentes
Los componentes de Shadcn/ui se pueden personalizar en `src/components/ui/`.

## 📄 Licencia

Este proyecto es parte de la aplicación de análisis de tortuosidad glandular.
