"use client";

import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Progress } from "@/components/ui/progress";
import { UploadZone } from "@/components/upload-zone";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import { ResultsDisplay } from "@/components/results-display";
import { PhotoManager } from "@/components/photo-manager";
import { StoredPhoto } from "@/lib/photo-storage";
import {
  Eye,
  Brain,
  Zap,
  AlertCircle,
  Loader2,
  BarChart3,
  Upload,
  Settings,
  Home,
  FileText,
  Sparkles,
  FileImage,
  Camera
} from "lucide-react";

interface AnalysisResult {
  success: boolean;
  message: string;
  data: {
    processed_image: string;
    avg_tortuosity: number;
    num_glands: number;
    individual_tortuosities: number[];
    analysis_info: {
      total_glands_analyzed: number;
      tortuosity_range: {
        min: number;
        max: number;
      };
    };
  };
}

export default function DashboardPage() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [results, setResults] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [progress, setProgress] = useState(0);
  const [activeTab, setActiveTab] = useState<'upload' | 'capture' | 'results' | 'info'>('upload');
  const [claheImage, setClaheImage] = useState<string | null>(null);
  const [isApplyingClahe, setIsApplyingClahe] = useState(false);
  const [convertToGray, setConvertToGray] = useState(true);
  const [selectedCapturedPhoto, setSelectedCapturedPhoto] = useState<StoredPhoto | null>(null);

  const handleFileSelect = (file: File | null) => {
    setSelectedFile(file);
    setError(null);
    setResults(null);
    setClaheImage(null);
  };

  const handleCapturedPhotoSelect = (photo: StoredPhoto) => {
    setSelectedCapturedPhoto(photo);
    setSelectedFile(null); // Clear any uploaded file
    setError(null);
    setResults(null);
    setClaheImage(null);
  };

  const handleConvertToGrayChange = (checked: boolean) => {
    setConvertToGray(checked);
    setClaheImage(null); // Limpiar imagen anterior cuando se cambia la opción
  };

  const loadExampleImage = async () => {
    try {
      // Load the example image from the frontend public folder
      const response = await fetch('/meibomio.jpg');
      const blob = await response.blob();
      const file = new File([blob], 'meibomio.jpg', { type: 'image/jpeg' });
      handleFileSelect(file);
    } catch (error) {
      console.error('Error loading example image:', error);
      setError('No se pudo cargar la imagen de ejemplo');
    }
  };

  const analyzeImage = async () => {
    if (!selectedFile && !selectedCapturedPhoto) return;

    setIsAnalyzing(true);
    setError(null);
    setProgress(0);

    // Simular progreso
    const progressInterval = setInterval(() => {
      setProgress((prev) => {
        if (prev >= 90) {
          clearInterval(progressInterval);
          return 90;
        }
        return prev + 10;
      });
    }, 500);

    try {
      let fileToAnalyze: File;

      if (selectedFile) {
        fileToAnalyze = selectedFile;
      } else if (selectedCapturedPhoto) {
        // Convert data URL to File object
        const response = await fetch(selectedCapturedPhoto.dataUrl);
        const blob = await response.blob();
        fileToAnalyze = new File([blob], selectedCapturedPhoto.fileName || 'captured-image.jpg', { type: 'image/jpeg' });
      } else {
        throw new Error('No hay imagen seleccionada para analizar');
      }

      const formData = new FormData();
      formData.append("file", fileToAnalyze);

      const response = await fetch("/api/analyze", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const contentType = response.headers.get("content-type");
        
        if (contentType && contentType.includes("application/json")) {
          try {
        const errorData = await response.json();
        throw new Error(errorData.detail || "Error en el análisis");
          } catch {
            throw new Error(`Error del servidor: ${response.status}`);
          }
        } else {
          try {
            const errorText = await response.text();
            throw new Error(`Error del servidor: ${response.status} - ${errorText.substring(0, 100)}`);
          } catch {
            throw new Error(`Error del servidor: ${response.status}`);
          }
        }
      }

      const result: AnalysisResult = await response.json();

      // Save analysis results to local storage if this was a captured photo
      if (selectedCapturedPhoto) {
        try {
          // Update the photo in local storage with analysis results
          const analysisResults = {
            avgTortuosity: result.data.avg_tortuosity,
            numGlands: result.data.num_glands,
            individualTortuosities: result.data.individual_tortuosities
          };

          // Note: The photoStorage.updatePhotoAnalysis would need to be implemented
          // For now, we'll just log the results
          console.log('Analysis results for captured photo:', {
            photoId: selectedCapturedPhoto.id,
            results: analysisResults
          });
        } catch (error) {
          console.warn('Could not save analysis results to local storage:', error);
        }
      }

      setResults(result);
      setProgress(100);
      setActiveTab('results');
    } catch (err) {
      setError(err instanceof Error ? err.message : "Error desconocido");
    } finally {
      setIsAnalyzing(false);
      clearInterval(progressInterval);
    }
  };

  const applyClaheFilter = async () => {
    if (!selectedFile && !selectedCapturedPhoto) return;

    setIsApplyingClahe(true);
    setError(null);

    try {
      let fileToProcess: File;

      if (selectedFile) {
        fileToProcess = selectedFile;
      } else if (selectedCapturedPhoto) {
        // Convert data URL to File object
        const response = await fetch(selectedCapturedPhoto.dataUrl);
        const blob = await response.blob();
        fileToProcess = new File([blob], selectedCapturedPhoto.fileName || 'captured-image.jpg', { type: 'image/jpeg' });
      } else {
        throw new Error('No hay imagen seleccionada para procesar');
      }

      const formData = new FormData();
      formData.append("file", fileToProcess);
      formData.append("convert_to_gray", convertToGray.toString());

      const response = await fetch("/api/apply-clahe", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const contentType = response.headers.get("content-type");
        
        if (contentType && contentType.includes("application/json")) {
          try {
            const errorData = await response.json();
            throw new Error(errorData.detail || "Error applying CLAHE filter");
          } catch {
            throw new Error(`Error del servidor: ${response.status}`);
          }
        } else {
          try {
            const errorText = await response.text();
            throw new Error(`Error del servidor: ${response.status} - ${errorText.substring(0, 100)}`);
          } catch {
            throw new Error(`Error del servidor: ${response.status}`);
          }
        }
      }

      const result = await response.json();
      setClaheImage(result.data.processed_image);

      // Replace the selected file with the CLAHE-processed image so the model uses it
      try {
        const dataUrl = result.data.processed_image as string;
        const res = await fetch(dataUrl);
        const blob = await res.blob();

        if (selectedFile) {
          const baseName = selectedFile.name.replace(/\.[^/.]+$/, "");
          const processedFile = new File([blob], `clahe_${baseName}.png`, { type: "image/png" });
          setSelectedFile(processedFile);
        } else if (selectedCapturedPhoto) {
          const baseName = selectedCapturedPhoto.fileName?.replace(/\.[^/.]+$/, "") || 'captured';
          const processedFile = new File([blob], `clahe_${baseName}.png`, { type: "image/png" });
          setSelectedFile(processedFile);
          setSelectedCapturedPhoto(null); // Clear captured photo since we're now using the processed file
        }

        // Clear previous results if any
        setResults(null);
      } catch {}
    } catch (err) {
      setError(err instanceof Error ? err.message : "Error desconocido");
    } finally {
      setIsApplyingClahe(false);
    }
  };

  const SidebarItem = ({ icon: Icon, label, active, onClick }: {
    icon: React.ComponentType<{ className?: string }>;
    label: string;
    active: boolean;
    onClick: () => void;
  }) => (
    <button
      onClick={onClick}
      className={`flex items-center gap-3 w-full p-3 rounded-lg transition-all ${
        active 
          ? 'bg-primary text-primary-foreground shadow-sm' 
          : 'text-muted-foreground hover:text-foreground hover:bg-muted'
      }`}
    >
      <Icon className="h-5 w-5" />
      <span className="font-medium">{label}</span>
    </button>
  );

  return (
    <div className="min-h-screen bg-background flex w-full">
      {/* Sidebar */}
      <div className="fixed left-0 top-0 h-full w-64 dashboard-sidebar p-4">
        <div className="flex flex-col h-full">
          {/* Logo */}
          <div className="flex items-center gap-2 mb-8">
            <div className="p-2 bg-primary rounded-lg">
              <Eye className="h-6 w-6 text-primary-foreground" />
            </div>
            <div>
              <h1 className="font-bold text-lg">Tortuosity AI</h1>
              <p className="text-xs text-muted-foreground">Análisis Avanzado</p>
            </div>
          </div>

          {/* Navigation */}
          <nav className="flex-1 space-y-2">
            <SidebarItem
              icon={Home}
              label="Inicio"
              active={activeTab === 'upload'}
              onClick={() => setActiveTab('upload')}
            />
            <SidebarItem
              icon={Camera}
              label="Capturar"
              active={activeTab === 'capture'}
              onClick={() => setActiveTab('capture')}
            />
            <SidebarItem
              icon={BarChart3}
              label="Resultados"
              active={activeTab === 'results'}
              onClick={() => setActiveTab('results')}
            />
            <SidebarItem
              icon={FileText}
              label="Información"
              active={activeTab === 'info'}
              onClick={() => setActiveTab('info')}
            />
          </nav>

          {/* Status */}
          <div className="pt-4 border-t border-border">
            <div className="flex items-center gap-2 text-sm text-muted-foreground">
              <div className="w-2 h-2 bg-green-500 rounded-full"></div>
              <span>Sistema Activo</span>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="ml-64 p-6 dashboard-content min-h-screen bg-background w-full">
        {/* Header */}
        <div className="mb-6">
          <h2 className="text-2xl font-bold">
            {activeTab === 'upload' && 'Análisis de Imagen'}
            {activeTab === 'capture' && 'Captura de Imagen IR - Módulo Especializado'}
            {activeTab === 'results' && 'Resultados del Análisis'}
            {activeTab === 'info' && 'Información del Sistema'}
          </h2>
          <p className="text-muted-foreground">
            {activeTab === 'upload' && 'Sube una imagen del párpado para analizar la tortuosidad glandular'}
            {activeTab === 'capture' && 'Conecta tu módulo IR especializado y captura imágenes para análisis profesional de meibografía'}
            {activeTab === 'results' && 'Visualiza los resultados detallados del análisis'}
            {activeTab === 'info' && 'Información sobre la metodología y modelos utilizados'}
          </p>
        </div>

        {/* Content */}
        {activeTab === 'upload' && (
          <div className="space-y-6 animate-fade-in bg-background">
            {/* Upload Zone */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Upload className="h-5 w-5" />
                  Cargar Imagen
                </CardTitle>
              </CardHeader>
              <CardContent>
                <UploadZone onFileSelect={handleFileSelect} selectedFile={selectedFile} />
                
                {/* Botón de ejemplo */}
                {!selectedFile && (
                  <div className="mt-4 text-center">
                    <Button 
                      variant="outline" 
                      onClick={loadExampleImage}
                      className="w-full"
                    >
                      <FileImage className="mr-2 h-4 w-4" />
                      Cargar Imagen de Ejemplo
                    </Button>
                    <p className="text-xs text-muted-foreground mt-2">
                      Prueba la aplicación con una imagen de glándulas de Meibomio
                    </p>
                  </div>
                )}
                
                {(selectedFile || selectedCapturedPhoto) && (
                  <div className="mt-4 flex flex-col gap-4">
                    {/* CLAHE Options */}
                    <div className="flex flex-col items-center space-y-2">
                      <div className="flex items-center space-x-2">
                        <Switch
                          id="convert-to-gray"
                          checked={convertToGray}
                          onCheckedChange={handleConvertToGrayChange}
                        />
                        <Label htmlFor="convert-to-gray" className="text-sm">
                          {convertToGray ? "Procesar en escala de grises" : "Procesar en color"}
                        </Label>
                      </div>
                      <p className="text-xs text-muted-foreground text-center">
                        {convertToGray 
                          ? "Recomendado para imágenes con filtros de color o biomédicas"
                          : "Mantiene la información de color original"
                        }
                      </p>
                    </div>
                    
                    <div className="flex gap-2 justify-center">
                      <Button 
                        onClick={analyzeImage} 
                        disabled={isAnalyzing}
                        className="flex-1"
                      >
                        {isAnalyzing ? (
                          <>
                            <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                            Analizando...
                          </>
                        ) : (
                          <>
                            <Zap className="mr-2 h-4 w-4" />
                            Analizar Imagen
                          </>
                        )}
                      </Button>
                      <Button 
                        onClick={applyClaheFilter} 
                        disabled={isApplyingClahe}
                        variant="outline"
                        className="flex-1"
                      >
                        {isApplyingClahe ? (
                          <>
                            <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                            Aplicando...
                          </>
                        ) : (
                          <>
                            <Sparkles className="mr-2 h-4 w-4" />
                            Aplicar CLAHE
                          </>
                        )}
                      </Button>
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>

            {/* Progress */}
            {isAnalyzing && (
              <Card>
                <CardContent className="pt-6">
                  <div className="space-y-4">
                    <div className="flex items-center gap-2">
                      <Loader2 className="h-4 w-4 animate-spin" />
                      <span className="text-sm font-medium">Procesando imagen...</span>
                    </div>
                    <Progress value={progress} className="w-full" />
                    <p className="text-xs text-muted-foreground text-center">
                      Cargando modelos y analizando glándulas...
                    </p>
                  </div>
                </CardContent>
              </Card>
            )}

            {/* CLAHE Result */}
            {claheImage && (
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Sparkles className="h-5 w-5" />
                    Imagen con CLAHE Aplicado
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <img
                    src={claheImage}
                    alt="Imagen con filtro CLAHE"
                    className="w-full rounded-lg shadow-lg"
                  />
                  <p className="text-sm text-muted-foreground mt-2 text-center">
                    Filtro CLAHE aplicado para mejorar el contraste local
                  </p>
                </CardContent>
              </Card>
            )}

            {/* Error */}
            {error && (
              <Alert className="border-red-200 bg-red-50 dark:border-red-800 dark:bg-red-950">
                <AlertCircle className="h-4 w-4 text-red-600" />
                <AlertDescription className="text-red-800 dark:text-red-200">
                  {error}
                </AlertDescription>
              </Alert>
            )}
          </div>
        )}

        {activeTab === 'capture' && (
          <div className="animate-fade-in bg-background">
            <PhotoManager onPhotoSelect={handleCapturedPhotoSelect} />
          </div>
        )}

        {activeTab === 'results' && (
          <div className="animate-fade-in bg-background">
            {results ? (
              <ResultsDisplay 
                data={results.data} 
                processedImage={results.data.processed_image} 
              />
            ) : (
              <Card>
                <CardContent className="pt-6">
                  <div className="text-center py-12">
                    <BarChart3 className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
                    <h3 className="text-lg font-medium mb-2">No hay resultados</h3>
                    <p className="text-muted-foreground">
                      Analiza una imagen primero para ver los resultados aquí.
                    </p>
                    <Button 
                      onClick={() => setActiveTab('upload')} 
                      className="mt-4"
                    >
                      Ir a Análisis
                    </Button>
                  </div>
                </CardContent>
              </Card>
            )}
          </div>
        )}

        {activeTab === 'info' && (
          <div className="space-y-6 animate-fade-in bg-background">
            {/* Methodology */}
            <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Brain className="h-5 w-5" />
                  Metodología
            </CardTitle>
          </CardHeader>
              <CardContent className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="space-y-2">
                <h4 className="font-semibold text-green-600">Fórmula de Tortuosidad</h4>
                <p className="text-sm text-muted-foreground">
                  Tortuosidad = (Perímetro / (2 × Altura del rectángulo mínimo externo)) - 1
                </p>
              </div>
              <div className="space-y-2">
                <h4 className="font-semibold text-blue-600">Interpretación</h4>
                <div className="text-sm text-muted-foreground space-y-1">
                  <p><strong>0.0 - 0.1:</strong> Tortuosidad baja (normal)</p>
                  <p><strong>0.1 - 0.2:</strong> Tortuosidad moderada</p>
                  <p><strong>&gt; 0.2:</strong> Tortuosidad alta (sugestivo de MGD)</p>
                </div>
              </div>
              <div className="space-y-2">
                <h4 className="font-semibold text-purple-600">Modelos Utilizados</h4>
                <div className="text-sm text-muted-foreground space-y-1">
                  <p><strong>Mask R-CNN:</strong> Detección de glándulas individuales</p>
                  <p><strong>UNet:</strong> Segmentación del contorno del párpado</p>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

            {/* System Info */}
            <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
                  <Settings className="h-5 w-5" />
                  Información del Sistema
            </CardTitle>
          </CardHeader>
          <CardContent>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <h4 className="font-semibold">Tecnologías</h4>
                    <ul className="text-sm text-muted-foreground space-y-1">
                      <li>• Frontend: Next.js 15 con TypeScript</li>
                      <li>• Backend: FastAPI con Python</li>
                      <li>• IA: PyTorch (Mask R-CNN & UNet)</li>
                      <li>• UI: Tailwind CSS + shadcn/ui</li>
                    </ul>
                  </div>
                  <div className="space-y-2">
                    <h4 className="font-semibold">Características</h4>
                    <ul className="text-sm text-muted-foreground space-y-1">
                      <li>• Análisis en tiempo real</li>
                      <li>• Visualización interactiva</li>
                      <li>• Métricas detalladas</li>
                      <li>• Interfaz responsiva</li>
                    </ul>
              </div>
              </div>
            </CardContent>
          </Card>
          </div>
        )}
      </div>
    </div>
  );
}
