"use client";

import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Progress } from "@/components/ui/progress";
import { UploadZone } from "@/components/upload-zone";
import { ResultsDisplay } from "@/components/results-display";
import { Eye, Brain, Zap, AlertCircle, Loader2 } from "lucide-react";

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

export default function Home() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [results, setResults] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [progress, setProgress] = useState(0);

  const handleFileSelect = (file: File) => {
    setSelectedFile(file);
    setError(null);
    setResults(null);
  };

  const analyzeImage = async () => {
    if (!selectedFile) return;

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
      const formData = new FormData();
      formData.append("file", selectedFile);

      const response = await fetch("/api/analyze", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        let errorMessage = "Error en el análisis";
        const contentType = response.headers.get("content-type");
        
        if (contentType && contentType.includes("application/json")) {
          try {
            const errorData = await response.json();
            errorMessage = errorData.detail || errorMessage;
          } catch {
            errorMessage = `Error del servidor: ${response.status}`;
          }
        } else {
          // Si no es JSON, leer como texto
          try {
            const errorText = await response.text();
            errorMessage = `Error del servidor: ${response.status} - ${errorText.substring(0, 100)}`;
          } catch {
            errorMessage = `Error del servidor: ${response.status}`;
          }
        }
        throw new Error(errorMessage);
      }

      const result: AnalysisResult = await response.json();
      setResults(result);
      setProgress(100);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Error desconocido");
    } finally {
      setIsAnalyzing(false);
      clearInterval(progressInterval);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 dark:from-slate-900 dark:to-slate-800">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent mb-4">
            👁️ Análisis Avanzado de Tortuosidad Glandular
          </h1>
          <p className="text-lg text-muted-foreground">
            Análisis de imágenes biomédicas usando PyTorch (Mask R-CNN & UNet)
          </p>
        </div>

        {/* Información del análisis */}
        <Card className="mb-8">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Brain className="h-5 w-5" />
              Información del Análisis
            </CardTitle>
          </CardHeader>
          <CardContent>
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

        {/* Zona de carga */}
        <Card className="mb-6">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Eye className="h-5 w-5" />
              Cargar Imagen del Párpado
            </CardTitle>
          </CardHeader>
          <CardContent>
            <UploadZone onFileSelect={handleFileSelect} selectedFile={selectedFile} />
            
            {selectedFile && (
              <div className="mt-4 flex justify-center">
                <Button 
                  onClick={analyzeImage} 
                  disabled={isAnalyzing}
                  className="w-full md:w-auto"
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
              </div>
            )}
          </CardContent>
        </Card>

        {/* Progreso de análisis */}
        {isAnalyzing && (
          <Card className="mb-6">
            <CardContent className="pt-6">
              <div className="space-y-4">
                <div className="flex items-center gap-2">
                  <Loader2 className="h-4 w-4 animate-spin" />
                  <span className="text-sm font-medium">Cargando modelos y procesando imagen...</span>
                </div>
                <Progress value={progress} className="w-full" />
                <p className="text-xs text-muted-foreground text-center">
                  Por favor espera, esto puede tomar unos momentos.
                </p>
              </div>
            </CardContent>
          </Card>
        )}

        {/* Error */}
        {error && (
          <Alert className="mb-6 border-red-200 bg-red-50 dark:border-red-800 dark:bg-red-950">
            <AlertCircle className="h-4 w-4 text-red-600" />
            <AlertDescription className="text-red-800 dark:text-red-200">
              {error}
            </AlertDescription>
          </Alert>
        )}

        {/* Resultados */}
        {results && (
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Zap className="h-5 w-5" />
                Resultados del Análisis
              </CardTitle>
            </CardHeader>
            <CardContent>
              <ResultsDisplay 
                data={results.data} 
                processedImage={results.data.processed_image} 
              />
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  );
}
