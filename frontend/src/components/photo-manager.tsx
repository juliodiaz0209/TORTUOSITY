"use client";

import React, { useState, useEffect, useCallback } from 'react';
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Camera,
  Download,
  Trash2,
  FileImage,
  Upload,
  AlertCircle,
  CheckCircle,
  BarChart3,
  Image as ImageIcon
} from "lucide-react";
import { CameraCapture } from "./camera-capture";
import { photoStorage, StoredPhoto } from "@/lib/photo-storage";

interface PhotoManagerProps {
  onPhotoSelect?: (photo: StoredPhoto) => void;
}

export function PhotoManager({ onPhotoSelect }: PhotoManagerProps) {
  const [storedPhotos, setStoredPhotos] = useState<StoredPhoto[]>([]);
  const [selectedPhoto, setSelectedPhoto] = useState<StoredPhoto | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string>('');

  // Load photos from storage
  const loadPhotos = useCallback(async () => {
    try {
      setIsLoading(true);
      setError('');
      const photos = await photoStorage.getAllPhotos();
      setStoredPhotos(photos);
    } catch (err) {
      console.error('Error loading photos:', err);
      setError('Error al cargar las fotos guardadas');
    } finally {
      setIsLoading(false);
    }
  }, []);

  // Handle photo capture from camera
  const handlePhotoCapture = useCallback(async (dataUrl: string) => {
    try {
      const photo: StoredPhoto = {
        id: Date.now().toString(),
        dataUrl,
        timestamp: new Date(),
        fileName: `meibomio-capture-${Date.now()}.jpg`
      };

      await photoStorage.savePhoto(photo);
      await loadPhotos(); // Refresh the list
    } catch (err) {
      console.error('Error saving captured photo:', err);
      setError('Error al guardar la foto capturada');
    }
  }, [loadPhotos]);

  // Handle photo selection
  const handlePhotoSelect = useCallback((photo: StoredPhoto) => {
    setSelectedPhoto(photo);
    if (onPhotoSelect) {
      onPhotoSelect(photo);
    }
  }, [onPhotoSelect]);

  // Delete photo
  const deletePhoto = useCallback(async (photoId: string) => {
    try {
      await photoStorage.deletePhoto(photoId);
      await loadPhotos();
      if (selectedPhoto?.id === photoId) {
        setSelectedPhoto(null);
      }
    } catch (err) {
      console.error('Error deleting photo:', err);
      setError('Error al eliminar la foto');
    }
  }, [loadPhotos, selectedPhoto]);

  // Download photo
  const downloadPhoto = useCallback((photo: StoredPhoto) => {
    const link = document.createElement('a');
    link.href = photo.dataUrl;
    link.download = photo.fileName || `meibomio-${photo.timestamp.getTime()}.jpg`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  }, []);

  // Export all photos
  const exportAllPhotos = useCallback(async () => {
    try {
      const jsonData = await photoStorage.exportPhotos();
      const blob = new Blob([jsonData], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `meibomio-photos-backup-${Date.now()}.json`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);
    } catch (err) {
      console.error('Error exporting photos:', err);
      setError('Error al exportar las fotos');
    }
  }, []);

  // Clear all photos
  const clearAllPhotos = useCallback(async () => {
    if (confirm('¿Estás seguro de que quieres eliminar todas las fotos? Esta acción no se puede deshacer.')) {
      try {
        await photoStorage.clearAllPhotos();
        await loadPhotos();
        setSelectedPhoto(null);
      } catch (err) {
        console.error('Error clearing photos:', err);
        setError('Error al eliminar todas las fotos');
      }
    }
  }, [loadPhotos]);

  // Load photos on component mount
  useEffect(() => {
    loadPhotos();
  }, [loadPhotos]);

  return (
    <div className="space-y-6">
      <Tabs defaultValue="capture" className="w-full">
        <TabsList className="grid w-full grid-cols-2">
          <TabsTrigger value="capture" className="flex items-center gap-2">
            <Camera className="h-4 w-4" />
            Capturar
          </TabsTrigger>
          <TabsTrigger value="gallery" className="flex items-center gap-2">
            <ImageIcon className="h-4 w-4" />
            Galería ({storedPhotos.length})
          </TabsTrigger>
        </TabsList>

        <TabsContent value="capture" className="space-y-4">
          <CameraCapture onPhotoCapture={handlePhotoCapture} />
        </TabsContent>

        <TabsContent value="gallery" className="space-y-4">
          {/* Gallery Header */}
          <Card>
            <CardHeader>
              <div className="flex items-center justify-between">
                <CardTitle className="flex items-center gap-2">
                  <FileImage className="h-5 w-5" />
                  Fotos Guardadas ({storedPhotos.length})
                </CardTitle>
                <div className="flex gap-2">
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={exportAllPhotos}
                    disabled={storedPhotos.length === 0}
                  >
                    <Download className="mr-2 h-4 w-4" />
                    Exportar
                  </Button>
                  <Button
                    variant="destructive"
                    size="sm"
                    onClick={clearAllPhotos}
                    disabled={storedPhotos.length === 0}
                  >
                    <Trash2 className="mr-2 h-4 w-4" />
                    Limpiar Todo
                  </Button>
                </div>
              </div>
            </CardHeader>
            <CardContent>
              {error && (
                <Alert className="border-red-200 bg-red-50 mb-4">
                  <AlertCircle className="h-4 w-4 text-red-600" />
                  <AlertDescription className="text-red-800">{error}</AlertDescription>
                </Alert>
              )}

              {isLoading ? (
                <div className="text-center py-8">
                  <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary mx-auto mb-2"></div>
                  <p className="text-muted-foreground">Cargando fotos...</p>
                </div>
              ) : storedPhotos.length === 0 ? (
                <div className="text-center py-12">
                  <FileImage className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
                  <h3 className="text-lg font-medium mb-2">No hay fotos guardadas</h3>
                  <p className="text-muted-foreground mb-4">
                    Captura algunas fotos usando la pestaña "Capturar"
                  </p>
                  <Button onClick={() => document.querySelector('[value="capture"]')?.click()}>
                    <Camera className="mr-2 h-4 w-4" />
                    Ir a Capturar
                  </Button>
                </div>
              ) : (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                  {storedPhotos.map((photo) => (
                    <Card
                      key={photo.id}
                      className={`cursor-pointer transition-all hover:shadow-md ${
                        selectedPhoto?.id === photo.id ? 'ring-2 ring-primary' : ''
                      }`}
                      onClick={() => handlePhotoSelect(photo)}
                    >
                      <CardContent className="p-4">
                        <div className="relative mb-3">
                          <img
                            src={photo.dataUrl}
                            alt={`Foto ${photo.timestamp.toLocaleDateString()}`}
                            className="w-full h-32 object-cover rounded-lg"
                          />
                          {photo.analysisResults && (
                            <Badge className="absolute top-2 right-2 bg-green-500">
                              <BarChart3 className="h-3 w-3 mr-1" />
                              Analizada
                            </Badge>
                          )}
                        </div>

                        <div className="space-y-2">
                          <p className="text-sm font-medium">
                            {photo.timestamp.toLocaleDateString()} {photo.timestamp.toLocaleTimeString()}
                          </p>

                          {photo.analysisResults && (
                            <div className="text-xs text-muted-foreground">
                              <p>Tortuosidad: {photo.analysisResults.avgTortuosity.toFixed(3)}</p>
                              <p>Glándulas: {photo.analysisResults.numGlands}</p>
                            </div>
                          )}

                          <div className="flex gap-1">
                            <Button
                              size="sm"
                              variant="outline"
                              onClick={(e) => {
                                e.stopPropagation();
                                downloadPhoto(photo);
                              }}
                              className="flex-1"
                            >
                              <Download className="h-3 w-3 mr-1" />
                              Descargar
                            </Button>
                            <Button
                              size="sm"
                              variant="destructive"
                              onClick={(e) => {
                                e.stopPropagation();
                                deletePhoto(photo.id);
                              }}
                              className="flex-1"
                            >
                              <Trash2 className="h-3 w-3 mr-1" />
                              Eliminar
                            </Button>
                          </div>
                        </div>
                      </CardContent>
                    </Card>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>

          {/* Selected Photo Details */}
          {selectedPhoto && (
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <CheckCircle className="h-5 w-5" />
                  Foto Seleccionada
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <img
                    src={selectedPhoto.dataUrl}
                    alt="Foto seleccionada"
                    className="w-full max-h-96 object-contain rounded-lg border"
                  />

                  <div className="grid grid-cols-2 gap-4 text-sm">
                    <div>
                      <p className="font-medium">Fecha:</p>
                      <p className="text-muted-foreground">
                        {selectedPhoto.timestamp.toLocaleDateString()} {selectedPhoto.timestamp.toLocaleTimeString()}
                      </p>
                    </div>

                    {selectedPhoto.analysisResults ? (
                      <div className="space-y-2">
                        <p className="font-medium">Resultados del Análisis:</p>
                        <div className="text-muted-foreground">
                          <p>Tortuosidad promedio: {selectedPhoto.analysisResults.avgTortuosity.toFixed(3)}</p>
                          <p>Número de glándulas: {selectedPhoto.analysisResults.numGlands}</p>
                          <p>Glándulas analizadas: {selectedPhoto.analysisResults.individualTortuosities.length}</p>
                        </div>
                      </div>
                    ) : (
                      <div>
                        <p className="font-medium">Estado:</p>
                        <Badge variant="outline">Sin analizar</Badge>
                      </div>
                    )}
                  </div>

                  <div className="flex gap-2">
                    <Button onClick={() => onPhotoSelect?.(selectedPhoto)} className="flex-1">
                      <BarChart3 className="mr-2 h-4 w-4" />
                      Analizar Esta Foto
                    </Button>
                    <Button variant="outline" onClick={() => downloadPhoto(selectedPhoto)}>
                      <Download className="mr-2 h-4 w-4" />
                      Descargar
                    </Button>
                  </div>
                </div>
              </CardContent>
            </Card>
          )}
        </TabsContent>
      </Tabs>
    </div>
  );
}
