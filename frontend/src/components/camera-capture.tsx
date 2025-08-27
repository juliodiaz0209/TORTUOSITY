"use client";

import React, { useRef, useState, useCallback, useEffect } from 'react';
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Camera, CameraOff, Download, Trash2, AlertCircle, CheckCircle } from "lucide-react";

interface CameraCaptureProps {
  onPhotoCapture?: (photoData: string) => void;
}

interface CapturedPhoto {
  id: string;
  dataUrl: string;
  timestamp: Date;
  deviceId?: string;
}

export function CameraCapture({ onPhotoCapture }: CameraCaptureProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);

  const [isStreaming, setIsStreaming] = useState(false);
  const [availableDevices, setAvailableDevices] = useState<MediaDeviceInfo[]>([]);
  const [selectedDeviceId, setSelectedDeviceId] = useState<string>('');
  const [error, setError] = useState<string>('');
  const [isCapturing, setIsCapturing] = useState(false);
  const [capturedPhotos, setCapturedPhotos] = useState<CapturedPhoto[]>([]);
  const [showDeviceInfo, setShowDeviceInfo] = useState(false);
  const [showAllDevices, setShowAllDevices] = useState(false);
  const [allVideoDevices, setAllVideoDevices] = useState<MediaDeviceInfo[]>([]);

  // Get all video devices (for debugging)
  const getAllVideoDevices = useCallback(async () => {
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      const allVideoDevices = devices.filter(device => device.kind === 'videoinput');
      return allVideoDevices;
    } catch (err) {
      console.error('Error getting all video devices:', err);
      return [];
    }
  }, []);

  // Get available IR camera devices
  const getAvailableDevices = useCallback(async () => {
    try {
      console.log('🔍 Buscando módulos IR especializados...');
      
      // First, try to get user media to trigger permissions
      try {
        const testStream = await navigator.mediaDevices.getUserMedia({ video: true });
        testStream.getTracks().forEach(track => track.stop());
        console.log('✅ Permisos de cámara otorgados');
      } catch (permError) {
        console.log('⚠️ Error de permisos:', permError);
      }

      const devices = await navigator.mediaDevices.enumerateDevices();
      
      // Filter only IR cameras and specialized medical imaging devices
      const irDevices = devices
        .filter(device => device.kind === 'videoinput')
        .filter(device => device.deviceId && device.deviceId.trim() !== '')
        .filter(device => {
          const label = device.label.toLowerCase();
          
          // Exclude virtual cameras and OBS
          if (
            label.includes('obs') ||
            label.includes('virtual') ||
            label.includes('virtual camera') ||
            label.includes('screen capture') ||
            label.includes('desktop') ||
            label.includes('monitor') ||
            label.includes('display')
          ) {
            return false;
          }
          
          // Include only specialized IR and medical devices
          return (
            label.includes('ir') ||
            label.includes('infrarrojo') ||
            label.includes('infrared') ||
            label.includes('meibomio') ||
            label.includes('meibography') ||
            label.includes('medical') ||
            label.includes('specialized') ||
            label.includes('professional') ||
            label.includes('clinical') ||
            label.includes('usb') ||
            label.includes('webcam')
          );
        });

      // Log all video devices for debugging
      const allVideoDevicesList = devices.filter(device => device.kind === 'videoinput');
      setAllVideoDevices(allVideoDevicesList);
      console.log('📹 Todos los dispositivos de video detectados:', allVideoDevicesList.map(d => ({
        label: d.label,
        deviceId: d.deviceId.substring(0, 8) + '...',
        groupId: d.groupId
      })));
      
      console.log('🔴 Módulos IR especializados detectados:', irDevices.map(d => ({
        label: d.label,
        deviceId: d.deviceId.substring(0, 8) + '...',
        groupId: d.groupId
      })));
      
      // Log excluded devices for debugging
      const excludedDevices = allVideoDevices.filter(device => {
        const label = device.label.toLowerCase();
        return (
          label.includes('obs') ||
          label.includes('virtual') ||
          label.includes('virtual camera') ||
          label.includes('screen capture') ||
          label.includes('desktop') ||
          label.includes('monitor') ||
          label.includes('display')
        );
      });
      
      if (excludedDevices.length > 0) {
        console.log('❌ Dispositivos excluidos (cámaras virtuales):', excludedDevices.map(d => ({
          label: d.label,
          reason: 'Cámara virtual detectada'
        })));
      }

      setAvailableDevices(irDevices);

      // Auto-select first IR device if available
      if (irDevices.length > 0 && !selectedDeviceId) {
        setSelectedDeviceId(irDevices[0].deviceId);
      }

      // Show detailed error if no IR devices found
      if (irDevices.length === 0) {
        console.log('❌ No se encontraron módulos IR especializados');
        console.log('💡 Verifica que:');
        console.log('   - El módulo IR esté conectado por USB');
        console.log('   - Los drivers estén instalados correctamente');
        console.log('   - El dispositivo esté encendido');
        console.log('   - No haya otros dispositivos interfiriendo');
        console.log('   - Solo se muestran dispositivos especializados (no cámaras virtuales)');
        
        // Show what devices were found but excluded
        if (allVideoDevices.length > 0) {
          console.log('📹 Dispositivos de video encontrados pero excluidos:');
          allVideoDevices.forEach(device => {
            const label = device.label.toLowerCase();
            let reason = 'Dispositivo estándar';
            if (label.includes('obs') || label.includes('virtual')) {
              reason = 'Cámara virtual (OBS, etc.)';
            } else if (label.includes('laptop') || label.includes('integrated')) {
              reason = 'Cámara integrada de laptop';
            }
            console.log(`   - ${device.label}: ${reason}`);
          });
        }
      }
    } catch (err) {
      console.error('Error enumerating IR devices:', err);
      setError('No se pudieron enumerar los módulos IR');
    }
  }, [selectedDeviceId]);

  // Start camera stream
  const startCamera = useCallback(async () => {
    try {
      setError('');
      setIsStreaming(true);

      // Stop any existing stream
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
      }

      const constraints: MediaStreamConstraints = {
        video: {
          deviceId: selectedDeviceId ? { exact: selectedDeviceId } : undefined,
          width: { ideal: 1920 },
          height: { ideal: 1080 },
          frameRate: { ideal: 30 }
        }
      };

      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      streamRef.current = stream;

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }

      setIsStreaming(true);
    } catch (err) {
      console.error('Error starting camera:', err);
      setError(`Error al iniciar la cámara: ${err instanceof Error ? err.message : 'Error desconocido'}`);
      setIsStreaming(false);
    }
  }, [selectedDeviceId]);

  // Stop camera stream
  const stopCamera = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    setIsStreaming(false);
  }, []);

  // Capture photo
  const capturePhoto = useCallback(() => {
    if (!videoRef.current || !canvasRef.current || !isStreaming) return;

    setIsCapturing(true);

    const video = videoRef.current;
    const canvas = canvasRef.current;
    const context = canvas.getContext('2d');

    if (!context) {
      setError('Error al acceder al contexto del canvas');
      setIsCapturing(false);
      return;
    }

    // Set canvas dimensions to match video
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    // Draw current video frame to canvas
    context.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Get image data as base64
    const dataUrl = canvas.toDataURL('image/jpeg', 0.9);

    // Create photo object
    const photo: CapturedPhoto = {
      id: Date.now().toString(),
      dataUrl,
      timestamp: new Date(),
      deviceId: selectedDeviceId
    };

    // Add to captured photos
    setCapturedPhotos(prev => [photo, ...prev]);

    // Notify parent component
    if (onPhotoCapture) {
      onPhotoCapture(dataUrl);
    }

    setIsCapturing(false);
  }, [isStreaming, selectedDeviceId, onPhotoCapture]);

  // Delete photo
  const deletePhoto = useCallback((photoId: string) => {
    setCapturedPhotos(prev => prev.filter(photo => photo.id !== photoId));
  }, []);

  // Download photo
  const downloadPhoto = useCallback((photo: CapturedPhoto) => {
    const link = document.createElement('a');
    link.href = photo.dataUrl;
    link.download = `meibomio-ir-capture-${photo.timestamp.getTime()}.jpg`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  }, []);

  // Load available devices on mount
  useEffect(() => {
    getAvailableDevices();

    // Listen for device changes
    const handleDeviceChange = () => {
      getAvailableDevices();
    };

    navigator.mediaDevices.addEventListener('devicechange', handleDeviceChange);

    return () => {
      navigator.mediaDevices.removeEventListener('devicechange', handleDeviceChange);
      stopCamera();
    };
  }, [getAvailableDevices, stopCamera]);

  // Handle device changes - reset selection if current device is no longer available
  useEffect(() => {
    if (selectedDeviceId && availableDevices.length > 0) {
      const deviceExists = availableDevices.some(device => device.deviceId === selectedDeviceId);
      if (!deviceExists) {
        setSelectedDeviceId(availableDevices[0].deviceId);
      }
    } else if (selectedDeviceId === 'no-devices' && availableDevices.length > 0) {
      setSelectedDeviceId(availableDevices[0].deviceId);
    }
  }, [availableDevices, selectedDeviceId]);

  return (
    <div className="space-y-4 overflow-visible">
      {/* Camera Controls */}
      <Card className="border-border">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Camera className="h-5 w-5" />
            Captura de Imagen IR - Módulo Especializado
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4 overflow-visible">
          {/* Device Selection */}
          <div className="flex flex-col gap-2">
            <label className="text-sm font-medium">Seleccionar Módulo IR:</label>
            <Select
              value={selectedDeviceId}
              onValueChange={(value) => {
                // Prevent selecting "no-devices"
                if (value !== 'no-devices') {
                  setSelectedDeviceId(value);
                }
              }}
              disabled={availableDevices.length === 0}
            >
              <SelectTrigger>
                <SelectValue placeholder={availableDevices.length === 0 ? "No hay módulos IR disponibles" : "Selecciona un módulo IR"} />
              </SelectTrigger>
              <SelectContent>
                                 {availableDevices.length === 0 ? (
                   <div className="p-2 text-sm text-muted-foreground">
                     No hay módulos IR especializados conectados
                   </div>
                 ) : (
                  availableDevices.map((device) => (
                                         <SelectItem key={device.deviceId} value={device.deviceId}>
                       {device.label && device.label.trim() !== ''
                         ? device.label
                         : `Módulo IR ${device.deviceId.slice(0, 8)}`}
                     </SelectItem>
                  ))
                )}
              </SelectContent>
            </Select>
            <div className="flex gap-2">
              <Button
                variant="outline"
                size="sm"
                onClick={getAvailableDevices}
                className="flex-1 border-border"
              >
                🔄 Actualizar Dispositivos
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={() => setShowDeviceInfo(!showDeviceInfo)}
                className="flex-1 border-border"
              >
                {showDeviceInfo ? 'Ocultar Info' : '🔍 Ver Info Dispositivos'}
              </Button>
              
              <Button
                variant="outline"
                size="sm"
                onClick={() => setShowAllDevices(!showAllDevices)}
                className="flex-1 border-border"
              >
                {showAllDevices ? 'Ocultar Todos' : '📹 Ver Todos'}
              </Button>
            </div>
                         <div className="text-xs text-muted-foreground">
               💡 <strong>Si no detecta tu módulo IR:</strong>
               <br />• Abre DevTools (F12) → Console para ver logs
               <br />• Verifica que el módulo IR esté conectado por USB
               <br />• Asegúrate de que los drivers estén instalados
               <br />• Prueba diferentes puertos USB
             </div>

                      {/* Device Information */}
          {showDeviceInfo && (
            <Card className="mt-4 border-border">
              <CardHeader>
                                 <CardTitle className="text-sm">📋 Información de Módulos IR Detectados</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                                 {availableDevices.length === 0 ? (
                   <div className="text-center py-4 text-muted-foreground">
                     <p>No hay módulos IR especializados conectados</p>
                     <p className="text-xs mt-1">
                       Conecta tu módulo IR USB para meibografía
                     </p>
                   </div>
                ) : (
                  <>
                                         <div className="text-xs text-muted-foreground mb-2">
                       💡 <strong>Módulos IR compatibles:</strong>
                       <br />• 🔴 <strong>Módulos IR USB</strong> - Para meibografía profesional
                       <br />• 🔴 <strong>Cámaras infrarrojas</strong> - Especializadas en IR
                       <br />• 🔴 <strong>Dispositivos médicos</strong> - Para análisis clínico
                       <br />• 🔴 <strong>Cámaras especializadas</strong> - Con filtros IR
                       <br />
                       <strong>📋 Características esperadas:</strong>
                       <br />• <strong>Resolución:</strong> 1920x1080 o superior
                       <br />• <strong>FPS:</strong> 30fps para captura fluida
                       <br />• <strong>Conectividad:</strong> USB 2.0/3.0
                       <br />• <strong>Drivers:</strong> Plug & Play
                     </div>
                    {availableDevices.map((device, index) => (
                      <div key={device.deviceId} className="p-2 bg-muted rounded text-xs">
                                                 <div className="flex justify-between items-center">
                           <span className="font-medium">
                             {device.label || `Módulo IR ${index + 1}`}
                           </span>
                          <span className="text-muted-foreground">
                            {selectedDeviceId === device.deviceId ? '✅ Seleccionada' : ''}
                          </span>
                        </div>
                        <div className="text-muted-foreground mt-1">
                          ID: {device.deviceId.substring(0, 12)}...
                          <br />
                          Grupo: {device.groupId ? device.groupId.substring(0, 8) + '...' : 'N/A'}
                        </div>
                                                 <div className="mt-1">
                           <span className="inline-flex items-center px-2 py-1 rounded-full text-xs bg-red-100 text-red-800">
                             🔴 Módulo IR Especializado
                           </span>
                         </div>
                      </div>
                    ))}
                  </>
                )}
              </CardContent>
            </Card>
          )}
          
          {/* All Devices Information (for debugging) */}
          {showAllDevices && (
            <Card className="mt-4 border-border">
              <CardHeader>
                <CardTitle className="text-sm">📹 Todos los Dispositivos de Video Detectados</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <div className="text-xs text-muted-foreground mb-2">
                  💡 <strong>Esta lista muestra TODOS los dispositivos, incluyendo los excluidos:</strong>
                  <br />• 🔴 <strong>Verde:</strong> Módulos IR especializados (compatibles)
                  <br />• ❌ <strong>Rojo:</strong> Cámaras virtuales y no compatibles (excluidas)
                  <br />• ⚠️ <strong>Amarillo:</strong> Dispositivos estándar (no especializados)
                </div>
                
                {allVideoDevices.length === 0 ? (
                  <div className="text-center py-4 text-muted-foreground">
                    <p>No se pudieron obtener los dispositivos</p>
                  </div>
                ) : (
                  allVideoDevices.map((device, index) => {
                    const label = device.label.toLowerCase();
                    let status = 'standard';
                    let statusText = 'Dispositivo Estándar';
                    let statusColor = 'bg-yellow-100 text-yellow-800';
                    
                    if (label.includes('obs') || label.includes('virtual') || 
                        label.includes('screen capture') || label.includes('desktop')) {
                      status = 'excluded';
                      statusText = 'Cámara Virtual (Excluida)';
                      statusColor = 'bg-red-100 text-red-800';
                    } else if (availableDevices.some(d => d.deviceId === device.deviceId)) {
                      status = 'compatible';
                      statusText = 'Módulo IR Especializado';
                      statusColor = 'bg-green-100 text-green-800';
                    }
                    
                    return (
                      <div key={device.deviceId} className="p-2 bg-muted rounded text-xs">
                        <div className="flex justify-between items-center">
                          <span className="font-medium">
                            {device.label || `Dispositivo ${index + 1}`}
                          </span>
                          <span className={`inline-flex items-center px-2 py-1 rounded-full text-xs ${statusColor}`}>
                            {status === 'compatible' ? '🔴' : status === 'excluded' ? '❌' : '⚠️'} {statusText}
                          </span>
                        </div>
                        <div className="text-muted-foreground mt-1">
                          ID: {device.deviceId.substring(0, 12)}...
                          <br />
                          Grupo: {device.groupId ? device.groupId.substring(0, 8) + '...' : 'N/A'}
                        </div>
                      </div>
                    );
                  })
                )}
              </CardContent>
            </Card>
          )}
          </div>

          {/* Camera Controls */}
          <div className="flex gap-2">
            {!isStreaming ? (
              <Button
                onClick={startCamera}
                disabled={!selectedDeviceId || availableDevices.length === 0 || selectedDeviceId === 'no-devices'}
              >
                                 <Camera className="mr-2 h-4 w-4" />
                 Iniciar Módulo IR
              </Button>
            ) : (
              <Button variant="destructive" onClick={stopCamera}>
                                 <CameraOff className="mr-2 h-4 w-4" />
                 Detener Módulo IR
              </Button>
            )}

            <Button
              onClick={capturePhoto}
              disabled={!isStreaming || isCapturing}
              variant="secondary"
            >
              {isCapturing ? (
                <>
                  <AlertCircle className="mr-2 h-4 w-4 animate-spin" />
                  Capturando...
                </>
              ) : (
                <>
                  <CheckCircle className="mr-2 h-4 w-4" />
                  Capturar Foto
                </>
              )}
            </Button>
          </div>

                     {/* Connection Tips */}
           {availableDevices.length === 0 && (
             <Alert className="border-red-200 bg-red-50">
               <AlertCircle className="h-4 w-4 text-red-600" />
               <AlertDescription className="text-red-800">
                 <strong>🔴 Para conectar tu módulo IR:</strong>
                 <br />1. Conecta el módulo IR por USB
                 <br />2. Espera 5-10 segundos para que Windows lo detecte
                 <br />3. Haz clic en &quot;🔄 Actualizar Dispositivos&quot;
                 <br />4. Verifica que los drivers estén instalados
                 <br />
                 <strong>🔍 Debug:</strong> Abre DevTools (F12) → Console para ver logs detallados
               </AlertDescription>
             </Alert>
           )}

          {/* Error Display */}
          {error && (
            <Alert className="border-red-200 bg-red-50">
              <AlertCircle className="h-4 w-4 text-red-600" />
              <AlertDescription className="text-red-800">{error}</AlertDescription>
            </Alert>
          )}
        </CardContent>
      </Card>

      {/* Video Preview */}
      {isStreaming && (
        <Card className="border-border">
          <CardHeader>
            <CardTitle>Vista Previa del Módulo IR</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="relative">
              <video
                ref={videoRef}
                className="w-full max-h-96 bg-black rounded-lg"
                playsInline
                muted
                autoPlay
              />
              <canvas ref={canvasRef} className="hidden" />
            </div>
                         <p className="text-sm text-muted-foreground mt-2 text-center">
               Asegúrate de que el párpado esté bien enfocado y centrado para análisis de meibografía
             </p>
          </CardContent>
        </Card>
      )}

      {/* Captured Photos Gallery */}
      {capturedPhotos.length > 0 && (
        <Card className="border-border">
          <CardHeader>
            <CardTitle>Imágenes IR Capturadas ({capturedPhotos.length})</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {capturedPhotos.map((photo) => (
                <div key={photo.id} className="relative group border rounded-lg p-2">
                                     <img
                     src={photo.dataUrl}
                     alt={`Imagen IR ${photo.timestamp.toLocaleTimeString()}`}
                     className="w-full h-32 object-cover rounded"
                   />
                  <div className="absolute top-2 right-2 flex gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                    <Button
                      size="sm"
                      variant="secondary"
                      onClick={() => downloadPhoto(photo)}
                      className="h-6 w-6 p-0 border-border"
                    >
                      <Download className="h-3 w-3" />
                    </Button>
                    <Button
                      size="sm"
                      variant="destructive"
                      onClick={() => deletePhoto(photo.id)}
                      className="h-6 w-6 p-0"
                    >
                      <Trash2 className="h-3 w-3" />
                    </Button>
                  </div>
                  <p className="text-xs text-muted-foreground mt-1 text-center">
                    {photo.timestamp.toLocaleTimeString()}
                  </p>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
