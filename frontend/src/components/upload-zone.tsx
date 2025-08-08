"use client";

import { useCallback } from "react";
import { useDropzone } from "react-dropzone";
import { Upload, FileImage } from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";

interface UploadZoneProps {
  onFileSelect: (file: File) => void;
  selectedFile?: File | null;
}

export function UploadZone({ onFileSelect, selectedFile }: UploadZoneProps) {
  const onDrop = useCallback(
    (acceptedFiles: File[]) => {
      if (acceptedFiles.length > 0) {
        onFileSelect(acceptedFiles[0]);
      }
    },
    [onFileSelect]
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      "image/*": [".jpg", ".jpeg", ".png"],
    },
    multiple: false,
  });

  return (
    <Card className="w-full">
      <CardContent className="p-6">
        <div
          {...getRootProps()}
          className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors ${
            isDragActive
              ? "border-primary bg-primary/10"
              : "border-muted-foreground/25 hover:border-primary/50"
          }`}
        >
          <input {...getInputProps()} />
          <div className="flex flex-col items-center gap-4">
            {selectedFile ? (
              <>
                <FileImage className="h-12 w-12 text-primary" />
                <div>
                  <p className="text-lg font-medium">Archivo seleccionado:</p>
                  <p className="text-sm text-muted-foreground">
                    {selectedFile.name}
                  </p>
                </div>
                <Button variant="outline" size="sm">
                  Cambiar archivo
                </Button>
              </>
            ) : (
              <>
                <Upload className="h-12 w-12 text-muted-foreground" />
                <div>
                  <p className="text-lg font-medium">
                    {isDragActive
                      ? "Suelta la imagen aquí"
                      : "Arrastra y suelta una imagen"}
                  </p>
                  <p className="text-sm text-muted-foreground">
                    o haz clic para seleccionar (JPG, JPEG, PNG)
                  </p>
                </div>
              </>
            )}
          </div>
        </div>
      </CardContent>
    </Card>
  );
} 