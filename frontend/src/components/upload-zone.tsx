"use client";

import { useCallback } from "react";
import { useDropzone } from "react-dropzone";
import { Upload, FileImage, X } from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";

interface UploadZoneProps {
  onFileSelect: (file: File | null) => void;
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

  const handleRemoveFile = () => {
    onFileSelect(null);
  };

  return (
    <div className="space-y-4">
      {!selectedFile ? (
        <Card className="border-border">
          <CardContent className="p-4 sm:p-6">
            <div
              {...getRootProps()}
              className={`border-2 border-dashed rounded-lg p-6 sm:p-8 text-center cursor-pointer transition-all ${
                isDragActive
                  ? "border-primary bg-primary/10"
                  : "border-muted-foreground/25 hover:border-primary/50"
              }`}
            >
              <input {...getInputProps()} />
              <div className="flex flex-col items-center gap-3 sm:gap-4">
                <Upload className="h-8 w-8 sm:h-12 sm:w-12 text-muted-foreground" />
                <div className="space-y-2">
                  <p className="text-base sm:text-lg font-medium">
                    {isDragActive
                      ? "Suelta la imagen aquí"
                      : "Arrastra y suelta una imagen"}
                  </p>
                  <p className="text-xs sm:text-sm text-muted-foreground">
                    o haz clic para seleccionar (JPG, JPEG, PNG)
                  </p>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      ) : (
        <Card className="border-border">
          <CardContent className="p-4 sm:p-6">
            <div className="flex items-center gap-3 sm:gap-4 p-3 sm:p-4 bg-muted/50 rounded-lg">
              <FileImage className="h-6 w-6 sm:h-8 sm:w-8 text-primary flex-shrink-0" />
              <div className="flex-1 min-w-0">
                <p className="font-medium text-sm sm:text-base truncate">{selectedFile.name}</p>
                <p className="text-xs sm:text-sm text-muted-foreground">
                  {(selectedFile.size / 1024 / 1024).toFixed(2)} MB
                </p>
              </div>
              <Button
                variant="ghost"
                size="sm"
                onClick={handleRemoveFile}
                className="text-muted-foreground hover:text-destructive flex-shrink-0"
              >
                <X className="h-4 w-4" />
              </Button>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
} 