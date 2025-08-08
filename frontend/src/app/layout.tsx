import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "Análisis de Tortuosidad Avanzado",
  description: "Análisis de imágenes biomédicas usando PyTorch (Mask R-CNN & UNet) para evaluar la tortuosidad de glándulas de Meibomio",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="es">
      <body className={inter.className}>{children}</body>
    </html>
  );
}
