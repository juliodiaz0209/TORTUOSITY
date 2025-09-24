/**
 * CLAHE Optimizado - Implementación con Web Workers para Frontend
 * Basado en CLAHE_OPTIMIZED.html
 */

// Web Worker code para CLAHE
const CLAHE_WORKER_CODE = `
// CLAHE Worker - Procesamiento paralelo optimizado
const roundPositive = (a) => (a + 0.5) | 0; // faster que Math.round para positivos

// Clipping en-place reutilizando buffers (evita ~900k allocs de 1KB)
function clipHistogramInPlace(src, dst, limit) {
  const nbins = src.length - 1; // src y dst tienen (bins+1)
  // copiar src -> dst manualmente (más rápido que set para tamaños pequeños repetidos)
  for (let i = 0; i <= nbins; i++) dst[i] = src[i];
  let clippedEntries = 0, prev;
  do {
    prev = clippedEntries;
    clippedEntries = 0;
    for (let i = 0; i <= nbins; i++) {
      const d = dst[i] - limit;
      if (d > 0) { clippedEntries += d; dst[i] = limit; }
    }
    if (!clippedEntries) break;
    const d = (clippedEntries / (nbins + 1)) | 0;
    const m = clippedEntries - d * (nbins + 1);
    if (d) for (let i = 0; i <= nbins; i++) dst[i] += d;
    if (m) {
      const s = (nbins / m) | 0 || 1;
      for (let i = 0, k = 0; k < m && i <= nbins; i += s, k++) dst[i]++;
    }
  } while (clippedEntries !== prev);
}

const clahe2DChunk = (fullData, width, height, blockRadius, bins, slope, startY, endY) => {
  const outH = endY - startY;
  const dst = new Uint8Array(outH * width);
  const hist = new Uint32Array(bins + 1);
  const clipped = new Uint32Array(bins + 1);
  const binScale = bins / 255.0;
  if (startY >= endY) return dst;

  for (let yLocal = 0; yLocal < outH; yLocal++) {
    const y = startY + yLocal;
    const yMin = y - blockRadius < 0 ? 0 : y - blockRadius;
    const yMax = y + blockRadius + 1 > height ? height : y + blockRadius + 1; // excl
    const h = yMax - yMin;

    // Hist inicial para x=0
    hist.fill(0);
    let xInitMax = blockRadius < width - 1 ? blockRadius : width - 1;
    for (let yi = yMin; yi < yMax; yi++) {
      const rowBase = yi * width;
      for (let xi = 0; xi <= xInitMax; xi++) {
        const v = (fullData[rowBase + xi] * binScale + 0.5) | 0;
        hist[v]++;
      }
    }

    for (let x = 0; x < width; x++) {
      const centerVal = fullData[y * width + x];
      const vCenter = (centerVal * binScale + 0.5) | 0;
      const xMin = x - blockRadius < 0 ? 0 : x - blockRadius;
      let xMax = x + blockRadius + 1; if (xMax > width) xMax = width; // excl
      const w = xMax - xMin;
      const n = h * w;
      const limit = ((slope * n / bins) + 0.5) | 0;

      // quitar columna removida
      if (xMin > 0) {
        const remX = xMin - 1;
        for (let yi = yMin; yi < yMax; yi++) {
          const val = fullData[yi * width + remX];
          hist[(val * binScale + 0.5) | 0]--;
        }
      }
      // añadir nueva columna
      if (xMax <= width) {
        const addX = xMax - 1;
        for (let yi = yMin; yi < yMax; yi++) {
          const val = fullData[yi * width + addX];
          hist[(val * binScale + 0.5) | 0]++;
        }
      }

      // clipping
      clipHistogramInPlace(hist, clipped, limit);
      // primer bin no cero
      let hMin = 0; while (hMin <= bins && clipped[hMin] === 0) hMin++;
      const cdfMin = clipped[hMin] || 1;
      // cdf
      let cdf = 0; for (let i = hMin; i <= vCenter; i++) cdf += clipped[i];
      const denom = n - cdfMin;
      let outV = 0;
      if (denom > 0) outV = ((cdf - cdfMin) * 255 / denom + 0.5) | 0;
      dst[yLocal * width + x] = outV > 255 ? 255 : (outV < 0 ? 0 : outV);
    }
  }
  return dst;
};

// Escuchar mensajes del hilo principal
self.onmessage = function(e) {
  const { gray8, width, height, blockRadius, bins, slope, startY, endY, chunkId } = e.data;
  
  try {
    const startTime = performance.now();
    const result = clahe2DChunk(gray8, width, height, blockRadius, bins, slope, startY, endY);
    const endTime = performance.now();
    self.postMessage({
      type: 'chunk_complete',
      chunkId,
      result,
      processingTime: endTime - startTime,
      startY,
      endY
    });
  } catch (error) {
    self.postMessage({ type: 'error', chunkId, error: error.message });
  }
};
`;

// Utilidades de Color
const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));

function srgbToLinear(u: number): number { 
  u /= 255; 
  return (u <= 0.04045) ? u / 12.92 : Math.pow((u + 0.055) / 1.055, 2.4); 
}

function linearToSrgb(u: number): number { 
  return (u <= 0.0031308) ? 12.92 * u : 1.055 * Math.pow(u, 1 / 2.4) - 0.055; 
}

function rgbToXyz(r: number, g: number, b: number): [number, number, number] {
  const R = srgbToLinear(r), G = srgbToLinear(g), B = srgbToLinear(b);
  const X = 0.4124564 * R + 0.3575761 * G + 0.1804375 * B;
  const Y = 0.2126729 * R + 0.7151522 * G + 0.0721750 * B;
  const Z = 0.0193339 * R + 0.1191920 * G + 0.9503041 * B;
  return [X, Y, Z];
}

function xyzToRgb(X: number, Y: number, Z: number): [number, number, number] {
  let R = 3.2404542 * X + -1.5371385 * Y + -0.4985314 * Z;
  let G = -0.9692660 * X + 1.8760108 * Y + 0.0415560 * Z;
  let B = 0.0556434 * X + -0.2040259 * Y + 1.0572252 * Z;
  R = clamp(Math.round(linearToSrgb(R) * 255), 0, 255);
  G = clamp(Math.round(linearToSrgb(G) * 255), 0, 255);
  B = clamp(Math.round(linearToSrgb(B) * 255), 0, 255);
  return [R, G, B];
}

function fLab(t: number): number { 
  const e = 216 / 24389, k = 24389 / 27; 
  return t > e ? Math.cbrt(t) : (k * t + 16) / 116; 
}

function rgbToLabL(r: number, g: number, b: number): number {
  const Xn = 0.95047, Yn = 1.00000, Zn = 1.08883;
  const [X, Y, Z] = rgbToXyz(r, g, b);
  const fx = fLab(X / Xn), fy = fLab(Y / Yn), fz = fLab(Z / Zn);
  const L = 116 * fy - 16;
  return L;
}

function labLToRgb(L: number, a: number, b: number, r0: number, g0: number, b0: number): [number, number, number] {
  const [Xn, Yn, Zn] = [0.95047, 1.00000, 1.08883];
  const [X0, Y0, Z0] = rgbToXyz(r0, g0, b0);
  const fx0 = fLab(X0 / Xn), fy0 = fLab(Y0 / Yn), fz0 = fLab(Z0 / Zn);
  const a0 = 500 * (fx0 - fy0);
  const b_lab = 200 * (fy0 - fz0);

  const fy = (L + 16) / 116;
  const fx = fy + (a0 / 500);
  const fz = fy - (b_lab / 200);
  const e = 216 / 24389, k = 24389 / 27;
  const fx3 = fx * fx * fx, fy3 = fy * fy * fy, fz3 = fz * fz * fz;
  const xr = (fx3 > e) ? fx3 : (116 * fx - 16) / k;
  const yr = (fy3 > e) ? fy3 : (116 * fy - 16) / k;
  const zr = (fz3 > e) ? fz3 : (116 * fz - 16) / k;
  const X = xr * Xn, Y = yr * Yn, Z = zr * Zn;
  return xyzToRgb(X, Y, Z);
}

// Interfaces
interface CLAHEResult {
  result: Uint8Array;
  processingTime: number;
  chunkCount: number;
  workerCount: number;
}

interface CLAHEProgressCallback {
  (progress: number, status: string): void;
}

interface CLAHEChunk {
  startY: number;
  endY: number;
}

interface CLAHEChunkResult {
  result: Uint8Array;
  startY: number;
  endY: number;
}

// Clase principal CLAHE Optimizado
export class CLAHEOptimized {
  private workers: Worker[] = [];
  private workerCount = 0;
  private isProcessing = false;
  private chunkSize = 64;
  private progressCallback: CLAHEProgressCallback | null = null;
  private completeCallback: ((result: CLAHEResult) => void) | null = null;
  private errorCallback: ((error: Error) => void) | null = null;

  async initialize(workerCount = 2): Promise<void> {
    this.workerCount = Math.min(workerCount, navigator.hardwareConcurrency || 4);
    this.workers = [];
    
    for (let i = 0; i < this.workerCount; i++) {
      const blob = new Blob([CLAHE_WORKER_CODE], { type: 'application/javascript' });
      const worker = new Worker(URL.createObjectURL(blob));
      this.workers.push(worker);
    }
    console.log(`Initialized ${this.workerCount} workers (requested: ${workerCount})`);
  }

  async processImage(
    gray8: Uint8Array, 
    width: number, 
    height: number, 
    blockRadius: number, 
    bins: number, 
    slope: number, 
    chunkSize = 64,
    progressCallback?: CLAHEProgressCallback
  ): Promise<CLAHEResult> {
    if (this.isProcessing) {
      throw new Error('Already processing an image');
    }

    this.isProcessing = true;
    this.chunkSize = chunkSize;
    this.progressCallback = progressCallback || null;
    
    return new Promise((resolve, reject) => {
      this.completeCallback = resolve;
      this.errorCallback = reject;
      
      const chunks = this.createChunks(height, chunkSize);
      const results = new Array<CLAHEChunkResult | null>(chunks.length).fill(null);
      let completedChunks = 0;
      let totalProcessingTime = 0;
      
      // Configurar callbacks de workers
      this.workers.forEach((worker, workerId) => {
        worker.onmessage = (e) => {
          if (e.data.type === 'chunk_complete') {
            const { result, processingTime, startY, endY, chunkId } = e.data;
            
            console.log(`Main: Received chunk ${chunkId} (startY=${startY}, endY=${endY}) with ${result.length} values from worker ${workerId}`);
            results[chunkId] = { result, startY, endY };
            totalProcessingTime += processingTime;
            completedChunks++;
            
            // Actualizar progreso
            const progress = (completedChunks / chunks.length) * 100;
            if (this.progressCallback) {
              this.progressCallback(progress, `Processing chunk ${completedChunks}/${chunks.length}`);
            }
            
            // Verificar si todos los chunks están completos
            if (completedChunks === chunks.length) {
              console.log(`All ${chunks.length} chunks completed. Combining results...`);
              
              this.isProcessing = false;
              
              // Combinar resultados
              const finalResult = this.combineChunks(results as CLAHEChunkResult[], width, height);
              
              if (this.completeCallback) {
                this.completeCallback({
                  result: finalResult,
                  processingTime: totalProcessingTime,
                  chunkCount: chunks.length,
                  workerCount: this.workerCount
                });
              }
            }
          } else if (e.data.type === 'error') {
            this.isProcessing = false;
            
            if (this.errorCallback) {
              this.errorCallback(new Error(`Worker error: ${e.data.error}`));
            }
          }
        };
      });
      
      // Procesar chunks en paralelo
      console.log(`Processing ${chunks.length} chunks with ${this.workerCount} workers in parallel`);
      
      chunks.forEach((chunk, chunkId) => {
        const workerId = chunkId % this.workerCount;
        const worker = this.workers[workerId];
        console.log(`Sending chunk ${chunkId} (startY=${chunk.startY}, endY=${chunk.endY}) to worker ${workerId}`);
        worker.postMessage({
          gray8: gray8,
          width,
          height,
          blockRadius,
          bins,
          slope,
          startY: chunk.startY,
          endY: chunk.endY,
          chunkId
        });
      });
      
      console.log(`All ${chunks.length} chunks sent to workers. Processing in parallel...`);
    });
  }

  private createChunks(height: number, chunkSize: number): CLAHEChunk[] {
    const chunks: CLAHEChunk[] = [];
    for (let startY = 0; startY < height; startY += chunkSize) {
      const endY = Math.min(startY + chunkSize, height);
      chunks.push({ startY, endY });
    }
    return chunks;
  }

  private combineChunks(chunkResults: CLAHEChunkResult[], width: number, height: number): Uint8Array {
    const finalResult = new Uint8Array(width * height);
    
    console.log(`Combining ${chunkResults.length} chunks for image ${width}x${height}`);
    
    // Verificar que todos los chunks estén presentes y ordenarlos
    const sortedChunks = chunkResults
      .filter(chunk => chunk && chunk.result && chunk.result.length > 0)
      .sort((a, b) => a.startY - b.startY);
    
    console.log(`Found ${sortedChunks.length} valid chunks out of ${chunkResults.length} total`);
    
    sortedChunks.forEach(({ result, startY, endY }, index) => {
      const chunkHeight = endY - startY;
      const chunkWidth = width;
      
      console.log(`Processing chunk ${index}: startY=${startY}, endY=${endY}, height=${chunkHeight}, result.length=${result.length}`);
      
      // Verificar que el chunk tenga el tamaño esperado
      const expectedChunkSize = chunkHeight * chunkWidth;
      if (result.length !== expectedChunkSize) {
        console.error(`Chunk ${index} size mismatch: expected ${expectedChunkSize}, got ${result.length}`);
        return;
      }
      
      // Copiar cada fila del chunk al resultado final
      for (let y = 0; y < chunkHeight; y++) {
        const srcStart = y * chunkWidth;
        const srcEnd = srcStart + chunkWidth;
        const dstStart = (startY + y) * width;
        
        // Verificar que los índices estén dentro de los límites
        if (srcEnd <= result.length && dstStart + chunkWidth <= finalResult.length) {
          finalResult.set(result.subarray(srcStart, srcEnd), dstStart);
        } else {
          console.warn(`Index out of bounds: srcStart=${srcStart}, srcEnd=${srcEnd}, dstStart=${dstStart}`);
        }
      }
    });
    
    return finalResult;
  }

  terminate(): void {
    this.workers.forEach(worker => worker.terminate());
    this.workers = [];
    this.isProcessing = false;
  }
}

// Función de utilidad para aplicar CLAHE a una imagen
export async function applyCLAHEToImage(
  imageData: ImageData,
  options: {
    blockSize?: number;
    bins?: number;
    slope?: number;
    chunkSize?: number;
    workerCount?: number;
    onProgress?: CLAHEProgressCallback;
  } = {}
): Promise<ImageData> {
  const {
    blockSize = 64,
    bins = 256,
    slope = 3,
    chunkSize = 64,
    workerCount = 2,
    onProgress
  } = options;

  const w = imageData.width;
  const h = imageData.height;
  const src = new Uint8ClampedArray(imageData.data);
  const outData = new Uint8ClampedArray(src.length);
  const blockRadius = Math.floor(blockSize / 2);

  // Inicializar procesador CLAHE
  const claheProcessor = new CLAHEOptimized();
  await claheProcessor.initialize(workerCount);

  try {
    // Convertir a Lab y procesar L*
    const Larr = new Float32Array(w * h);
    for (let i = 0, p = 0; i < src.length; i += 4, p++) {
      Larr[p] = rgbToLabL(src[i], src[i + 1], src[i + 2]);
    }
    
    const l8 = new Uint8Array(w * h);
    for (let p = 0; p < w * h; p++) {
      l8[p] = Math.round((Larr[p] / 100) * 255);
    }

    // Aplicar CLAHE optimizado
    const result = await claheProcessor.processImage(
      l8, w, h, blockRadius, bins, slope, chunkSize, onProgress
    );
    
    // Reconstruir imagen RGB
    for (let i = 0, p = 0; i < src.length; i += 4, p++) {
      const Lnew = (result.result[p] / 255) * 100;
      const [r, g, b] = labLToRgb(Lnew, 0, 0, src[i], src[i + 1], src[i + 2]);
      outData[i] = r;
      outData[i + 1] = g;
      outData[i + 2] = b;
      outData[i + 3] = src[i + 3];
    }

    return new ImageData(outData, w, h);
  } finally {
    claheProcessor.terminate();
  }
}

// Función de utilidad para convertir File a ImageData
export function fileToImageData(file: File): Promise<ImageData> {
  return new Promise((resolve, reject) => {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    const img = new Image();
    
    img.onload = () => {
      canvas.width = img.width;
      canvas.height = img.height;
      ctx?.drawImage(img, 0, 0);
      const imageData = ctx?.getImageData(0, 0, img.width, img.height);
      if (imageData) {
        resolve(imageData);
      } else {
        reject(new Error('Failed to get image data'));
      }
      URL.revokeObjectURL(img.src);
    };
    
    img.onerror = () => {
      URL.revokeObjectURL(img.src);
      reject(new Error('Failed to load image'));
    };
    
    img.src = URL.createObjectURL(file);
  });
}

// Función de utilidad para convertir ImageData a File
export function imageDataToFile(imageData: ImageData, filename: string, quality = 0.9): Promise<File> {
  return new Promise((resolve) => {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    canvas.width = imageData.width;
    canvas.height = imageData.height;
    ctx?.putImageData(imageData, 0, 0);
    
    canvas.toBlob((blob) => {
      if (blob) {
        const file = new File([blob], filename, { type: 'image/png' });
        resolve(file);
      }
    }, 'image/png', quality);
  });
}