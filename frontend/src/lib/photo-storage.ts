export interface StoredPhoto {
  id: string;
  dataUrl: string;
  timestamp: Date;
  deviceId?: string;
  fileName?: string;
  analysisResults?: {
    avgTortuosity: number;
    numGlands: number;
    individualTortuosities: number[];
  };
}

class PhotoStorage {
  private dbName = 'TortuosityPhotos';
  private dbVersion = 1;
  private storeName = 'photos';

  private async openDB(): Promise<IDBDatabase> {
    return new Promise((resolve, reject) => {
      const request = indexedDB.open(this.dbName, this.dbVersion);

      request.onerror = () => reject(request.error);
      request.onsuccess = () => resolve(request.result);

      request.onupgradeneeded = (event) => {
        const db = (event.target as IDBOpenDBRequest).result;

        // Create photos store if it doesn't exist
        if (!db.objectStoreNames.contains(this.storeName)) {
          const store = db.createObjectStore(this.storeName, { keyPath: 'id' });
          store.createIndex('timestamp', 'timestamp', { unique: false });
          store.createIndex('deviceId', 'deviceId', { unique: false });
        }
      };
    });
  }

  async savePhoto(photo: StoredPhoto): Promise<void> {
    try {
      const db = await this.openDB();
      const transaction = db.transaction([this.storeName], 'readwrite');
      const store = transaction.objectStore(this.storeName);

      // Convert Date to ISO string for storage
      const photoForStorage = {
        ...photo,
        timestamp: photo.timestamp.toISOString()
      };

      return new Promise((resolve, reject) => {
        const request = store.put(photoForStorage);
        request.onsuccess = () => resolve();
        request.onerror = () => reject(request.error);
      });
    } catch (error) {
      console.error('Error saving photo:', error);
      throw error;
    }
  }

  async getAllPhotos(): Promise<StoredPhoto[]> {
    try {
      const db = await this.openDB();
      const transaction = db.transaction([this.storeName], 'readonly');
      const store = transaction.objectStore(this.storeName);

      return new Promise((resolve, reject) => {
        const request = store.getAll();
        request.onsuccess = () => {
          // Convert ISO strings back to Date objects
          const photos = request.result.map((photo: any) => ({
            ...photo,
            timestamp: new Date(photo.timestamp)
          }));
          resolve(photos);
        };
        request.onerror = () => reject(request.error);
      });
    } catch (error) {
      console.error('Error getting photos:', error);
      throw error;
    }
  }

  async getPhotoById(id: string): Promise<StoredPhoto | null> {
    try {
      const db = await this.openDB();
      const transaction = db.transaction([this.storeName], 'readonly');
      const store = transaction.objectStore(this.storeName);

      return new Promise((resolve, reject) => {
        const request = store.get(id);
        request.onsuccess = () => {
          if (request.result) {
            const photo = {
              ...request.result,
              timestamp: new Date(request.result.timestamp)
            };
            resolve(photo);
          } else {
            resolve(null);
          }
        };
        request.onerror = () => reject(request.error);
      });
    } catch (error) {
      console.error('Error getting photo:', error);
      throw error;
    }
  }

  async deletePhoto(id: string): Promise<void> {
    try {
      const db = await this.openDB();
      const transaction = db.transaction([this.storeName], 'readwrite');
      const store = transaction.objectStore(this.storeName);

      return new Promise((resolve, reject) => {
        const request = store.delete(id);
        request.onsuccess = () => resolve();
        request.onerror = () => reject(request.error);
      });
    } catch (error) {
      console.error('Error deleting photo:', error);
      throw error;
    }
  }

  async updatePhotoAnalysis(id: string, analysisResults: StoredPhoto['analysisResults']): Promise<void> {
    try {
      const photo = await this.getPhotoById(id);
      if (!photo) {
        throw new Error(`Photo with id ${id} not found`);
      }

      const updatedPhoto = {
        ...photo,
        analysisResults
      };

      await this.savePhoto(updatedPhoto);
    } catch (error) {
      console.error('Error updating photo analysis:', error);
      throw error;
    }
  }

  async clearAllPhotos(): Promise<void> {
    try {
      const db = await this.openDB();
      const transaction = db.transaction([this.storeName], 'readwrite');
      const store = transaction.objectStore(this.storeName);

      return new Promise((resolve, reject) => {
        const request = store.clear();
        request.onsuccess = () => resolve();
        request.onerror = () => reject(request.error);
      });
    } catch (error) {
      console.error('Error clearing photos:', error);
      throw error;
    }
  }

  async getPhotosCount(): Promise<number> {
    try {
      const photos = await this.getAllPhotos();
      return photos.length;
    } catch (error) {
      console.error('Error getting photos count:', error);
      return 0;
    }
  }

  // Export photos as JSON
  async exportPhotos(): Promise<string> {
    try {
      const photos = await this.getAllPhotos();
      return JSON.stringify(photos, null, 2);
    } catch (error) {
      console.error('Error exporting photos:', error);
      throw error;
    }
  }

  // Import photos from JSON
  async importPhotos(jsonData: string): Promise<void> {
    try {
      const photos: StoredPhoto[] = JSON.parse(jsonData);

      for (const photo of photos) {
        // Validate photo structure
        if (!photo.id || !photo.dataUrl || !photo.timestamp) {
          console.warn('Skipping invalid photo:', photo);
          continue;
        }

        // Convert timestamp string back to Date if needed
        if (typeof photo.timestamp === 'string') {
          photo.timestamp = new Date(photo.timestamp);
        }

        await this.savePhoto(photo);
      }
    } catch (error) {
      console.error('Error importing photos:', error);
      throw error;
    }
  }
}

// Create a singleton instance
export const photoStorage = new PhotoStorage();
