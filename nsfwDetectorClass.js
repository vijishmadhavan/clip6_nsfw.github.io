class NsfwDetector {
    constructor() {
        this._threshold = 0.35;
        this._nsfwLabels = [
            'FEMALE_BREAST_EXPOSED', 'FEMALE_GENITALIA_EXPOSED', 'BUTTOCKS_EXPOSED', 'ANUS_EXPOSED',
            'MALE_GENITALIA_EXPOSED', 'BLOOD_SHED', 'VIOLENCE', 'GORE', 'PORNOGRAPHY', 'DRUGS', 'ALCOHOL',
        ];
        // Load the TensorFlow model once and reuse
        this._classifierPromise = window.tensorflowPipeline('zero-shot-image-classification', 'Xenova/clip-vit-base-patch32');
    }

    async isNsfw(imageUrl) {
        let blobUrl = '';
        try {
            blobUrl = await this._loadAndResizeImage(imageUrl);
            const classifier = await this._classifierPromise; // Use the preloaded model
            const output = await classifier(blobUrl, this._nsfwLabels);
            console.log(output);
            const nsfwDetected = output.some(result => result.score > this._threshold);
            return nsfwDetected;
        } catch (error) {
            console.error('Error during NSFW classification: ', error);
            throw error;
        } finally {
            if (blobUrl) {
                URL.revokeObjectURL(blobUrl); // Free up memory
            }
        }
    }

    async _loadAndResizeImage(imageUrl) {
        const img = await this._loadImage(imageUrl);
        const offScreenCanvas = document.createElement('canvas');
        const ctx = offScreenCanvas.getContext('2d');

        // Set the canvas size to the target resolution
        offScreenCanvas.width = 128;
        offScreenCanvas.height = 128;

        // Draw the image onto the canvas at the new size
        ctx.drawImage(img, 0, 0, offScreenCanvas.width, offScreenCanvas.height);

        // Convert the canvas to a Blob and create a Blob URL
        return new Promise((resolve, reject) => {
            offScreenCanvas.toBlob(blob => {
                if (!blob) {
                    reject('Canvas to Blob conversion failed');
                    return;
                }
                const blobUrl = URL.createObjectURL(blob);
                resolve(blobUrl);
            }, 'image/jpeg');
        });
    }

    async _loadImage(url) {
        return new Promise((resolve, reject) => {
            const img = new Image();
            img.crossOrigin = 'anonymous'; // This is important for loading images from external URLs
            img.onload = () => resolve(img);
            img.onerror = () => reject(`Failed to load image: ${url}`);
            img.src = url;
        });
    }
}

window.NsfwDetector = NsfwDetector;
