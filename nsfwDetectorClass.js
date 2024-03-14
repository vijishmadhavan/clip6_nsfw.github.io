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
            // Initial check with lower resolution
            blobUrl = await this._loadAndResizeImage(imageUrl, 124);
            let classifier = await this._classifierPromise;
            let output = await classifier(blobUrl, this._nsfwLabels);
            console.log('Initial classification (124x124):', output); // Log initial results
    
            let nsfwDetected = output.some(result => result.score > this._threshold);
            
            // If NSFW is detected, recheck with higher resolution
            if (nsfwDetected) {
                URL.revokeObjectURL(blobUrl); // Free up memory from the first blob
                blobUrl = await this._loadAndResizeImage(imageUrl, 224);
                output = await classifier(blobUrl, this._nsfwLabels);
                console.log('Re-evaluation classification (224x224):', output); // Log re-evaluation results
                nsfwDetected = output.some(result => result.score > this._threshold);
            }
            
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
    


