/**
 * API client for communicating with the FastAPI backend.
 */

import axios from 'axios';

// Use Vite Proxy to avoid CORS/Network issues
const API_BASE_URL = '/api';

// Create axios instance
const api = axios.create({
    baseURL: API_BASE_URL,
    timeout: 600000, // 10 minutes for slow CPU inference
    headers: {
        'Content-Type': 'application/json',
    },
});

// MedGemma API instance (proxied through /medgemma-api → port 8000)
const medgemmaApi = axios.create({
    baseURL: '/medgemma-api',
    timeout: 600000,
    headers: { 'Content-Type': 'application/json' },
});

// Health check
export const checkHealth = async () => {
    const response = await api.get('/health');
    return response.data;
};

// Analyze DICOM files
export const analyzeDicom = async (files, onProgress) => {
    const formData = new FormData();

    files.forEach((file) => {
        formData.append('files', file);
    });

    const response = await api.post('/analyze', formData, {
        headers: {
            'Content-Type': 'multipart/form-data',
        },
        onUploadProgress: (progressEvent) => {
            if (onProgress) {
                const percentCompleted = Math.round(
                    (progressEvent.loaded * 100) / progressEvent.total
                );
                onProgress(percentCompleted);
            }
        },
    });

    return response.data;
};

// Get analysis by ID
export const getAnalysis = async (analysisId) => {
    const response = await api.get(`/analysis/${analysisId}`);
    return response.data;
};

// Get anatomical locations list
export const getLocations = async () => {
    const response = await api.get('/locations');
    return response.data;
};

// Demo prediction (for testing without model)
export const getDemoPrediction = async () => {
    const response = await api.post('/demo/predict');
    return response.data;
};

// ── MedGemma: Upload & Analyze ────────────────────────────────────────────────
export const analyzeMedGemma = async (files, onProgress) => {
    const formData = new FormData();
    files.forEach(file => formData.append('files', file));

    const response = await medgemmaApi.post('/medgemma/analyze-upload', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
        onUploadProgress: progressEvent => {
            if (onProgress) {
                const pct = Math.round((progressEvent.loaded * 100) / progressEvent.total);
                onProgress(pct);
            }
        },
    });
    return response.data;
};

// MedGemma finding image URL helper
export const getMedGemmaImageUrl = (analysisId, sliceIndex) =>
    `/medgemma-api/medgemma/finding-image/${analysisId}/${sliceIndex}`;

// Export the API instances
export { medgemmaApi };
export default api;
