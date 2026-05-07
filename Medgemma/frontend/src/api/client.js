/**
 * API client — MedGemma Aneurysm Detection Backend
 */

import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const api = axios.create({
    baseURL: API_BASE_URL,
    timeout: 300000, // 5 min — MedGemma inference can be slow
    headers: { 'Content-Type': 'application/json' },
});

api.interceptors.response.use(
    response => response,
    error => {
        console.error('API Error:', error.message);
        return Promise.reject(error);
    }
);

// ── Health check ──────────────────────────────────────────────────────────────
export const checkHealth = async () => {
    const response = await api.get('/health');
    return response.data;
};

// ── Upload DICOM / NIfTI slices and run MedGemma analysis ────────────────────
export const analyzeMedGemmaUpload = async (files, onProgress) => {
    const formData = new FormData();
    files.forEach(file => formData.append('files', file));

    const response = await api.post('/medgemma/analyze-upload', formData, {
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

// ── Get the URL for a specific annotated finding image ────────────────────────
export const getFindingImageUrl = (analysisId, sliceIndex) =>
    `${API_BASE_URL}/medgemma/finding-image/${analysisId}/${sliceIndex}`;

export default api;
