import React, { useState, useCallback } from 'react'
import { useNavigate } from 'react-router-dom'
import { motion } from 'framer-motion'
import { useDropzone } from 'react-dropzone'
import { 
  CloudArrowUpIcon, 
  DocumentIcon, 
  XMarkIcon,
  BeakerIcon,
  ExclamationTriangleIcon,
  SparklesIcon,
  CpuChipIcon,
} from '@heroicons/react/24/outline'
import toast from 'react-hot-toast'
import { analyzeDicom, getDemoPrediction, analyzeMedGemma } from '../api/client'



// --- Sci-Fi Log Terminal Component ---
const LogTerminal = () => {
  const [logs, setLogs] = useState([])
  const bottomRef = React.useRef(null)

  React.useEffect(() => {
    // Open WebSocket
    const ws = new WebSocket('ws://localhost:8001/ws/logs')
    
    ws.onmessage = (event) => {
      setLogs(prev => [...prev, event.data])
    }

    ws.onerror = (e) => {
      console.error("WebSocket error:", e)
      setLogs(prev => [...prev, "⚠️ Connection Warning: Real-time logs unavailable."])
    }

    return () => {
      ws.close()
    }
  }, [])

  React.useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [logs])

  return (
    <div className="space-y-1">
      {logs.map((log, i) => (
        <motion.div 
          key={i}
          initial={{ opacity: 0, x: -10 }}
          animate={{ opacity: 1, x: 0 }}
          className="text-primary-600/80 hover:text-primary-800 dark:text-primary-300/80 dark:hover:text-primary-300"
        >
          <span className="text-surface-400 dark:text-surface-600 mr-2">[{new Date().toLocaleTimeString()}]</span>
          <span>{">"} {log}</span>
        </motion.div>
      ))}
      <div ref={bottomRef} />
    </div>
  )
}

// Pipeline Step indicator
const PipelineStep = ({ icon: Icon, label, status }) => {
  const colors = {
    pending:  'text-slate-400 dark:text-slate-500',
    running:  'text-primary-600 dark:text-primary-400',
    done:     'text-emerald-600 dark:text-emerald-400',
    error:    'text-red-500',
  }
  return (
    <div className={`flex items-center gap-2 text-sm font-mono ${colors[status]}`}>
      {status === 'running' ? (
        <div className="w-4 h-4 border-2 border-current border-t-transparent rounded-full animate-spin flex-shrink-0" />
      ) : (
        <Icon className="w-4 h-4 flex-shrink-0" />
      )}
      <span>{label}</span>
      {status === 'done'  && <span className="ml-auto text-xs">✓ Done</span>}
      {status === 'error' && <span className="ml-auto text-xs">✗ Failed</span>}
    </div>
  )
}

export default function Analysis() {
  const navigate = useNavigate()
  const [files, setFiles] = useState([])
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [progress, setProgress] = useState(0)
  const [error, setError] = useState(null)

  // Pipeline step statuses
  const [nnunetStatus, setNnunetStatus] = useState('pending')   // pending | running | done | error
  const [gemmaStatus,  setGemmaStatus]  = useState('pending')

  const onDrop = useCallback((acceptedFiles) => {
    // Accept DICOM, NIfTI, and no-extension files
    const validFiles = acceptedFiles.filter(file => 
      file.name.endsWith('.dcm') || 
      file.name.endsWith('.dicom') ||
      file.name.endsWith('.nii') ||
      file.name.endsWith('.nii.gz') ||
      !file.name.includes('.')
    )
    
    if (validFiles.length === 0) {
      toast.error('Please upload DICOM (.dcm) or NIfTI (.nii / .nii.gz) files')
      return
    }
    
    setFiles(prev => [...prev, ...validFiles])
    setError(null)
  }, [])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/dicom': ['.dcm', '.dicom'],
      'application/octet-stream': ['.nii', '.nii.gz'],
    },
    multiple: true,
  })

  const removeFile = (index) => {
    setFiles(prev => prev.filter((_, i) => i !== index))
  }

  const handleAnalyze = async () => {
    if (files.length === 0) {
      toast.error('Please upload at least one scan file')
      return
    }

    setIsAnalyzing(true)
    setProgress(0)
    setError(null)
    setNnunetStatus('pending')
    setGemmaStatus('pending')

    let nnunetResult = null
    let gemmaResult  = null
    let nnunetError  = null
    let gemmaError   = null

    try {
      // ── Run nnU-Net ──────────────────────────────────────────────────────────
      setNnunetStatus('running')
      try {
        nnunetResult = await analyzeDicom(files, (p) => setProgress(Math.round(p * 0.5)))
        setNnunetStatus('done')
        toast.success('nnU-Net analysis complete!')
      } catch (err) {
        nnunetError = err.response?.data?.detail || err.message || 'nnU-Net analysis failed'
        setNnunetStatus('error')
        toast.error('nnU-Net: ' + nnunetError)
        console.error('nnU-Net error:', err)
      }

      // ── Run MedGemma ─────────────────────────────────────────────────────────
      setGemmaStatus('running')
      try {
        gemmaResult = await analyzeMedGemma(files, (p) => setProgress(50 + Math.round(p * 0.5)))
        setGemmaStatus('done')
        toast.success('MedGemma analysis complete!')
      } catch (err) {
        gemmaError = err.response?.data?.detail || err.message || 'MedGemma analysis failed'
        setGemmaStatus('error')
        toast.error('MedGemma: ' + gemmaError)
        console.error('MedGemma error:', err)
      }

      // At least one must succeed to navigate
      if (!nnunetResult && !gemmaResult) {
        setError('Both analyses failed. Please check if both backends are running.')
        return
      }

      setProgress(100)
      navigate('/results/combined', {
        state: {
          nnunetResult,
          gemmaResult,
          nnunetError,
          gemmaError,
        }
      })
    } finally {
      setIsAnalyzing(false)
    }
  }

  const handleDemo = async () => {
    setIsAnalyzing(true)
    setProgress(50)
    
    try {
      const result = await getDemoPrediction()
      setProgress(100)
      toast.success('Demo analysis complete!')
      navigate(`/results/${result.id}`, { state: { result } })
    } catch (err) {
      console.error('Demo error:', err)
      toast.error('Demo failed. Is the backend running?')
    } finally {
      setIsAnalyzing(false)
    }
  }

  return (
    <div className="min-h-screen py-12 px-4 bg-slate-50 text-slate-900 dark:bg-slate-950 dark:text-white">
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center mb-12"
        >
          <h1 className="text-4xl font-bold mb-3">
            Upload <span className="text-primary-600 dark:text-primary-400">Brain Scans</span>
          </h1>
          <p className="text-slate-500 dark:text-slate-400 text-sm">
            Scans are analyzed by <strong>both</strong> our nnU-Net model and Google MedGemma — results shown side-by-side.
          </p>
          
          {/* Pipeline badge */}
          <div className="flex items-center justify-center gap-3 mt-5">
            <div className="flex items-center gap-1.5 bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-700/40 rounded-full px-3 py-1 text-xs font-medium text-blue-700 dark:text-blue-300">
              <CpuChipIcon className="w-3.5 h-3.5" />
              nnU-Net Model
            </div>
            <span className="text-slate-400 dark:text-slate-600 text-xs">+</span>
            <div className="flex items-center gap-1.5 bg-violet-50 dark:bg-violet-900/20 border border-violet-200 dark:border-violet-700/40 rounded-full px-3 py-1 text-xs font-medium text-violet-700 dark:text-violet-300">
              <img src="/medgemma-icon.png" alt="MedGemma" className="w-4 h-4 object-contain rounded-sm" />
              Google MedGemma
            </div>
          </div>
        </motion.div>

        <div className="space-y-6">
          {/* Upload Area */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
          >
            <div
              {...getRootProps()}
              className={`card p-12 border-2 border-dashed transition-all cursor-pointer ${
                isDragActive
                  ? 'border-primary-500 bg-primary-500/10'
                  : 'border-surface-300 hover:border-surface-400 dark:border-surface-700 dark:hover:border-surface-600'
              }`}
            >
              <input {...getInputProps()} />
              <div className="text-center">
                <CloudArrowUpIcon className={`w-16 h-16 mx-auto mb-4 ${
                  isDragActive ? 'text-primary-600 dark:text-primary-400' : 'text-surface-400 dark:text-surface-500'
                }`} />
                <p className="text-lg font-medium mb-2">
                  {isDragActive ? 'Drop files here...' : 'Drag & drop scans here'}
                </p>
                <p className="text-surface-500 dark:text-surface-500 text-sm">
                  or click to browse — supports <code className="bg-slate-100 dark:bg-slate-800 px-1 rounded text-xs">.dcm</code>, <code className="bg-slate-100 dark:bg-slate-800 px-1 rounded text-xs">.nii</code>, <code className="bg-slate-100 dark:bg-slate-800 px-1 rounded text-xs">.nii.gz</code>
                </p>
              </div>
            </div>
          </motion.div>

          {/* File List */}
          {files.length > 0 && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              className="card p-6"
            >
              <div className="flex items-center justify-between mb-4">
                <h3 className="font-medium">Uploaded Files ({files.length})</h3>
                <button
                  onClick={() => setFiles([])}
                  className="text-sm text-surface-500 hover:text-surface-900 dark:text-surface-400 dark:hover:text-white"
                >
                  Clear All
                </button>
              </div>
              <div className="space-y-2 max-h-60 overflow-y-auto">
                {files.map((file, index) => (
                  <motion.div
                    key={index}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: index * 0.05 }}
                    className="flex items-center justify-between p-3 bg-surface-100 dark:bg-surface-800/50 rounded-lg"
                  >
                    <div className="flex items-center gap-3">
                      <DocumentIcon className="w-5 h-5 text-primary-600 dark:text-primary-400" />
                      <span className="text-sm truncate max-w-xs">{file.name}</span>
                      <span className="text-xs text-surface-500 dark:text-surface-500">
                        {(file.size / 1024).toFixed(1)} KB
                      </span>
                    </div>
                    <button
                      onClick={() => removeFile(index)}
                      className="p-1 hover:bg-surface-200 dark:hover:bg-surface-700 rounded"
                    >
                      <XMarkIcon className="w-4 h-4 text-surface-500 dark:text-surface-400" />
                    </button>
                  </motion.div>
                ))}
              </div>
            </motion.div>
          )}

          {/* Error Display */}
          {error && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="flex items-center gap-3 p-4 bg-risk-high/10 border border-risk-high/30 rounded-xl text-risk-high"
            >
              <ExclamationTriangleIcon className="w-5 h-5 flex-shrink-0" />
              <span>{error}</span>
            </motion.div>
          )}

          {/* Analysis status panel */}
          {isAnalyzing && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="card p-6 bg-slate-100 border-primary-500/30 dark:bg-black dark:border-primary-500/30"
            >
              <div className="flex items-center gap-4 mb-4 border-b border-primary-500/20 pb-3">
                <BeakerIcon className="w-5 h-5 text-primary-600 dark:text-primary-400 animate-pulse" />
                <span className="font-mono text-primary-600 dark:text-primary-400 text-sm uppercase tracking-wider">Dual Pipeline Running</span>
                <span className="ml-auto text-xs text-surface-500 font-mono animate-pulse">{progress}%</span>
              </div>

              {/* Pipeline steps */}
              <div className="space-y-2.5 mb-4">
                <PipelineStep icon={CpuChipIcon}   label="nnU-Net Model — Segmentation"      status={nnunetStatus} />
                <PipelineStep icon={SparklesIcon}  label="MedGemma — LLM Analysis"           status={gemmaStatus} />
              </div>

              {/* Progress bar */}
              <div className="w-full bg-slate-200 dark:bg-surface-800 rounded-full h-1.5 mb-4">
                <motion.div
                  className="bg-primary-500 h-1.5 rounded-full"
                  animate={{ width: `${progress}%` }}
                  transition={{ duration: 0.3 }}
                />
              </div>

              {/* Sci-Fi Log Terminal */}
              <div className="h-48 overflow-y-auto font-mono text-xs p-2 bg-slate-200/50 rounded border border-surface-200 dark:bg-black/50 dark:border-surface-800 shadow-inner">
                 <LogTerminal />
              </div>
            </motion.div>
          )}

          {/* Action Buttons */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 }}
            className="flex flex-col sm:flex-row gap-4"
          >
            <motion.button
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              onClick={handleAnalyze}
              disabled={isAnalyzing || files.length === 0}
              className="flex-1 btn-primary py-4 text-lg disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isAnalyzing ? 'Analyzing with both models...' : 'Analyze Files (Both Models)'}
            </motion.button>
           
          </motion.div>

            
        </div>
      </div>
    </div>
  )
}
