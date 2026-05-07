import { useLocation, Link, useNavigate } from 'react-router-dom'
import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { 
  DocumentTextIcon,
  ExclamationTriangleIcon,
  CheckCircleIcon,
  SparklesIcon,
  EyeIcon,
  ArrowLeftIcon,
  ClockIcon,
  BeakerIcon,
  MagnifyingGlassIcon,
  XMarkIcon,
  MapPinIcon,
  ChevronDownIcon,
  ChevronUpIcon,
  CubeTransparentIcon,
  SignalIcon,
  ShieldExclamationIcon,
  CpuChipIcon,
  CheckBadgeIcon,
} from '@heroicons/react/24/outline'

// Use the Vite proxy for MedGemma images (/medgemma-api → port 8000)
const API_BASE = '/medgemma-api'

// Resolve image URL — handles both base64 (data:) and API paths (/medgemma/...)
const resolveImageUrl = (url) => {
  if (!url) return null
  if (url.startsWith('data:')) return url // base64 passthrough
  return `${API_BASE}${url}` // prepend API base for relative paths
}

export default function MedGemmaResults() {
  const location = useLocation()
  const navigate = useNavigate()
  const result = location.state?.result
  const [selectedSlice, setSelectedSlice] = useState(null)
  const [expandedLocations, setExpandedLocations] = useState({})

  // Toggle a location's expanded state
  const toggleLocation = (loc) => {
    setExpandedLocations(prev => ({
      ...prev,
      [loc]: !prev[loc]
    }))
  }

  // Expand all locations
  const expandAll = () => {
    if (!result?.findings_by_location) return
    const all = {}
    Object.keys(result.findings_by_location).forEach(loc => { all[loc] = true })
    setExpandedLocations(all)
  }

  // Collapse all
  const collapseAll = () => setExpandedLocations({})

  // No results - show fallback
  if (!result) {
    return (
      <div className="min-h-screen bg-slate-50 dark:bg-[#09090b] flex items-center justify-center transition-colors">
        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center"
        >
          <img src="/medgemma-icon.png" alt="MedGemma" className="w-16 h-16 object-contain rounded-xl mx-auto mb-4 shadow-sm" />
          <h2 className="text-2xl font-bold mb-2 text-slate-800 dark:text-white">No Results Available</h2>
          <p className="text-slate-500 dark:text-slate-400 mb-6">Please upload and analyze scans first.</p>
          <button onClick={() => navigate(-1)} className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg inline-flex items-center gap-2 transition-colors">
            <ArrowLeftIcon className="w-4 h-4" />
            Back to Combined Results
          </button>
        </motion.div>
      </div>
    )
  }

  const findingsByLocation = result.findings_by_location || {}
  const locationNames = Object.keys(findingsByLocation)
  const totalFindings = result.findings?.length || 0

  return (
    <div className="min-h-screen bg-slate-50 dark:bg-[#09090b] text-slate-800 dark:text-slate-200 font-sans py-12 px-4 selection:bg-blue-500/30 transition-colors">
      <div className="max-w-6xl mx-auto">
        {/* Header with Back Button */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-8"
        >
          <button 
            onClick={() => navigate(-1)}
            className="inline-flex items-center gap-2 text-slate-400 hover:text-slate-600 dark:text-slate-500 dark:hover:text-white font-medium transition-colors mb-6 group"
          >
            <ArrowLeftIcon className="w-4 h-4 group-hover:-translate-x-1 transition-transform" />
            Back to Combined Results
          </button>
          
          <div className="flex items-center gap-4">
            <div className="w-12 h-12 rounded-xl bg-blue-50 dark:bg-blue-500/10 border border-blue-100 dark:border-blue-500/20 flex items-center justify-center shadow-sm">
              <img src="/medgemma-icon.png" alt="MedGemma" className="w-8 h-8 object-contain rounded-md" />
            </div>
            <div>
              <h1 className="text-3xl font-bold text-slate-800 dark:text-white tracking-tight">
                <span className="text-blue-500 dark:text-blue-400">Analysis</span> Results
              </h1>
              <p className="text-slate-400 dark:text-slate-500 font-medium text-sm mt-1">
                AI-Powered Intracranial Aneurysm Detection Report
              </p>
            </div>
          </div>
        </motion.div>

        <div className="space-y-6">
          {/* Summary Stats */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
            className="grid grid-cols-2 md:grid-cols-5 gap-4"
          >
            <div className="bg-white dark:bg-[#111113] border border-slate-100 dark:border-white/5 shadow-sm dark:shadow-none rounded-xl p-5 text-center transition-colors">
              <div className="w-10 h-10 rounded-lg bg-blue-50 dark:bg-blue-500/10 flex items-center justify-center mx-auto mb-3">
                <BeakerIcon className="w-5 h-5 text-blue-400 dark:text-blue-400" />
              </div>
              <p className="text-2xl font-bold text-blue-400 dark:text-blue-400">{result.slices_analyzed}</p>
              <p className="text-xs text-slate-400 dark:text-slate-500 font-medium mt-1">Slices Analyzed</p>
            </div>
            
            <div className="bg-white dark:bg-[#111113] border border-slate-100 dark:border-white/5 shadow-sm dark:shadow-none rounded-xl p-5 text-center transition-colors">
              <div className={`w-10 h-10 rounded-lg flex items-center justify-center mx-auto mb-3 ${
                result.has_findings ? 'bg-rose-50 dark:bg-rose-500/10' : 'bg-emerald-50 dark:bg-emerald-500/10'
              }`}>
                <MapPinIcon className={`w-5 h-5 ${result.has_findings ? 'text-rose-400' : 'text-emerald-400'}`} />
              </div>
              <p className={`text-2xl font-bold ${result.has_findings ? 'text-red-500 dark:text-red-400' : 'text-emerald-500 dark:text-emerald-400'}`}>
                {result.num_locations || 0}
              </p>
              <p className="text-xs text-slate-400 dark:text-slate-500 font-medium mt-1">Locations</p>
            </div>

            <div className="bg-white dark:bg-[#111113] border border-slate-100 dark:border-white/5 shadow-sm dark:shadow-none rounded-xl p-5 text-center transition-colors">
              <div className={`w-10 h-10 rounded-lg flex items-center justify-center mx-auto mb-3 ${
                result.has_findings ? 'bg-rose-50 dark:bg-rose-500/10' : 'bg-emerald-50 dark:bg-emerald-500/10'
              }`}>
                {result.has_findings ? (
                  <ExclamationTriangleIcon className="w-5 h-5 text-rose-400" />
                ) : (
                  <CheckCircleIcon className="w-5 h-5 text-emerald-400" />
                )}
              </div>
              <p className={`text-2xl font-bold ${result.has_findings ? 'text-red-500 dark:text-red-400' : 'text-emerald-500 dark:text-emerald-400'}`}>
                {totalFindings}
              </p>
              <p className="text-xs text-slate-400 dark:text-slate-500 font-medium mt-1">Detected Slices</p>
            </div>
            
            <div className="bg-white dark:bg-[#111113] border border-slate-100 dark:border-white/5 shadow-sm dark:shadow-none rounded-xl p-5 text-center transition-colors">
              <div className="w-10 h-10 rounded-lg bg-blue-50 dark:bg-blue-500/10 flex items-center justify-center mx-auto mb-3">
                <ClockIcon className="w-5 h-5 text-blue-400 dark:text-blue-400" />
              </div>
              <p className="text-2xl font-bold text-blue-400 dark:text-blue-400">{result.processing_time?.toFixed(1)}s</p>
              <p className="text-xs text-slate-400 dark:text-slate-500 font-medium mt-1">Processing Time</p>
            </div>

            <div className="bg-white dark:bg-[#111113] border border-slate-100 dark:border-white/5 shadow-sm dark:shadow-none rounded-xl p-5 text-center transition-colors">
              <div className={`w-10 h-10 rounded-lg flex items-center justify-center mx-auto mb-3 ${
                result.has_findings ? 'bg-rose-50 dark:bg-rose-500/10' : 'bg-emerald-50 dark:bg-emerald-500/10'
              }`}>
                <MagnifyingGlassIcon className={`w-5 h-5 ${result.has_findings ? 'text-rose-400' : 'text-emerald-400'}`} />
              </div>
              <p className={`text-2xl font-bold ${result.has_findings ? 'text-red-500 dark:text-red-400' : 'text-emerald-500 dark:text-emerald-400'}`}>
                {result.has_findings ? 'Detected' : 'Clear'}
              </p>
              <p className="text-xs text-slate-400 dark:text-slate-500 font-medium mt-1">Status</p>
            </div>
          </motion.div>

          {/* Status Badge */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.15 }}
            className={`p-4 border-l-4 shadow-sm dark:shadow-none transition-colors ${
              result.has_findings 
                ? 'border-red-500 bg-red-50 dark:bg-red-500/10 dark:border-red-500/50'
                : 'border-emerald-500 bg-emerald-50 dark:bg-emerald-500/10 dark:border-emerald-500/50'
            }`}
          >
            <div className="flex items-center gap-3">
              {result.has_findings ? (
                <ExclamationTriangleIcon className="w-6 h-6 text-red-500 dark:text-red-400 flex-shrink-0" />
              ) : (
                <CheckCircleIcon className="w-6 h-6 text-emerald-500 dark:text-emerald-400 flex-shrink-0" />
              )}
              <div>
                <p className={`font-bold text-[15px] ${result.has_findings ? 'text-red-600 dark:text-red-400' : 'text-emerald-700 dark:text-emerald-400'}`}>
                  {result.has_findings ? 'Potential Findings Detected' : 'No Obvious Abnormalities'}
                </p>
                <p className={`text-[13px] font-medium mt-0.5 ${result.has_findings ? 'text-slate-500 dark:text-red-400/80' : 'text-slate-500 dark:text-emerald-400/80'}`}>
                  {result.has_findings 
                    ? `${totalFindings} slice(s) across ${locationNames.length} anatomical location(s). Review each location below.`
                    : 'The AI did not detect any obvious abnormalities in the uploaded scans.'}
                </p>
              </div>
            </div>
          </motion.div>

          {/* AI Report */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
            className="bg-white dark:bg-[#111113] border border-slate-100 dark:border-white/5 shadow-sm dark:shadow-none rounded-xl p-6 transition-colors"
          >
            <div className="flex items-center gap-3 mb-4">
              <div className="w-8 h-8 rounded-lg bg-blue-50 dark:bg-blue-500/10 flex items-center justify-center">
                <DocumentTextIcon className="w-5 h-5 text-blue-400 dark:text-blue-400" />
              </div>
              <h2 className="text-lg font-bold text-slate-800 dark:text-white">AI Analysis Report</h2>
            </div>
            
            <div className="bg-slate-500 dark:bg-slate-800/50 dark:border dark:border-white/10 rounded-xl p-6 shadow-inner dark:shadow-none space-y-4 text-white dark:text-slate-200 transition-colors">
              {/* Report Title */}
              <div className="flex items-center gap-2 pb-3 border-b border-slate-400/30 dark:border-white/10">
                <ShieldExclamationIcon className="w-5 h-5 text-blue-200 dark:text-blue-400" />
                <h3 className="font-semibold text-slate-50 dark:text-white">CT Scan Analysis Report</h3>
              </div>
              
              {/* Stats Row */}
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                <div className="flex items-center gap-2.5 bg-slate-700 dark:bg-black/40 rounded-lg px-4 py-3">
                  <BeakerIcon className="w-4 h-4 text-blue-300 dark:text-blue-400 flex-shrink-0" />
                  <span className="text-[13px] text-slate-300 dark:text-slate-400">Analyzed: <span className="text-white font-bold">{result.slices_analyzed} slice(s)</span></span>
                </div>
                {result.has_findings && (
                  <div className="flex items-center gap-2.5 bg-slate-700 dark:bg-black/40 rounded-lg px-4 py-3">
                    <MagnifyingGlassIcon className="w-4 h-4 text-red-400 dark:text-red-400 flex-shrink-0" />
                    <span className="text-[13px] text-slate-300 dark:text-slate-400">Findings: <span className="text-red-400 font-bold">{totalFindings} slice(s)</span></span>
                  </div>
                )}
                {result.has_findings && (
                  <div className="flex items-center gap-2.5 bg-slate-700 dark:bg-black/40 rounded-lg px-4 py-3">
                    <MapPinIcon className="w-4 h-4 text-amber-300 dark:text-amber-400 flex-shrink-0" />
                    <span className="text-[13px] text-slate-300 dark:text-slate-400">Locations: <span className="text-amber-300 dark:text-amber-400 font-bold">{result.num_locations || 0}</span></span>
                  </div>
                )}
              </div>

              {/* Location Breakdown */}
              {result.has_findings && locationNames.length > 0 && (
                <div className="space-y-1.5 pt-2">
                  <p className="text-[11px] text-slate-300 dark:text-slate-500 font-bold uppercase tracking-widest">Locations with findings</p>
                  {locationNames.map(loc => (
                    <div key={loc} className="flex items-center gap-2 text-[13px] mt-1">
                      <div className="w-1.5 h-1.5 rounded-full bg-red-400 dark:bg-red-500 flex-shrink-0" />
                      <span className="text-slate-300 dark:text-slate-400 font-medium">{loc}:</span>
                      <span className="text-white font-bold">{findingsByLocation[loc].length} slice(s)</span>
                    </div>
                  ))}
                </div>
              )}

              {/* Method & Warning */}
              <div className="flex items-center gap-2.5 pt-4 border-t border-slate-400/30 dark:border-white/10">
                <CpuChipIcon className="w-4 h-4 text-slate-300 dark:text-slate-500 flex-shrink-0" />
                <span className="text-[11px] font-medium text-slate-300 dark:text-slate-400 uppercase tracking-wide">Detection Method: Semantic Visual Parsing</span>
              </div>
              {result.has_findings && (
                <div className="flex items-start gap-2.5 bg-slate-800/50 dark:bg-amber-500/10 dark:border dark:border-amber-500/20 rounded-lg px-4 py-3 mt-3">
                  <ExclamationTriangleIcon className="w-4 h-4 text-amber-400 flex-shrink-0 mt-0.5" />
                  <span className="text-[12px] font-medium text-amber-200/90 dark:text-amber-400/90 leading-tight">This is automated detection. All findings require verification by a qualified radiologist.</span>
                </div>
              )}
            </div>
          </motion.div>

          {/* ═══════════════════════════════════════════════════ */}
          {/* FINDINGS GROUPED BY LOCATION */}
          {/* ═══════════════════════════════════════════════════ */}
          {locationNames.length > 0 && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.25 }}
              className="space-y-4 pt-4"
            >
              {/* Section Header */}
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <div className="w-8 h-8 rounded-lg bg-red-50 dark:bg-red-500/10 flex items-center justify-center">
                    <MapPinIcon className="w-5 h-5 text-red-500 dark:text-red-400" />
                  </div>
                  <h2 className="text-lg font-bold text-slate-800 dark:text-white">Findings by Location</h2>
                  <span className="bg-red-50 border border-red-100 text-red-600 dark:bg-red-500/10 dark:border-red-500/20 dark:text-red-400 text-[11px] font-bold px-2.5 py-1 rounded-full uppercase tracking-wider">
                    {locationNames.length} location(s)
                  </span>
                </div>
                <div className="flex gap-2">
                  <button 
                    onClick={expandAll}
                    className="text-[11px] font-bold uppercase tracking-wider text-blue-600 hover:text-blue-700 dark:text-blue-400 dark:hover:text-blue-300 transition-colors px-3 py-1.5 rounded-md bg-blue-50 hover:bg-blue-100 dark:bg-blue-500/10 dark:hover:bg-blue-500/20"
                  >
                    Expand All
                  </button>
                  <button 
                    onClick={collapseAll}
                    className="text-[11px] font-bold uppercase tracking-wider text-slate-500 hover:text-slate-700 dark:text-slate-400 dark:hover:text-slate-300 transition-colors px-3 py-1.5 rounded-md bg-slate-100 hover:bg-slate-200 dark:bg-white/5 dark:hover:bg-white/10"
                  >
                    Collapse All
                  </button>
                </div>
              </div>

              {/* Location Cards */}
              {locationNames.map((locationName, locIdx) => {
                const slices = findingsByLocation[locationName]
                const isExpanded = expandedLocations[locationName] !== false // default expanded
                
                return (
                  <motion.div
                    key={locationName}
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.3 + locIdx * 0.05 }}
                    className="bg-white dark:bg-[#111113] border border-slate-200 dark:border-white/10 shadow-sm dark:shadow-none rounded-xl overflow-hidden transition-colors"
                  >
                    {/* Location Header - Clickable */}
                    <button
                      onClick={() => toggleLocation(locationName)}
                      className="w-full flex items-center justify-between p-5 hover:bg-slate-50 dark:hover:bg-white/5 transition-colors text-left border-b border-slate-100 dark:border-white/5 last:border-0"
                    >
                      <div className="flex items-center gap-3">
                        <div className="w-10 h-10 rounded-lg bg-red-50 dark:bg-red-500/10 border border-red-100 dark:border-red-500/20 flex items-center justify-center">
                          <MapPinIcon className="w-5 h-5 text-red-500 dark:text-red-400" />
                        </div>
                        <div>
                          <h3 className="font-bold text-slate-800 dark:text-white text-[15px]">{locationName}</h3>
                          <p className="text-slate-500 dark:text-slate-400 font-medium text-[13px] mt-0.5">
                            {slices.length} slice{slices.length !== 1 ? 's' : ''} detected
                            {' — '}
                            Slices: {slices.map(s => `#${s.slice_number}`).join(', ')}
                          </p>
                        </div>
                      </div>
                      <div className="flex items-center gap-3">
                        <span className="bg-red-50 border border-red-100 text-red-600 dark:bg-red-500/10 dark:border-red-500/20 dark:text-red-400 text-xs font-bold px-2.5 py-1 rounded-full">
                          {slices.length}
                        </span>
                        {isExpanded ? (
                          <ChevronUpIcon className="w-5 h-5 text-slate-400 dark:text-slate-500" />
                        ) : (
                          <ChevronDownIcon className="w-5 h-5 text-slate-400 dark:text-slate-500" />
                        )}
                      </div>
                    </button>

                    {/* Expanded: Slice Gallery */}
                    <AnimatePresence>
                      {isExpanded && (
                        <motion.div
                          initial={{ height: 0, opacity: 0 }}
                          animate={{ height: 'auto', opacity: 1 }}
                          exit={{ height: 0, opacity: 0 }}
                          transition={{ duration: 0.2 }}
                          className="overflow-hidden bg-slate-50 dark:bg-black/20 border-t border-slate-100 dark:border-white/5"
                        >
                          <div className="px-5 py-5">
                            {/* Slice Grid */}
                            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-4">
                              {slices.map((slice, sliceIdx) => (
                                <motion.div
                                  key={slice.slice_index}
                                  initial={{ opacity: 0, scale: 0.95 }}
                                  animate={{ opacity: 1, scale: 1 }}
                                  transition={{ delay: sliceIdx * 0.03 }}
                                  className="group cursor-pointer flex flex-col"
                                  onClick={() => setSelectedSlice(slice)}
                                >
                                  {/* Thumbnail */}
                                  <div className="relative aspect-square rounded-xl overflow-hidden border border-slate-200 dark:border-white/10 bg-white dark:bg-black shadow-sm group-hover:border-blue-400 dark:group-hover:border-blue-500 group-hover:shadow-md transition-all">
                                    {slice.image ? (
                                      <img 
                                        src={resolveImageUrl(slice.image)} 
                                        alt={`Slice ${slice.slice_number}`}
                                        className="w-full h-full object-cover"
                                        loading="lazy"
                                      />
                                    ) : (
                                      <div className="w-full h-full bg-slate-100 dark:bg-white/5 flex items-center justify-center">
                                        <span className="text-slate-400 dark:text-slate-500 text-[11px] font-medium">No image</span>
                                      </div>
                                    )}
                                    {/* Hover Overlay */}
                                    <div className="absolute inset-0 bg-slate-900/40 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center backdrop-blur-[1px]">
                                      <EyeIcon className="w-8 h-8 text-white drop-shadow-md" />
                                    </div>
                                    {/* Slice Number Badge */}
                                    <div className="absolute top-2 left-2 bg-white/90 dark:bg-black/90 backdrop-blur border border-slate-200 dark:border-white/10 text-slate-800 dark:text-white text-[10px] font-bold px-2 py-1 rounded shadow-sm">
                                      #{slice.slice_number}
                                    </div>
                                    {slice.ground_truth && slice.ground_truth.length > 0 && (
                                      <div className="absolute top-2 right-2 bg-emerald-500 dark:bg-emerald-600 text-white text-[10px] font-bold px-2 py-1 rounded shadow-sm flex items-center gap-1">
                                        <CheckBadgeIcon className="w-3 h-3" /> GT
                                      </div>
                                    )}
                                  </div>
                                  {/* Slice Label */}
                                  <p className="text-[12px] font-bold text-slate-500 dark:text-slate-400 mt-2 text-center">
                                    Slice {slice.slice_number}
                                  </p>
                                </motion.div>
                              ))}
                            </div>
                          </div>
                        </motion.div>
                      )}
                    </AnimatePresence>
                  </motion.div>
                )
              })}
            </motion.div>
          )}

          {/* Image Viewer Modal */}
          <AnimatePresence>
            {selectedSlice && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="fixed inset-0 bg-slate-900/80 dark:bg-black/80 backdrop-blur-sm z-50 flex items-center justify-center p-4"
                onClick={() => setSelectedSlice(null)}
              >
                <motion.div
                  initial={{ opacity: 0, scale: 0.95, y: 10 }}
                  animate={{ opacity: 1, scale: 1, y: 0 }}
                  exit={{ opacity: 0, scale: 0.95, y: 10 }}
                  className="bg-white dark:bg-[#111113] border border-slate-200 dark:border-white/10 shadow-2xl rounded-2xl p-6 max-w-4xl w-full max-h-[90vh] overflow-y-auto"
                  onClick={(e) => e.stopPropagation()}
                >
                  <div className="flex justify-between items-start mb-6">
                    <div>
                      <h3 className="text-xl font-bold text-slate-800 dark:text-white flex items-center gap-2">
                        <EyeIcon className="w-6 h-6 text-blue-500 dark:text-blue-400" />
                        Slice {selectedSlice.slice_number} — Enlarged View
                      </h3>
                      {selectedSlice.location && (
                        <p className="text-slate-500 dark:text-slate-400 font-medium text-sm mt-1.5 flex items-center gap-1.5">
                          <MapPinIcon className="w-4 h-4" />
                          {selectedSlice.location || findingsByLocation && Object.entries(findingsByLocation).find(([_, slices]) => slices.some(s => s.slice_index === selectedSlice.slice_index))?.[0]}
                        </p>
                      )}
                    </div>
                    <button
                      onClick={() => setSelectedSlice(null)}
                      className="text-slate-400 hover:text-slate-600 dark:hover:text-white transition-colors p-1.5 rounded-lg hover:bg-slate-100 dark:hover:bg-white/10"
                    >
                      <XMarkIcon className="w-6 h-6" />
                    </button>
                  </div>
                  
                  <div className="flex justify-center mb-6 bg-slate-50 dark:bg-black/40 rounded-xl p-4 border border-slate-100 dark:border-white/5">
                    <img 
                      src={resolveImageUrl(selectedSlice.image)} 
                      alt={`Slice ${selectedSlice.slice_number}`}
                      className="max-w-full max-h-[50vh] rounded-lg shadow-sm border border-slate-200 dark:border-white/10"
                    />
                  </div>
                  
                  {/* Slice details - structured with icons */}
                  {selectedSlice && (
                    <div className="space-y-4">
                      {/* Location */}
                      {selectedSlice.location && (
                        <div className="flex items-center gap-3 bg-white dark:bg-white/5 rounded-xl px-5 py-4 border border-slate-200 dark:border-white/10 shadow-sm">
                          <MapPinIcon className="w-6 h-6 text-red-500 dark:text-red-400 flex-shrink-0" />
                          <div>
                            <p className="text-[11px] text-slate-400 dark:text-slate-500 font-bold uppercase tracking-widest">Location</p>
                            <p className="text-sm text-slate-800 dark:text-white font-bold">{selectedSlice.location}</p>
                          </div>
                        </div>
                      )}

                      {/* Bbox & Intensity Row */}
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        {selectedSlice.bbox && selectedSlice.bbox[2] > 0 && (
                          <div className="flex items-center gap-3 bg-white dark:bg-white/5 rounded-xl px-5 py-4 border border-slate-200 dark:border-white/10 shadow-sm">
                            <CubeTransparentIcon className="w-6 h-6 text-blue-500 dark:text-blue-400 flex-shrink-0" />
                            <div>
                              <p className="text-[11px] text-slate-400 dark:text-slate-500 font-bold uppercase tracking-widest">Detected Box</p>
                              <p className="text-sm text-slate-800 dark:text-white font-mono font-medium">({selectedSlice.bbox[0]}, {selectedSlice.bbox[1]}) to ({selectedSlice.bbox[2]}, {selectedSlice.bbox[3]})</p>
                            </div>
                          </div>
                        )}
                        {selectedSlice.ground_truth && selectedSlice.ground_truth.length > 0 && (
                          <div className="flex items-center gap-3 bg-emerald-50 dark:bg-emerald-500/10 rounded-xl px-5 py-4 border border-emerald-100 dark:border-emerald-500/20 shadow-sm">
                            <CheckBadgeIcon className="w-6 h-6 text-emerald-500 dark:text-emerald-400 flex-shrink-0" />
                            <div>
                              <p className="text-[11px] text-emerald-600 dark:text-emerald-500/80 font-bold uppercase tracking-widest">Ground Truth</p>
                              <p className="text-sm text-emerald-800 dark:text-emerald-400 font-mono font-medium">
                                {selectedSlice.ground_truth.map(gt => `(${gt.coordinates?.x?.toFixed(0) || 0}, ${gt.coordinates?.y?.toFixed(0) || 0})`).join(', ')}
                              </p>
                            </div>
                          </div>
                        )}
                        {selectedSlice.intensity != null && selectedSlice.intensity > 0 && (
                          <div className="flex items-center gap-3 bg-white dark:bg-white/5 rounded-xl px-5 py-4 border border-slate-200 dark:border-white/10 shadow-sm">
                            <SignalIcon className="w-6 h-6 text-amber-500 dark:text-amber-400 flex-shrink-0" />
                            <div>
                              <p className="text-[11px] text-slate-400 dark:text-slate-500 font-bold uppercase tracking-widest">Max Intensity</p>
                              <p className="text-sm text-slate-800 dark:text-white font-mono font-medium">{Number(selectedSlice.intensity).toFixed(0)}</p>
                            </div>
                          </div>
                        )}
                      </div>

                      {/* AI Analysis */}
                      {selectedSlice.response && (
                        <div className="bg-slate-50 dark:bg-blue-500/5 rounded-xl px-5 py-4 border border-slate-200 dark:border-blue-500/20 shadow-sm">
                          <div className="flex items-center gap-2 mb-2 border-b border-slate-200 dark:border-blue-500/20 pb-2">
                            <SparklesIcon className="w-5 h-5 text-blue-500 dark:text-blue-400" />
                            <p className="text-[12px] text-slate-700 dark:text-blue-200 font-bold uppercase tracking-widest">AI Analysis</p>
                          </div>
                          <p className="text-[14px] text-slate-700 dark:text-slate-300 font-medium whitespace-pre-wrap leading-relaxed">
                            {selectedSlice.response}
                          </p>
                        </div>
                      )}

                      {/* Warning */}
                      <div className="flex items-start gap-3 bg-amber-50 dark:bg-amber-500/10 border border-amber-200 dark:border-amber-500/20 rounded-xl px-5 py-4 shadow-sm">
                        <ExclamationTriangleIcon className="w-5 h-5 text-amber-500 dark:text-amber-400 flex-shrink-0 mt-0.5" />
                        <span className="text-[13px] font-bold text-amber-800 dark:text-amber-400">Automated detection — requires radiologist review.</span>
                      </div>
                    </div>
                  )}
                </motion.div>
              </motion.div>
            )}
          </AnimatePresence>

          {/* Actions Bar */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.35 }}
            className="flex items-center justify-center mt-8"
          >
            <button 
              onClick={() => navigate(-1)}
              className="px-6 py-2.5 bg-white dark:bg-white/5 border border-slate-200 dark:border-white/10 hover:bg-slate-50 dark:hover:bg-white/10 text-slate-700 dark:text-slate-300 font-bold rounded-xl shadow-sm inline-flex items-center gap-2 transition-colors"
            >
              <ArrowLeftIcon className="w-4 h-4" />
              Back to Summary
            </button>
          </motion.div>

          {/* Disclaimer */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.4 }}
            className="p-4 mt-8"
          >
            <p className="text-[11px] font-medium text-slate-400 dark:text-slate-500 text-center max-w-2xl mx-auto">
              <strong>Disclaimer:</strong> This AI analysis is for research and educational purposes only. 
              It is not intended to replace professional medical diagnosis. 
              Always consult with qualified healthcare professionals for medical decisions.
            </p>
          </motion.div>
        </div>
      </div>
    </div>
  )
}
