import { useState, useEffect } from 'react'
import { useLocation, useParams, Link, useNavigate } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import { 
  ArrowLeftIcon,
  ArrowDownTrayIcon,
  PrinterIcon,
  ChevronLeftIcon,
  ChevronRightIcon,
  EyeIcon,
  MagnifyingGlassIcon,
} from '@heroicons/react/24/outline'

export default function Results() {
  const { id } = useParams()
  const location = useLocation()
  const navigate = useNavigate()
  const result = location.state?.result
  const [activeSliceIdx, setActiveSliceIdx] = useState(0)
  const [filenameSearch, setFilenameSearch] = useState('')

  if (!result) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <h2 className="text-2xl font-bold mb-4">No Report Data</h2>
          <Link to="/analysis" className="btn-primary">Return to Console</Link>
        </div>
      </div>
    )
  }

  const { predictions, overall_risk, confidence, processing_time } = result
  const sliceImages = result.slice_images || []
  const hasGallery = sliceImages.length > 0

  const activeImage = hasGallery ? sliceImages[activeSliceIdx] : null
  const activeBase64 = activeImage?.image_base64 || result.image_base64

  useEffect(() => {
    if (!hasGallery) return
    const handleKeyDown = (e) => {
      if (e.key === 'ArrowLeft') {
        setActiveSliceIdx(prev => Math.max(0, prev - 1))
      } else if (e.key === 'ArrowRight') {
        setActiveSliceIdx(prev => Math.min(sliceImages.length - 1, prev + 1))
      }
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [hasGallery, sliceImages.length])

  return (
    <div className="py-4 px-4 min-h-screen bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-blue-50 via-white to-slate-50 dark:from-slate-900 dark:via-[#09090b] dark:to-slate-950 text-slate-900 dark:text-white">
      
      {/* Utility Bar */}
      <div className="flex items-center justify-between mb-4 pb-3 border-b border-slate-300 dark:border-slate-800">
        <button onClick={() => navigate(-1)} className="flex items-center gap-2 text-slate-600 hover:text-slate-900 dark:text-slate-400 dark:hover:text-white transition-colors">
          <ArrowLeftIcon className="w-4 h-4" />
          <span className="text-sm font-medium">Back to Combined Results</span>
        </button>
        <div className="flex items-center gap-4">
          <div className={`flex items-center gap-2 px-3 py-1 rounded-full border shadow-sm ${
            overall_risk === 'High' ? 'bg-red-500/10 border-red-500/30 dark:bg-red-500/20' : 
            overall_risk === 'Moderate' ? 'bg-amber-500/10 border-amber-500/30 dark:bg-amber-500/20' : 
            'bg-emerald-500/10 border-emerald-500/30 dark:bg-emerald-500/20'
          }`}>
            <div className={`w-2.5 h-2.5 rounded-full ${overall_risk === 'High' ? 'bg-red-500 animate-pulse shadow-[0_0_8px_rgba(239,68,68,0.8)]' : overall_risk === 'Moderate' ? 'bg-amber-500' : 'bg-green-500'}`}></div>
            <span className={`font-mono text-sm font-bold ${
              overall_risk === 'High' ? 'text-red-700 dark:text-red-400' : 
              overall_risk === 'Moderate' ? 'text-amber-700 dark:text-amber-400' : 
              'text-emerald-700 dark:text-emerald-400'
            }`}>RISK: {overall_risk.toUpperCase()}</span>
          </div>
          <button className="btn-secondary flex items-center gap-2 text-xs">
            <PrinterIcon className="w-4 h-4" /> Print
          </button>
          <button className="btn-primary flex items-center gap-2 text-xs">
            <ArrowDownTrayIcon className="w-4 h-4" /> Export
          </button>
        </div>
      </div>

      {/* TOP SECTION: Split Grid */}
      <div className="grid grid-cols-12 gap-4 mb-4">
        
        {/* LEFT: Image Viewer + Thumbnails + Stats (Span 7) */}
        <div className="col-span-12 lg:col-span-7 flex flex-col gap-3">
          {/* Image Viewer */}
          <div className="bg-black rounded border border-slate-300 dark:border-slate-800 relative flex items-center justify-center overflow-hidden" style={{ height: '480px' }}>
            {/* DICOM Info */}
            <div className="absolute top-3 left-3 z-10 flex flex-col gap-0.5">
              {activeImage && (
                <>
                  <span className="text-[10px] font-mono text-emerald-500">
                    {activeImage.location}
                  </span>
                  <span className="text-[10px] font-mono text-cyan-400 break-all">
                    {activeImage.filename || 'N/A'}
                  </span>
                </>
              )}
            </div>
            
            {/* Nav Arrows */}
            {hasGallery && sliceImages.length > 1 && (
              <>
                <button 
                  onClick={() => setActiveSliceIdx(Math.max(0, activeSliceIdx - 1))}
                  className="absolute left-1 top-1/2 -translate-y-1/2 z-20 bg-white/60 hover:bg-white/80 dark:bg-black/60 dark:hover:bg-black/80 p-1.5 rounded-full border border-slate-300 dark:border-slate-700 disabled:opacity-30"
                  disabled={activeSliceIdx === 0}
                >
                  <ChevronLeftIcon className="w-4 h-4 text-slate-900 dark:text-white" />
                </button>
                <button 
                  onClick={() => setActiveSliceIdx(Math.min(sliceImages.length - 1, activeSliceIdx + 1))}
                  className="absolute right-1 top-1/2 -translate-y-1/2 z-20 bg-white/60 hover:bg-white/80 dark:bg-black/60 dark:hover:bg-black/80 p-1.5 rounded-full border border-slate-300 dark:border-slate-700 disabled:opacity-30"
                  disabled={activeSliceIdx === sliceImages.length - 1}
                >
                  <ChevronRightIcon className="w-4 h-4 text-slate-900 dark:text-white" />
                </button>
              </>
            )}

            {/* Main Image */}
            <AnimatePresence mode="wait">
              {activeBase64 ? (
                <motion.img
                  key={activeSliceIdx}
                  src={`data:image/png;base64,${activeBase64}`}
                  alt="Analysis"
                  className="max-h-full max-w-full object-contain"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  transition={{ duration: 0.15 }}
                />
              ) : (
                <div className="text-slate-400 font-mono text-sm">NO VISUAL DATA</div>
              )}
            </AnimatePresence>

            {/* Slice Counter */}
            {hasGallery && (
              <div className="absolute bottom-2 right-2 bg-black/70 px-2 py-1 rounded border border-white/10">
                <span className="font-mono text-[10px] text-slate-300 dark:text-slate-400">
                  {activeSliceIdx + 1}/{sliceImages.length}
                </span>
              </div>
            )}
          </div>

          {/* Thumbnail Strip */}
          {hasGallery && (
            <div className="bg-white/60 dark:bg-black/40 backdrop-blur-md rounded border border-slate-200 dark:border-white/10 p-2 shadow-lg">
              <div className="flex gap-1.5 overflow-x-auto pb-1">
                {sliceImages.map((slice, idx) => {
                  const locCount = sliceImages.filter(s => s.location === slice.location).length;
                  return (
                  <motion.button
                    whileHover={{ y: -2, scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                    key={idx}
                    onClick={() => setActiveSliceIdx(idx)}
                    className={`flex-shrink-0 relative rounded overflow-hidden border-2 transition-all duration-150 ${
                      idx === activeSliceIdx 
                        ? 'border-emerald-500 ring-2 ring-emerald-500/40 shadow-[0_0_12px_rgba(16,185,129,0.4)] opacity-100 scale-105 z-10' 
                        : 'border-slate-300 hover:border-slate-400 dark:border-slate-700 dark:hover:border-slate-500 opacity-60 hover:opacity-100'
                    }`}
                    style={{ width: '64px', height: '64px' }}
                  >
                    <img 
                      src={`data:image/png;base64,${slice.image_base64}`}
                      alt={slice.location}
                      className="w-full h-full object-cover"
                    />
                    {locCount > 1 && (
                      <div className="absolute top-0 right-0 bg-cyan-600 text-white text-[7px] font-bold px-0.5 rounded-bl">
                        {locCount}
                      </div>
                    )}
                    <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/90 to-transparent px-0.5 py-0.5">
                      <span className={`text-[8px] font-mono font-bold ${slice.probability > 0.7 ? 'text-red-400' : 'text-amber-400'}`}>
                        {(slice.probability * 100).toFixed(0)}%
                      </span>
                    </div>
                  </motion.button>
                )})}
              </div>
            </div>
          )}

          {/* Stats Cards - MOVED HERE */}
          <div className="grid grid-cols-4 gap-3">
            <motion.div whileHover={{ y: -2 }} className="card p-3 bg-cyan-500/5 dark:bg-cyan-500/10 border-cyan-500/20 shadow-[0_0_15px_rgba(6,182,212,0.05)] transition-all hover:shadow-[0_0_20px_rgba(6,182,212,0.15)] cursor-default">
              <div className="label-text text-[10px] text-cyan-600 dark:text-cyan-400 font-bold uppercase tracking-wider">Inference Time</div>
              <div className="value-text text-lg text-slate-800 dark:text-slate-100">{processing_time.toFixed(2)}s</div>
            </motion.div>
            <motion.div whileHover={{ y: -2 }} className="card p-3 bg-purple-500/5 dark:bg-purple-500/10 border-purple-500/20 shadow-[0_0_15px_rgba(168,85,247,0.05)] transition-all hover:shadow-[0_0_20px_rgba(168,85,247,0.15)] cursor-default">
              <div className="label-text text-[10px] text-purple-600 dark:text-purple-400 font-bold uppercase tracking-wider">Max Confidence</div>
              <div className="value-text text-lg text-slate-800 dark:text-slate-100">{(confidence * 100).toFixed(1)}%</div>
            </motion.div>
            <motion.div whileHover={{ y: -2 }} className="card p-3 bg-amber-500/5 dark:bg-amber-500/10 border-amber-500/20 shadow-[0_0_15px_rgba(245,158,11,0.05)] transition-all hover:shadow-[0_0_20px_rgba(245,158,11,0.15)] cursor-default">
              <div className="label-text text-[10px] text-amber-600 dark:text-amber-400 font-bold uppercase tracking-wider">Findings</div>
              <div className="value-text text-lg text-slate-800 dark:text-slate-100">{sliceImages.length} slices</div>
            </motion.div>
            <motion.div whileHover={{ y: -2 }} className="card p-3 bg-emerald-500/5 dark:bg-emerald-500/10 border-emerald-500/20 shadow-[0_0_15px_rgba(16,185,129,0.05)] transition-all hover:shadow-[0_0_20px_rgba(16,185,129,0.15)] cursor-default">
              <div className="label-text text-[10px] text-emerald-600 dark:text-emerald-400 font-bold uppercase tracking-wider">Modality</div>
              <div className="value-text text-lg text-slate-800 dark:text-slate-100">{result.modality || 'CTA'}</div>
            </motion.div>
          </div>
        </div>

        {/* RIGHT: Segmentation Report (Span 5) - Full Height */}
        <div className="col-span-12 lg:col-span-5 flex flex-col">
          <div className="bg-white/70 dark:bg-slate-900/50 backdrop-blur-xl border-slate-200 dark:border-white/10 border rounded shadow-xl flex-1 flex flex-col h-full overflow-hidden">
            <div className="p-3 border-b border-slate-200/50 dark:border-white/5 bg-slate-50/50 dark:bg-black/20">
              <h2 className="text-xs font-bold bg-clip-text text-transparent bg-gradient-to-r from-slate-600 to-slate-400 dark:from-white dark:to-slate-400 uppercase tracking-widest">
                Segmentation Report
              </h2>
            </div>
            <div className="overflow-auto flex-1">
              <table className="w-full">
                <thead className="sticky top-0 bg-white/95 dark:bg-slate-900/95 backdrop-blur-xl z-10">
                  <tr className="border-b border-slate-200/50 dark:border-white/5">
                    <th className="text-left py-2 px-3 font-mono text-[10px] text-slate-600 dark:text-slate-500">Region</th>
                    <th className="text-right py-2 px-3 font-mono text-[10px] text-slate-600 dark:text-slate-500">Prob</th>
                    <th className="text-center py-2 px-3 font-mono text-[10px] text-slate-600 dark:text-slate-500">Status</th>
                  </tr>
                </thead>
                <tbody>
                  {predictions
                    .filter(p => p.location !== 'Aneurysm Present')
                    .sort((a, b) => b.probability - a.probability)
                    .map((pred, idx) => {
                      const galleryIdx = sliceImages.findIndex(s => s.location === pred.location)
                      return (
                      <motion.tr 
                        initial={{ opacity: 0, x: -10 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: idx * 0.05 }}
                        whileHover={{ scale: 1.005 }}
                        key={idx} 
                        className={`border-b border-slate-100 dark:border-white/5 transition-colors cursor-pointer relative ${
                          pred.detected ? 'bg-red-50 hover:bg-red-100 dark:bg-red-500/10 dark:hover:bg-red-500/20' : 'hover:bg-slate-50 dark:hover:bg-white/5'
                        } ${galleryIdx === activeSliceIdx && galleryIdx !== -1 ? 'ring-1 ring-inset ring-emerald-500/40 bg-emerald-50/30 dark:bg-emerald-500/10 z-10 shadow-sm' : ''}`}
                        onClick={() => galleryIdx !== -1 && setActiveSliceIdx(galleryIdx)}
                      >
                        <td className="py-1.5 px-3 text-xs text-slate-700 dark:text-slate-300">
                          <div className="flex items-center gap-1">
                            {galleryIdx !== -1 && <EyeIcon className="w-3 h-3 text-emerald-500 flex-shrink-0" />}
                            <span className="truncate">{pred.location}</span>
                          </div>
                        </td>
                        <td className="py-1.5 px-3 text-right font-mono text-xs text-slate-600 dark:text-slate-400">
                          {(pred.probability).toFixed(4)}
                        </td>
                        <td className="py-1.5 px-3 text-center">
                          {pred.detected ? (
                            <span className="badge badge-danger text-[10px]">DETECTED</span>
                          ) : (
                            <span className="badge badge-neutral text-[10px]">NORMAL</span>
                          )}
                        </td>
                      </motion.tr>
                    )})}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>

      {/* BOTTOM: Full-Width Detailed Findings Table */}
      <div className="bg-white/70 dark:bg-slate-900/50 backdrop-blur-xl border-slate-200 dark:border-white/10 border rounded shadow-xl overflow-hidden mt-2">
        <div className="p-3 border-b border-slate-200/50 dark:border-white/5 bg-slate-50/50 dark:bg-black/20 flex items-center justify-between gap-4">
          <h2 className="text-xs font-bold bg-clip-text text-transparent bg-gradient-to-r from-slate-600 to-slate-400 dark:from-white dark:to-slate-400 uppercase tracking-widest flex-shrink-0">
            Detailed Findings (Multi-Slice)
          </h2>
          <div className="relative max-w-sm w-full">
            <MagnifyingGlassIcon className="w-3.5 h-3.5 text-slate-400 dark:text-slate-500 absolute left-2.5 top-1/2 -translate-y-1/2" />
            <input
              type="text"
              placeholder="Search by filename..."
              value={filenameSearch}
              onChange={(e) => setFilenameSearch(e.target.value)}
              className="w-full bg-white/50 backdrop-blur-sm border-slate-300 text-slate-900 placeholder-slate-400 dark:bg-black/40 border dark:border-white/10 rounded text-xs font-mono dark:text-white pl-8 pr-3 py-1.5 dark:placeholder-slate-500 focus:outline-none focus:border-cyan-500 focus:ring-1 focus:ring-cyan-500/50 transition-colors shadow-inner"
            />
            {filenameSearch && (
              <button
                onClick={() => setFilenameSearch('')}
                className="absolute right-2 top-1/2 -translate-y-1/2 text-slate-400 hover:text-slate-600 dark:text-slate-500 dark:hover:text-white text-xs"
              >
                ✕
              </button>
            )}
          </div>
        </div>
        <div className="overflow-auto" style={{ maxHeight: '450px' }}>
          <table className="w-full">
            <thead className="sticky top-0 bg-white/95 dark:bg-slate-900/95 backdrop-blur-xl z-10">
              <tr className="border-b border-slate-200/50 dark:border-white/5">
                <th className="text-left py-2.5 px-4 font-mono text-[10px] text-slate-600 dark:text-slate-500 uppercase">Region</th>
                <th className="text-left py-2.5 px-4 font-mono text-[10px] text-slate-600 dark:text-slate-500">File Name</th>
                <th className="text-right py-2.5 px-4 font-mono text-[10px] text-slate-600 dark:text-slate-500">XY</th>
                <th className="text-right py-2.5 px-4 font-mono text-[10px] text-slate-600 dark:text-slate-500">Probability</th>
              </tr>
            </thead>
            <tbody>
              {predictions
                .filter(p => p.detected && p.detailed_coordinates && p.detailed_coordinates.length > 0)
                .flatMap(pred => {
                  const coords = filenameSearch
                    ? pred.detailed_coordinates.filter(d => (d.filename || '').toLowerCase().includes(filenameSearch.toLowerCase()))
                    : pred.detailed_coordinates;
                  return coords.map((detail, i) => {
                    const galleryIdx = sliceImages.findIndex(
                      s => s.location === pred.location && s.slice_z === detail.z
                    );
                    return (
                    <motion.tr 
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: i * 0.03 + 0.1 }}
                      whileHover={{ scale: 1.002, x: 4 }}
                      key={`${pred.location}-${i}`} 
                      className={`border-b border-slate-100 dark:border-white/5 transition-colors hover:bg-slate-50 dark:hover:bg-white/5 cursor-pointer ${
                        galleryIdx === activeSliceIdx && galleryIdx !== -1 ? 'bg-emerald-50/50 dark:bg-emerald-500/10 shadow-sm z-10 relative' : ''
                      }`}
                      onClick={() => galleryIdx !== -1 && setActiveSliceIdx(galleryIdx)}
                    >
                      <td className="py-1.5 px-4 text-xs text-slate-700 dark:text-slate-300">
                        {i === 0 ? (
                          <div className="flex items-center gap-1.5">
                            <span className="font-medium">{pred.location}</span>
                            {pred.detailed_coordinates.length > 1 && (
                              <span className="bg-cyan-800 text-cyan-200 text-[9px] px-1 rounded flex-shrink-0">
                                {pred.detailed_coordinates.length} slices
                              </span>
                            )}
                          </div>
                        ) : ''}
                      </td>
                      <td className="py-1.5 px-4 text-cyan-600 dark:text-cyan-400 font-mono text-xs">
                        {detail.filename || '-'}
                      </td>
                      <td className="py-1.5 px-4 text-right text-slate-500 dark:text-slate-400 font-mono text-xs">
                        {detail.x}, {detail.y}
                      </td>
                      <td className="py-1.5 px-4 text-right text-slate-500 dark:text-slate-400 font-mono text-xs">
                        {detail.prob.toFixed(4)}
                      </td>
                    </motion.tr>
                  )})
                })
              }
              {predictions.filter(p => p.detected && p.detailed_coordinates && p.detailed_coordinates.length > 0).length === 0 && (
                <tr><td colSpan="4" className="text-center py-6 text-slate-500 text-xs">No multi-slice detections found.</td></tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}
