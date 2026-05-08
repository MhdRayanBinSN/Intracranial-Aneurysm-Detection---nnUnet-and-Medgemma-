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
  CpuChipIcon,
  ClockIcon,
  ChartBarIcon,
  BeakerIcon,
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
      <div className="min-h-screen flex items-center justify-center bg-slate-50 dark:bg-slate-950">
        <div className="text-center">
          <h2 className="text-2xl font-bold mb-4 text-slate-800 dark:text-white">No Report Data</h2>
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

  const riskColor = overall_risk === 'High'
    ? { bg: 'bg-red-500/10 dark:bg-red-500/20', border: 'border-red-400/40 dark:border-red-500/30', dot: 'bg-red-500', text: 'text-red-700 dark:text-red-400', pulse: true }
    : overall_risk === 'Moderate'
    ? { bg: 'bg-amber-500/10 dark:bg-amber-500/20', border: 'border-amber-400/40 dark:border-amber-500/30', dot: 'bg-amber-500', text: 'text-amber-700 dark:text-amber-400', pulse: false }
    : { bg: 'bg-emerald-500/10 dark:bg-emerald-500/20', border: 'border-emerald-400/40 dark:border-emerald-500/30', dot: 'bg-emerald-500', text: 'text-emerald-700 dark:text-emerald-400', pulse: false }

  return (
    <div className="min-h-screen bg-slate-100 dark:bg-[#09090b] text-slate-900 dark:text-white">

      {/* ── Top Bar ── */}
      <div className="sticky top-0 z-30 bg-white/90 dark:bg-slate-950/90 backdrop-blur-xl border-b border-slate-200 dark:border-slate-800 px-5 py-2.5 flex items-center justify-between gap-4 shadow-sm">
        <button
          onClick={() => navigate(-1)}
          className="flex items-center gap-2 text-slate-500 hover:text-slate-900 dark:text-slate-400 dark:hover:text-white transition-colors text-sm font-medium group"
        >
          <ArrowLeftIcon className="w-4 h-4 group-hover:-translate-x-0.5 transition-transform" />
          Combined Results
        </button>

        <div className="flex items-center gap-3">
          {/* Risk badge */}
          <div className={`flex items-center gap-2 px-3 py-1.5 rounded-full border text-sm font-bold tracking-wide shadow-sm ${riskColor.bg} ${riskColor.border} ${riskColor.text}`}>
            <span className={`w-2 h-2 rounded-full flex-shrink-0 ${riskColor.dot} ${riskColor.pulse ? 'animate-pulse shadow-[0_0_8px_rgba(239,68,68,0.8)]' : ''}`} />
            RISK: {overall_risk.toUpperCase()}
          </div>
          <button className="btn-secondary flex items-center gap-1.5 text-xs py-1.5">
            <PrinterIcon className="w-3.5 h-3.5" /> Print
          </button>
          <button className="btn-primary flex items-center gap-1.5 text-xs py-1.5">
            <ArrowDownTrayIcon className="w-3.5 h-3.5" /> Export
          </button>
        </div>
      </div>

      <div className="p-5 space-y-5">

        {/* ── Stats Row ── */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          {[
            { icon: ClockIcon,    label: 'Inference Time',  value: `${processing_time.toFixed(2)}s`,             color: 'text-cyan-600 dark:text-cyan-400',   glow: 'shadow-cyan-500/10',   ring: 'ring-cyan-500/20 dark:ring-cyan-500/30'   },
            { icon: ChartBarIcon, label: 'Max Confidence',  value: `${(confidence * 100).toFixed(1)}%`,          color: 'text-violet-600 dark:text-violet-400', glow: 'shadow-violet-500/10', ring: 'ring-violet-500/20 dark:ring-violet-500/30' },
            { icon: BeakerIcon,   label: 'Detected Slices', value: `${sliceImages.length}`,                      color: 'text-amber-600 dark:text-amber-400',  glow: 'shadow-amber-500/10',  ring: 'ring-amber-500/20 dark:ring-amber-500/30'  },
            { icon: CpuChipIcon,  label: 'Modality',        value: result.modality || 'CTA',                    color: 'text-emerald-600 dark:text-emerald-400', glow: 'shadow-emerald-500/10', ring: 'ring-emerald-500/20 dark:ring-emerald-500/30' },
          ].map(({ icon: Icon, label, value, color, glow, ring }) => (
            <motion.div
              key={label}
              whileHover={{ y: -2 }}
              className={`bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 p-4 shadow-sm ${glow} ring-1 ${ring} flex items-center gap-3`}
            >
              <div className={`p-2 rounded-lg bg-slate-100 dark:bg-slate-800 ${color}`}>
                <Icon className="w-4 h-4" />
              </div>
              <div>
                <p className="text-[10px] uppercase tracking-widest font-semibold text-slate-400 dark:text-slate-500">{label}</p>
                <p className={`text-lg font-bold font-mono leading-tight ${color}`}>{value}</p>
              </div>
            </motion.div>
          ))}
        </div>

        {/* ── Main Content: Image + Report ── */}
        <div className="grid grid-cols-12 gap-5">

          {/* LEFT: Image Viewer + Thumbnails (col-span 7) */}
          <div className="col-span-12 lg:col-span-7 flex flex-col gap-3">

            {/* Image Viewer */}
            <div
              className="relative bg-black rounded-2xl overflow-hidden border border-slate-300 dark:border-slate-700/80 shadow-2xl shadow-black/20 dark:shadow-black/60 flex items-center justify-center"
              style={{ height: '560px' }}
            >
              {/* DICOM Metadata Overlay — top-left pill badges */}
              {activeImage && (
                <div className="absolute top-3 left-3 z-10 flex flex-col gap-1.5">
                  <span className="inline-flex items-center gap-1.5 bg-black/70 backdrop-blur-sm text-emerald-400 font-mono text-[10px] px-2.5 py-1 rounded-full border border-emerald-500/30 shadow-sm">
                    <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
                    {activeImage.location}
                  </span>
                  {activeImage.filename && (
                    <span className="inline-flex items-center bg-black/70 backdrop-blur-sm text-cyan-400 font-mono text-[9px] px-2.5 py-1 rounded-full border border-cyan-500/30 max-w-[280px] truncate shadow-sm">
                      {activeImage.filename}
                    </span>
                  )}
                </div>
              )}

              {/* Keyboard hint */}
              {hasGallery && sliceImages.length > 1 && (
                <div className="absolute top-3 right-3 z-10">
                  <span className="bg-black/60 backdrop-blur-sm text-slate-400 font-mono text-[9px] px-2 py-1 rounded-full border border-white/10">
                    ← → navigate
                  </span>
                </div>
              )}

              {/* Nav Arrows */}
              {hasGallery && sliceImages.length > 1 && (
                <>
                  <button
                    onClick={() => setActiveSliceIdx(Math.max(0, activeSliceIdx - 1))}
                    disabled={activeSliceIdx === 0}
                    className="absolute left-3 top-1/2 -translate-y-1/2 z-20 bg-white/20 hover:bg-white/40 dark:bg-black/50 dark:hover:bg-black/70 p-2 rounded-full border border-white/20 disabled:opacity-20 disabled:cursor-not-allowed transition-all backdrop-blur-sm shadow-lg"
                  >
                    <ChevronLeftIcon className="w-5 h-5 text-white" />
                  </button>
                  <button
                    onClick={() => setActiveSliceIdx(Math.min(sliceImages.length - 1, activeSliceIdx + 1))}
                    disabled={activeSliceIdx === sliceImages.length - 1}
                    className="absolute right-3 top-1/2 -translate-y-1/2 z-20 bg-white/20 hover:bg-white/40 dark:bg-black/50 dark:hover:bg-black/70 p-2 rounded-full border border-white/20 disabled:opacity-20 disabled:cursor-not-allowed transition-all backdrop-blur-sm shadow-lg"
                  >
                    <ChevronRightIcon className="w-5 h-5 text-white" />
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
                    initial={{ opacity: 0, scale: 0.98 }}
                    animate={{ opacity: 1, scale: 1 }}
                    exit={{ opacity: 0, scale: 0.98 }}
                    transition={{ duration: 0.2 }}
                  />
                ) : (
                  <div className="text-slate-500 font-mono text-sm tracking-widest">NO VISUAL DATA</div>
                )}
              </AnimatePresence>

              {/* Slice Counter - bottom right */}
              {hasGallery && (
                <div className="absolute bottom-3 right-3 bg-black/70 backdrop-blur-sm border border-white/10 px-3 py-1 rounded-full">
                  <span className="font-mono text-[11px] text-slate-300">
                    {activeSliceIdx + 1} / {sliceImages.length}
                  </span>
                </div>
              )}

              {/* Detection probability overlay - bottom left */}
              {activeImage && (
                <div className="absolute bottom-3 left-3">
                  <span className={`font-mono text-[11px] font-bold px-2.5 py-1 rounded-full border backdrop-blur-sm ${
                    activeImage.probability > 0.7
                      ? 'bg-red-500/20 border-red-500/40 text-red-300'
                      : 'bg-amber-500/20 border-amber-500/40 text-amber-300'
                  }`}>
                    {(activeImage.probability * 100).toFixed(1)}% confidence
                  </span>
                </div>
              )}
            </div>

            {/* Thumbnail Strip */}
            {hasGallery && (
              <div className="bg-white dark:bg-slate-900/80 rounded-xl border border-slate-200 dark:border-slate-800 p-3 shadow-sm">
                <p className="text-[9px] font-bold uppercase tracking-widest text-slate-400 dark:text-slate-500 mb-2 px-0.5">Detected Slices</p>
                <div className="flex gap-2 overflow-x-auto pb-1">
                  {sliceImages.map((slice, idx) => {
                    const locCount = sliceImages.filter(s => s.location === slice.location).length
                    const isActive = idx === activeSliceIdx
                    return (
                      <motion.button
                        whileHover={{ y: -3, scale: 1.06 }}
                        whileTap={{ scale: 0.94 }}
                        key={idx}
                        onClick={() => setActiveSliceIdx(idx)}
                        className={`flex-shrink-0 relative rounded-lg overflow-hidden border-2 transition-all duration-150 ${
                          isActive
                            ? 'border-emerald-500 ring-2 ring-emerald-500/40 shadow-[0_0_16px_rgba(16,185,129,0.45)] opacity-100 z-10'
                            : 'border-slate-200 dark:border-slate-700 hover:border-slate-400 dark:hover:border-slate-500 opacity-55 hover:opacity-90'
                        }`}
                        style={{ width: '80px', height: '80px' }}
                      >
                        <img
                          src={`data:image/png;base64,${slice.image_base64}`}
                          alt={slice.location}
                          className="w-full h-full object-cover"
                        />
                        {/* Multi-slice badge */}
                        {locCount > 1 && (
                          <div className="absolute top-0.5 right-0.5 bg-cyan-600 text-white text-[7px] font-bold px-1 py-0.5 rounded">
                            {locCount}
                          </div>
                        )}
                        {/* Probability bar */}
                        <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/90 via-black/40 to-transparent px-1 py-1">
                          <span className={`text-[9px] font-mono font-bold ${slice.probability > 0.7 ? 'text-red-400' : 'text-amber-400'}`}>
                            {(slice.probability * 100).toFixed(0)}%
                          </span>
                        </div>
                        {/* Active indicator */}
                        {isActive && (
                          <div className="absolute inset-0 ring-2 ring-inset ring-emerald-400/60 rounded-lg pointer-events-none" />
                        )}
                      </motion.button>
                    )
                  })}
                </div>
              </div>
            )}
          </div>

          {/* RIGHT: Segmentation Report (col-span 5) */}
          <div className="col-span-12 lg:col-span-5 flex flex-col">
            <div className="bg-white dark:bg-slate-900/80 rounded-2xl border border-slate-200 dark:border-slate-800 shadow-sm overflow-hidden flex flex-col h-full">
              {/* Header */}
              <div className="px-5 py-4 border-b border-slate-100 dark:border-slate-800 bg-slate-50 dark:bg-slate-900 flex items-center gap-3 flex-shrink-0">
                <div className="p-2 bg-blue-100 dark:bg-blue-900/30 rounded-lg">
                  <CpuChipIcon className="w-4 h-4 text-blue-600 dark:text-blue-400" />
                </div>
                <div>
                  <h2 className="text-xs font-bold uppercase tracking-widest text-slate-700 dark:text-slate-300">
                    Segmentation Report
                  </h2>
                  <p className="text-[10px] text-slate-400 dark:text-slate-500 font-mono mt-0.5">
                    {predictions.filter(p => p.detected).length} region(s) flagged
                  </p>
                </div>
              </div>

              {/* Table */}
              <div className="overflow-auto flex-1" style={{ maxHeight: '580px' }}>
                <table className="w-full">
                  <thead className="sticky top-0 bg-white/95 dark:bg-slate-900/95 backdrop-blur-xl z-10 border-b border-slate-100 dark:border-slate-800">
                    <tr>
                      <th className="text-left py-2.5 px-4 font-mono text-[10px] text-slate-500 uppercase tracking-wider">Region</th>
                      <th className="text-right py-2.5 px-4 font-mono text-[10px] text-slate-500 uppercase tracking-wider">Prob</th>
                      <th className="text-center py-2.5 px-4 font-mono text-[10px] text-slate-500 uppercase tracking-wider">Status</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-50 dark:divide-slate-800/60">
                    {predictions
                      .filter(p => p.location !== 'Aneurysm Present')
                      .sort((a, b) => b.probability - a.probability)
                      .map((pred, idx) => {
                        const galleryIdx = sliceImages.findIndex(s => s.location === pred.location)
                        const isCurrentSlice = galleryIdx === activeSliceIdx && galleryIdx !== -1
                        return (
                          <motion.tr
                            initial={{ opacity: 0, x: -8 }}
                            animate={{ opacity: 1, x: 0 }}
                            transition={{ delay: idx * 0.04 }}
                            key={idx}
                            onClick={() => galleryIdx !== -1 && setActiveSliceIdx(galleryIdx)}
                            className={`transition-all duration-150 group cursor-pointer ${
                              pred.detected
                                ? 'bg-red-50/80 hover:bg-red-100/80 dark:bg-red-500/8 dark:hover:bg-red-500/15'
                                : 'hover:bg-slate-50 dark:hover:bg-slate-800/40'
                            } ${isCurrentSlice ? 'ring-1 ring-inset ring-emerald-500/50 bg-emerald-50/50 dark:bg-emerald-500/10 relative z-10' : ''}`}
                          >
                            <td className="py-2.5 px-4 text-xs text-slate-700 dark:text-slate-300">
                              <div className="flex items-center gap-1.5">
                                {galleryIdx !== -1 && (
                                  <EyeIcon className={`w-3 h-3 flex-shrink-0 transition-colors ${isCurrentSlice ? 'text-emerald-500' : 'text-slate-300 dark:text-slate-600 group-hover:text-emerald-400'}`} />
                                )}
                                <span className={`truncate ${pred.detected ? 'font-semibold text-rose-700 dark:text-rose-300' : ''}`}>
                                  {pred.location}
                                </span>
                              </div>
                            </td>
                            <td className="py-2.5 px-4 text-right font-mono text-xs text-slate-500 dark:text-slate-400">
                              {pred.probability.toFixed(4)}
                            </td>
                            <td className="py-2.5 px-4 text-center">
                              {pred.detected ? (
                                <span className="badge badge-danger text-[10px]">DETECTED</span>
                              ) : (
                                <span className="badge badge-neutral text-[10px]">NORMAL</span>
                              )}
                            </td>
                          </motion.tr>
                        )
                      })}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>

        {/* ── Bottom: Detailed Findings Table ── */}
        <div className="bg-white dark:bg-slate-900/80 rounded-2xl border border-slate-200 dark:border-slate-800 shadow-sm overflow-hidden">
          {/* Header */}
          <div className="px-5 py-4 border-b border-slate-100 dark:border-slate-800 bg-slate-50 dark:bg-slate-900 flex items-center justify-between gap-4 flex-wrap">
            <div>
              <h2 className="text-xs font-bold uppercase tracking-widest text-slate-700 dark:text-slate-300">
                Detailed Findings — Multi-Slice Coordinates
              </h2>
              <p className="text-[10px] text-slate-400 dark:text-slate-500 font-mono mt-0.5">
                Click a row to jump to that slice in the viewer
              </p>
            </div>
            {/* Search */}
            <div className="relative w-full sm:w-64">
              <MagnifyingGlassIcon className="w-3.5 h-3.5 text-slate-400 absolute left-3 top-1/2 -translate-y-1/2 pointer-events-none" />
              <input
                type="text"
                placeholder="Search filename…"
                value={filenameSearch}
                onChange={e => setFilenameSearch(e.target.value)}
                className="w-full bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-lg text-xs font-mono text-slate-900 dark:text-white placeholder-slate-400 dark:placeholder-slate-500 pl-8 pr-7 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500/40 focus:border-blue-400 dark:focus:border-blue-500 transition-colors shadow-sm"
              />
              {filenameSearch && (
                <button
                  onClick={() => setFilenameSearch('')}
                  className="absolute right-2.5 top-1/2 -translate-y-1/2 text-slate-400 hover:text-slate-600 dark:hover:text-white text-xs"
                >✕</button>
              )}
            </div>
          </div>

          <div className="overflow-auto" style={{ maxHeight: '420px' }}>
            <table className="w-full">
              <thead className="sticky top-0 bg-white/95 dark:bg-slate-900/95 backdrop-blur-xl z-10 border-b border-slate-100 dark:border-slate-800">
                <tr>
                  <th className="text-left py-2.5 px-5 font-mono text-[10px] text-slate-500 uppercase tracking-wider">Region</th>
                  <th className="text-left py-2.5 px-5 font-mono text-[10px] text-slate-500 uppercase tracking-wider">File Name</th>
                  <th className="text-right py-2.5 px-5 font-mono text-[10px] text-slate-500 uppercase tracking-wider">XY</th>
                  <th className="text-right py-2.5 px-5 font-mono text-[10px] text-slate-500 uppercase tracking-wider">Probability</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-50 dark:divide-slate-800/60">
                {predictions
                  .filter(p => p.detected && p.detailed_coordinates && p.detailed_coordinates.length > 0)
                  .flatMap(pred => {
                    const coords = filenameSearch
                      ? pred.detailed_coordinates.filter(d => (d.filename || '').toLowerCase().includes(filenameSearch.toLowerCase()))
                      : pred.detailed_coordinates
                    return coords.map((detail, i) => {
                      const galleryIdx = sliceImages.findIndex(
                        s => s.location === pred.location && s.slice_z === detail.z
                      )
                      const isCurrentSlice = galleryIdx === activeSliceIdx && galleryIdx !== -1
                      return (
                        <motion.tr
                          initial={{ opacity: 0, y: 6 }}
                          animate={{ opacity: 1, y: 0 }}
                          transition={{ delay: i * 0.025 + 0.1 }}
                          key={`${pred.location}-${i}`}
                          onClick={() => galleryIdx !== -1 && setActiveSliceIdx(galleryIdx)}
                          className={`transition-all duration-150 cursor-pointer hover:bg-slate-50 dark:hover:bg-slate-800/40 ${
                            isCurrentSlice ? 'bg-emerald-50/60 dark:bg-emerald-500/10 relative z-10' : ''
                          }`}
                        >
                          <td className="py-2.5 px-5 text-xs text-slate-700 dark:text-slate-300">
                            {i === 0 ? (
                              <div className="flex items-center gap-2">
                                <span className="font-semibold text-slate-800 dark:text-slate-200">{pred.location}</span>
                                {pred.detailed_coordinates.length > 1 && (
                                  <span className="bg-cyan-100 dark:bg-cyan-900/40 text-cyan-700 dark:text-cyan-400 text-[9px] font-bold px-1.5 py-0.5 rounded border border-cyan-200 dark:border-cyan-800">
                                    {pred.detailed_coordinates.length} slices
                                  </span>
                                )}
                              </div>
                            ) : (
                              <span className="text-slate-300 dark:text-slate-600 pl-2 text-[10px] font-mono">↳</span>
                            )}
                          </td>
                          <td className="py-2.5 px-5 text-cyan-600 dark:text-cyan-400 font-mono text-[11px] truncate max-w-[200px]">
                            {detail.filename || '—'}
                          </td>
                          <td className="py-2.5 px-5 text-right text-slate-500 dark:text-slate-400 font-mono text-xs">
                            {detail.x}, {detail.y}
                          </td>
                          <td className="py-2.5 px-5 text-right font-mono text-xs">
                            <span className={detail.prob > 0.7 ? 'text-rose-600 dark:text-rose-400 font-bold' : 'text-slate-500 dark:text-slate-400'}>
                              {detail.prob.toFixed(4)}
                            </span>
                          </td>
                        </motion.tr>
                      )
                    })
                  })}
                {predictions.filter(p => p.detected && p.detailed_coordinates && p.detailed_coordinates.length > 0).length === 0 && (
                  <tr>
                    <td colSpan="4" className="text-center py-10 text-slate-400 dark:text-slate-500 text-sm font-mono">
                      No multi-slice detections found.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </div>

      </div>
    </div>
  )
}
