import { useState, useEffect } from 'react'
import { useLocation, Link, useNavigate } from 'react-router-dom'
import { motion } from 'framer-motion'
import {
  ArrowLeftIcon, SparklesIcon, CpuChipIcon, ExclamationTriangleIcon,
  CheckCircleIcon, ArrowRightIcon, DocumentChartBarIcon, ShieldExclamationIcon,
  ChartBarIcon
} from '@heroicons/react/24/outline'

export default function CombinedResults() {
  const location = useLocation()
  const navigate = useNavigate()
  const { nnunetResult, gemmaResult, nnunetError, gemmaError } = location.state || {}

  if (!nnunetResult && !gemmaResult) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <h2 className="text-2xl font-bold mb-4">No Results Available</h2>
          <Link to="/analysis" className="btn-primary inline-flex items-center gap-2">
            <ArrowLeftIcon className="w-4 h-4" /> Back to Upload
          </Link>
        </div>
      </div>
    )
  }

  const navigateToNnunet = () => {
    if (!nnunetResult) return
    navigate(`/results/nnUnet`, { 
      state: { result: nnunetResult, fullState: location.state } 
    })
  }

  const navigateToMedGemma = () => {
    if (!gemmaResult) return
    navigate(`/results/medgemma`, { 
      state: { result: gemmaResult, fullState: location.state } 
    })
  }

  const seriesGT = gemmaResult?.series_ground_truth || null;
  const gemmaLocations = gemmaResult?.findings_by_location ? Object.keys(gemmaResult.findings_by_location) : [];
  
  const detectedNnUnet = nnunetResult?.predictions?.filter(p => p.detected && p.location !== 'Aneurysm Present') || [];
  const detectedGemma = gemmaResult?.findings_by_location ? Object.entries(gemmaResult.findings_by_location) : [];


  const renderComparisonGraph = () => {
    if (!seriesGT) return null;

    const tableData = [];
    const nnunetPredictions = nnunetResult?.predictions || [];

    Object.entries(seriesGT).forEach(([key, val]) => {
      if (key !== "Aneurysm Present" && key !== "PatientAge" && key !== "PatientSex" && key !== "Modality" && key !== "SeriesInstanceUID") {
        const isGtPresent = val === 1;
        const isGemmaDetected = gemmaLocations.includes(key);
        
        // Find nnU-Net specific prediction for this exact location
        const nnunetLoc = nnunetPredictions.find(p => p.location === key);
        const isNnunetDetected = nnunetLoc ? nnunetLoc.detected : false;
        let gemmaColor = "text-slate-400 font-medium";
        if (isGtPresent && isGemmaDetected) gemmaColor = "text-emerald-500 font-bold";
        else if (!isGtPresent && isGemmaDetected) gemmaColor = "text-amber-500 font-bold";
        else if (isGtPresent && !isGemmaDetected) gemmaColor = "text-rose-500 font-bold";

        let nnunetColor = "text-slate-400 font-medium";
        if (isGtPresent && isNnunetDetected) nnunetColor = "text-emerald-500 font-bold";
        else if (!isGtPresent && isNnunetDetected) nnunetColor = "text-amber-500 font-bold";
        else if (isGtPresent && !isNnunetDetected) nnunetColor = "text-rose-500 font-bold";

        tableData.push({ 
          location: key, 
          gt: isGtPresent, 
          gemma: isGemmaDetected, 
          nnunet: isNnunetDetected,
          gemmaColor,
          nnunetColor
        });
      }
    });

    return (
      <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.2 }}
        className="max-w-6xl mx-auto mb-12 overflow-hidden rounded-3xl border border-slate-200/50 dark:border-slate-800 bg-white/70 dark:bg-slate-900/50 backdrop-blur-2xl shadow-[0_8px_30px_rgb(0,0,0,0.04)] dark:shadow-[0_8px_30px_rgb(0,0,0,0.2)] relative">
        <div className="absolute top-0 left-1/4 w-96 h-96 bg-primary-500/10 rounded-full blur-3xl -z-10 mix-blend-multiply dark:mix-blend-screen pointer-events-none"></div>
        <div className="absolute bottom-0 right-1/4 w-96 h-96 bg-violet-500/10 rounded-full blur-3xl -z-10 mix-blend-multiply dark:mix-blend-screen pointer-events-none"></div>

        <div className="p-8 border-b border-slate-200/50 dark:border-slate-800/80 flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <h3 className="text-2xl font-light tracking-tight text-slate-900 dark:text-white flex items-center gap-4">
              <span className="p-2.5 bg-gradient-to-br from-primary-500 to-violet-600 rounded-2xl text-white shadow-lg shadow-primary-500/25">
                <ShieldExclamationIcon className="w-5 h-5" />
              </span>
              Diagnostic Validation Matrix
            </h3>
            <p className="text-sm text-slate-500 dark:text-slate-400 mt-3 font-medium tracking-wide">
              Cross-referencing MedGemma semantic detections against confirmed dataset ground truth.
            </p>
          </div>
          <div className="self-start md:self-auto px-4 py-2 bg-slate-100/80 dark:bg-slate-800/80 rounded-xl text-xs font-bold uppercase tracking-widest text-slate-500 dark:text-slate-400 border border-slate-200/50 dark:border-slate-700/50 shadow-sm flex items-center gap-2">
            <DocumentChartBarIcon className="w-4 h-4" /> Dataset: train.csv
          </div>
        </div>
        
        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse min-w-[800px]">
            <thead>
              <tr className="bg-slate-50/50 dark:bg-slate-900/50 text-[11px] uppercase tracking-[0.2em] font-bold text-slate-400 border-b border-slate-200/50 dark:border-slate-800/80">
                <th className="py-6 px-8">Anatomical Region</th>
                <th className="py-6 px-8">Dataset Truth</th>
                <th className="py-6 px-8">MedGemma (Semantic)</th>
                <th className="py-6 px-8">nnU-Net (Volumetric)</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100/80 dark:divide-slate-800/50 text-sm">
              {tableData.map((row, idx) => (
                <tr key={idx} className="hover:bg-slate-50/80 dark:hover:bg-slate-800/30 transition-all duration-300 group">
                  <td className="py-4 px-8 font-medium text-slate-800 dark:text-slate-200 group-hover:text-primary-600 dark:group-hover:text-primary-400 transition-colors">
                    {row.location}
                  </td>
                  <td className="py-4 px-8">
                    {row.gt ? (
                      <span className="text-slate-800 dark:text-slate-200 font-bold tracking-wide">
                        Present
                      </span>
                    ) : (
                      <span className="text-slate-400 font-medium tracking-wide">
                        Not Present
                      </span>
                    )}
                  </td>
                  <td className="py-4 px-8">
                    <span className={`${row.gemmaColor} tracking-wide`}>
                      {row.gemma ? 'Detected' : 'Negative'}
                    </span>
                  </td>
                  <td className="py-4 px-8">
                    <span className={`${row.nnunetColor} tracking-wide`}>
                      {row.nnunet ? 'Detected' : 'Negative'}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </motion.div>
    );
  };

  return (
    <div className="min-h-screen py-12 px-4 bg-slate-50 dark:bg-slate-950 text-slate-900 dark:text-white">
      <div className="max-w-5xl mx-auto">

        {/* Header */}
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="mb-10 text-center">
          <Link to="/analysis" className="inline-flex items-center gap-2 text-slate-500 hover:text-slate-900 dark:hover:text-white transition-colors mb-6 group">
            <ArrowLeftIcon className="w-4 h-4 group-hover:-translate-x-1 transition-transform" />
            Back to Upload
          </Link>
          <div className="flex justify-center mb-4">
             <div className="flex items-center gap-3 bg-white dark:bg-slate-900 shadow-sm border border-slate-200 dark:border-slate-800 p-2 pr-4 rounded-full">
                <div className="bg-primary-100 text-primary-600 dark:bg-primary-900/30 dark:text-primary-400 p-2 rounded-full">
                   <DocumentChartBarIcon className="w-5 h-5" />
                </div>
                <span className="font-semibold text-sm uppercase tracking-widest text-slate-700 dark:text-slate-300">Dual-Model Analysis Complete</span>
             </div>
          </div>
          <h1 className="text-4xl font-bold mb-3">
            Analysis <span className="text-primary-600 dark:text-primary-400">Summary</span>
          </h1>
          <p className="text-slate-500 dark:text-slate-400 text-lg max-w-3xl mx-auto">
            Your scans have been processed by both diagnostic pipelines. Select a report below to view detailed findings.
          </p>
        </motion.div>

        {/* Error banners */}
        <div className="space-y-3 mb-8">
          {nnunetError && (
            <div className="flex items-center gap-3 p-4 bg-amber-50 dark:bg-amber-900/10 border border-amber-300 dark:border-amber-700 rounded-xl text-amber-700 dark:text-amber-400 text-sm max-w-3xl mx-auto">
              <ExclamationTriangleIcon className="w-5 h-5 flex-shrink-0" />
              <span><strong>nnU-Net Model Failed:</strong> {nnunetError}</span>
            </div>
          )}
          {gemmaError && (
            <div className="flex items-center gap-3 p-4 bg-amber-50 dark:bg-amber-900/10 border border-amber-300 dark:border-amber-700 rounded-xl text-amber-700 dark:text-amber-400 text-sm max-w-3xl mx-auto">
              <ExclamationTriangleIcon className="w-5 h-5 flex-shrink-0" />
              <span><strong>MedGemma Model Failed:</strong> {gemmaError}</span>
            </div>
          )}
        </div>

        {/* Cards Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8 max-w-4xl mx-auto mb-12">

          {/* ── nnU-Net Card ── */}
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 }}
            className={`relative flex flex-col p-8 rounded-2xl border-2 transition-all ${nnunetResult ? 'bg-white dark:bg-slate-900 border-blue-200 dark:border-blue-900/50 hover:border-blue-400 dark:hover:border-blue-500 shadow-xl shadow-blue-900/5' : 'bg-slate-100 dark:bg-slate-900/50 border-slate-200 dark:border-slate-800 opacity-70'}`}>
            
            <div className="flex items-center gap-4 mb-6">
               <div className={`p-4 rounded-xl ${nnunetResult ? 'bg-blue-100 text-blue-600 dark:bg-blue-900/30 dark:text-blue-400' : 'bg-slate-200 text-slate-500 dark:bg-slate-800'}`}>
                  <CpuChipIcon className="w-8 h-8" />
               </div>
               <div>
                  <h2 className="text-2xl font-bold">nnU-Net Pipeline</h2>
                  <p className="text-sm text-slate-500">Volumetric Segmentation</p>
               </div>
            </div>

            {nnunetResult ? (
               <div className="flex-1 flex flex-col">
                 {detectedNnUnet.length > 0 ? (
                   <div className="mb-6 flex-1">
                     <div className="flex gap-4 mb-4">
                       <div className="flex-1 bg-blue-50 dark:bg-blue-900/20 rounded-xl p-3 text-center border border-blue-100 dark:border-blue-800/40">
                         <p className="text-2xl font-bold text-blue-600 dark:text-blue-400">{nnunetResult.slice_images?.length || 0}</p>
                         <p className="text-xs text-slate-500 mt-0.5">Slices Detected</p>
                       </div>
                       <div className="flex-1 bg-blue-50 dark:bg-blue-900/20 rounded-xl p-3 text-center border border-blue-100 dark:border-blue-800/40">
                         <p className="text-2xl font-bold text-blue-600 dark:text-blue-400">{detectedNnUnet.length}</p>
                         <p className="text-xs text-slate-500 mt-0.5">Locations Detected</p>
                       </div>
                     </div>
                     <h4 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">Detected Locations</h4>
                     <div className="flex flex-wrap gap-1.5">
                       {detectedNnUnet.map((p, i) => (
                         <span key={i} className="bg-blue-100 text-blue-700 dark:bg-blue-900/40 dark:text-blue-300 px-2.5 py-1 rounded-lg text-xs font-medium border border-blue-200/50 dark:border-blue-800/50">
                           {p.location}
                         </span>
                       ))}
                     </div>
                   </div>
                 ) : (
                   <div className="mb-6 flex-1 flex flex-col items-center justify-center py-6">
                     <CheckCircleIcon className="w-10 h-10 text-emerald-400 mb-2" />
                     <p className="text-lg font-semibold text-emerald-600 dark:text-emerald-400">Not Detected</p>
                     <p className="text-sm text-slate-500 mt-1">No aneurysm regions were identified.</p>
                   </div>
                 )}
                 
                 <button onClick={navigateToNnunet} className="w-full btn-primary bg-blue-600 hover:bg-blue-700 text-white flex items-center justify-center gap-2 py-4 mt-auto">
                   View Detailed Report <ArrowRightIcon className="w-4 h-4" />
                 </button>
               </div>
            ) : (
               <div className="flex-1 flex flex-col items-center justify-center text-center py-8">
                  <span className="text-slate-500">Analysis did not complete.</span>
               </div>
            )}
          </motion.div>

          {/* ── MedGemma Card ── */}
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.2 }}
            className={`relative flex flex-col p-8 rounded-2xl border-2 transition-all ${gemmaResult ? 'bg-white dark:bg-slate-900 border-violet-200 dark:border-violet-900/50 hover:border-violet-400 dark:hover:border-violet-500 shadow-xl shadow-violet-900/5' : 'bg-slate-100 dark:bg-slate-900/50 border-slate-200 dark:border-slate-800 opacity-70'}`}>
            
            <div className="flex items-center gap-4 mb-6">
               <div className={`p-2 rounded-xl ${gemmaResult ? 'bg-violet-100 dark:bg-violet-900/30' : 'bg-slate-200 dark:bg-slate-800'}`}>
                  <img src="/medgemma-icon.png" alt="MedGemma" className="w-12 h-12 object-contain rounded-lg" />
               </div>
               <div>
                  <h2 className="text-2xl font-bold">Google MedGemma</h2>
                  <p className="text-sm text-slate-500">Visual Language Analysis</p>
               </div>
            </div>

            {gemmaResult ? (
               <div className="flex-1 flex flex-col">
                 {detectedGemma.length > 0 ? (
                   <div className="mb-6 flex-1">
                     <div className="flex gap-4 mb-4">
                       <div className="flex-1 bg-violet-50 dark:bg-violet-900/20 rounded-xl p-3 text-center border border-violet-100 dark:border-violet-800/40">
                         <p className="text-2xl font-bold text-violet-600 dark:text-violet-400">{gemmaResult.slices_analyzed || 0}</p>
                         <p className="text-xs text-slate-500 mt-0.5">Slices Detected</p>
                       </div>
                       <div className="flex-1 bg-violet-50 dark:bg-violet-900/20 rounded-xl p-3 text-center border border-violet-100 dark:border-violet-800/40">
                         <p className="text-2xl font-bold text-violet-600 dark:text-violet-400">{detectedGemma.length}</p>
                         <p className="text-xs text-slate-500 mt-0.5">Locations Detected</p>
                       </div>
                     </div>
                     <h4 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">Detected Locations</h4>
                     <div className="flex flex-wrap gap-1.5">
                       {detectedGemma.map(([locName], i) => (
                         <span key={i} className="bg-violet-100 text-violet-700 dark:bg-violet-900/40 dark:text-violet-300 px-2.5 py-1 rounded-lg text-xs font-medium border border-violet-200/50 dark:border-violet-800/50">
                           {locName}
                         </span>
                       ))}
                     </div>
                   </div>
                 ) : (
                   <div className="mb-6 flex-1 flex flex-col items-center justify-center py-6">
                     <CheckCircleIcon className="w-10 h-10 text-emerald-400 mb-2" />
                     <p className="text-lg font-semibold text-emerald-600 dark:text-emerald-400">Not Detected</p>
                     <p className="text-sm text-slate-500 mt-1">No aneurysm locations were identified.</p>
                   </div>
                 )}

                 <button onClick={navigateToMedGemma} className="w-full btn-primary bg-violet-600 hover:bg-violet-700 text-white flex items-center justify-center gap-2 py-4 mt-auto">
                   View Detailed Report <ArrowRightIcon className="w-4 h-4" />
                 </button>
               </div>
            ) : (
               <div className="flex-1 flex flex-col items-center justify-center text-center py-8">
                  <span className="text-slate-500">Analysis did not complete.</span>
               </div>
            )}
          </motion.div>

        </div>

        {/* Ground Truth Analysis Section */}
        {seriesGT && renderComparisonGraph()}

      </div>
    </div>
  )
}
