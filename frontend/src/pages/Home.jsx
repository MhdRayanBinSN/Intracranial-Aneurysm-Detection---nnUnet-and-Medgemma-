import { motion } from 'framer-motion'
import { Link } from 'react-router-dom'
import { DocumentTextIcon, ArrowUpTrayIcon } from '@heroicons/react/24/outline'

export default function Home() {
  return (
    <div className="container-fluid py-8">
      <div className="max-w-5xl mx-auto">
        
        {/* Header Section */}
        <header className="mb-12 border-b border-slate-200 dark:border-slate-800 pb-8">
          <h1 className="text-3xl font-bold text-slate-900 dark:text-slate-100 mb-4">
            Automated Detection of Intracranial Aneurysms from 3D Angiography
          </h1>
          <div className="flex flex-wrap gap-6 text-sm text-slate-600 dark:text-slate-400 font-mono">
            <span>Model: nnU-Net (ResEncM)</span>
            <span>Dataset: Medical CTA</span>
            <span>Accuracy: High Precision</span>
            <span>Framework: PyTorch</span>
          </div>
        </header>

        {/* Abstract & Upload Grid */}
        <div className="grid md:grid-cols-3 gap-8">
          
          {/* Left Column: Abstract (2/3 width) */}
          <div className="md:col-span-2 space-y-8">
            <section>
              <h2 className="text-sm font-bold text-slate-600 dark:text-slate-500 uppercase tracking-wider mb-3">Abstract</h2>
              <div className="prose prose-sm max-w-none text-slate-700 dark:prose-invert dark:text-slate-300 space-y-4 text-justify">
                <p>
                  Intracranial aneurysms are pathological dilations of cerebral arteries that pose a significant risk of rupture and subarachnoid hemorrhage. Early detection is critical for patient survival and management. This project implements a deep learning-based solution for the automated detection and segmentation of aneurysms from 3D Computed Tomography Angiography (CTA) scans.
                </p>
                <p>
                  The system utilizes the <strong>nnU-Net framework</strong>, a state-of-the-art medical image segmentation tool. It employs a Residual Encoder U-Net architecture trained on extensive 3D angiography datasets to identify aneurysms across 13 distinct anatomical locations, including the Internal Carotid Artery (ICA), Middle Cerebral Artery (MCA), and Anterior Communicating Artery (ACoA).
                </p>
                <p>
                  Our implementation integrates this pretrained model into a clinical workstation interface, enabling rapid inference, 3D slice visualization, and precise localization of potential anomalies.
                </p>
              </div>
            </section>

            <section className="grid grid-cols-2 gap-4">
              <div className="card p-4 border-l-4 border-l-blue-500">
                <div className="label-text mb-2">Architecture</div>
                <div className="font-mono text-sm text-slate-800 dark:text-slate-200">Residual Encoder U-Net</div>
              </div>
              <div className="card p-4 border-l-4 border-l-emerald-500">
                <div className="label-text mb-2">Input Modality</div>
                <div className="font-mono text-sm text-slate-800 dark:text-slate-200">3D CT Angiography (NIfTI)</div>
              </div>
            </section>
          </div>

          {/* Right Column: Key Actions */}
          <div className="space-y-6">
            <div className="card p-6 bg-slate-50 dark:bg-slate-900/50">
              <h3 className="text-lg font-semibold text-slate-900 dark:text-white mb-4 flex items-center gap-2">
                <ArrowUpTrayIcon className="w-5 h-5 text-blue-500 dark:text-blue-400" />
                Run Inference
              </h3>
              <p className="text-sm text-slate-600 dark:text-slate-400 mb-6">
                Upload a DICOM series or NIfTI file to generate a segmentation report.
              </p>
              <Link to="/analysis">
                <button className="w-full btn-primary py-3 flex items-center justify-center gap-2">
                  Initialize Analysis Module
                </button>
              </Link>
            </div>

            <div className="card p-6">
              <h3 className="text-lg font-semibold text-slate-900 dark:text-white mb-4 flex items-center gap-2">
                <DocumentTextIcon className="w-5 h-5 text-slate-500 dark:text-slate-400" />
                Documentation
              </h3>
              <ul className="space-y-3 text-sm text-slate-600 dark:text-slate-400">
                <li className="flex items-center gap-2">
                  <span className="w-1.5 h-1.5 bg-slate-400 dark:bg-slate-600 rounded-full"></span>
                  Model Architecture
                </li>
                <li className="flex items-center gap-2">
                  <span className="w-1.5 h-1.5 bg-slate-400 dark:bg-slate-600 rounded-full"></span>
                  Training Methodology
                </li>
                <li className="flex items-center gap-2">
                  <span className="w-1.5 h-1.5 bg-slate-400 dark:bg-slate-600 rounded-full"></span>
                  Evaluation Metrics
                </li>
              </ul>
            </div>
          </div>

        </div>
      </div>
    </div>
  )
}
