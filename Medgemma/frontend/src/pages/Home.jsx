import { motion } from 'framer-motion'
import { Link } from 'react-router-dom'
import { ArrowRightIcon, BeakerIcon, ShieldCheckIcon, BoltIcon } from '@heroicons/react/24/outline'

const features = [
  {
    icon: BeakerIcon,
    title: 'MedGemma AI Analysis',
    description: "Google's MedGemma 4B multimodal model analyzes your CT/MRI scans with expert neuroradiology prompting.",
  },
  {
    icon: BoltIcon,
    title: 'GPU-Accelerated',
    description: 'CUDA-powered inference with CLAHE preprocessing for crisp vessel visualisation in seconds.',
  },
  {
    icon: ShieldCheckIcon,
    title: '13-Location Detection',
    description: 'Saccular aneurysms checked across all 13 RSNA anatomical locations with bounding-box localisation.',
  },
]

const stats = [
  { label: 'Anatomical Locations', value: '13' },
  { label: 'Supported Formats',    value: 'DICOM & NIfTI' },
  { label: 'Processing Mode',      value: 'GPU' },
]

export default function Home() {
  return (
    <div className="relative overflow-hidden">

      {/* ── Hero ───────────────────────────────────────────────────────────── */}
      <section className="relative min-h-[90vh] flex items-center justify-center px-4">
        <div className="absolute inset-0 overflow-hidden">
          <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-primary-500/10 rounded-full blur-3xl animate-float" />
          <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-accent-500/10 rounded-full blur-3xl animate-float delay-1000" />
        </div>

        <div className="relative max-w-6xl mx-auto text-center">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
          >
            {/* Badge */}
            <div className="inline-flex items-center px-4 py-2 rounded-full bg-surface-800/50 border border-surface-700 mb-8">
              <span className="w-2 h-2 bg-risk-low rounded-full mr-2 animate-pulse" />
              <span className="text-sm text-surface-300">Powered by Google MedGemma 4B</span>
            </div>

            <h1 className="text-5xl md:text-7xl font-bold mb-6 leading-tight">
              Detect Intracranial{' '}
              <span className="gradient-text">Aneurysms</span>
              <br />
              with AI Precision
            </h1>

            <p className="text-xl md:text-2xl text-surface-400 max-w-2xl mx-auto mb-10">
              Upload DICOM or NIfTI brain scans. MedGemma analyses each slice,
              highlights suspicious regions, and produces a structured radiological report.
            </p>

            <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
              <Link to="/medgemma">
                <motion.button
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                  className="btn-primary px-8 py-4 text-lg flex items-center gap-2 glow-primary"
                >
                  Start MedGemma Analysis
                  <ArrowRightIcon className="w-5 h-5" />
                </motion.button>
              </Link>
            </div>
          </motion.div>

          {/* Stats */}
          <motion.div
            initial={{ opacity: 0, y: 50 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.3 }}
            className="flex flex-wrap justify-center gap-12 mt-16"
          >
            {stats.map((stat, i) => (
              <div key={i} className="text-center">
                <div className="text-3xl font-bold text-primary-400 mb-1">{stat.value}</div>
                <div className="text-sm text-surface-500">{stat.label}</div>
              </div>
            ))}
          </motion.div>
        </div>
      </section>

      {/* ── Features ───────────────────────────────────────────────────────── */}
      <section className="py-24 px-4">
        <div className="max-w-6xl mx-auto">
          <motion.div
            initial={{ opacity: 0 }}
            whileInView={{ opacity: 1 }}
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <h2 className="text-4xl font-bold mb-4">
              How It <span className="text-primary-400">Works</span>
            </h2>
            <p className="text-surface-400 max-w-xl mx-auto">
              A clinical-grade pipeline built on MedGemma with CLAHE preprocessing and aneurysm-specific detection logic.
            </p>
          </motion.div>

          <div className="grid md:grid-cols-3 gap-6">
            {features.map((feature, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, y: 30 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: i * 0.1 }}
                whileHover={{ y: -5 }}
                className="card p-8 group"
              >
                <div className="w-14 h-14 bg-primary-500/10 rounded-2xl flex items-center justify-center mb-6 group-hover:bg-primary-500/20 transition-colors">
                  <feature.icon className="w-7 h-7 text-primary-400" />
                </div>
                <h3 className="text-xl font-semibold mb-3">{feature.title}</h3>
                <p className="text-surface-400">{feature.description}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* ── 13 Locations ───────────────────────────────────────────────────── */}
      <section className="py-24 px-4 bg-surface-900/30">
        <div className="max-w-6xl mx-auto">
          <div className="grid md:grid-cols-2 gap-12 items-center">
            <motion.div
              initial={{ opacity: 0, x: -30 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }}
            >
              <h2 className="text-4xl font-bold mb-6">
                13 Anatomical <span className="text-accent-400">Locations</span>
              </h2>
              <p className="text-surface-400 mb-8">
                MedGemma checks all RSNA-specified saccular aneurysm locations:
              </p>
              <ul className="space-y-3">
                {[
                  'Left / Right Infraclinoid ICA',
                  'Left / Right Supraclinoid ICA',
                  'Left / Right Middle Cerebral Artery',
                  'Left / Right Anterior Cerebral Artery',
                  'Left / Right Posterior Communicating Artery',
                  'Basilar Tip',
                  'Other Posterior Circulation',
                ].map((loc, i) => (
                  <li key={i} className="flex items-center gap-3 text-surface-300">
                    <div className="w-2 h-2 bg-primary-500 rounded-full flex-shrink-0" />
                    {loc}
                  </li>
                ))}
              </ul>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, x: 30 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }}
            >
              <div className="aspect-square bg-gradient-to-br from-surface-800 to-surface-900 rounded-3xl flex items-center justify-center border border-surface-700">
                <div className="text-center p-8">
                  <div className="w-32 h-32 mx-auto mb-6 bg-primary-500/20 rounded-full flex items-center justify-center">
                    <svg className="w-16 h-16 text-primary-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
                        d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"
                      />
                    </svg>
                  </div>
                  <p className="text-surface-400">Circle of Willis coverage</p>
                </div>
              </div>
            </motion.div>
          </div>
        </div>
      </section>

      {/* ── CTA ────────────────────────────────────────────────────────────── */}
      <section className="py-24 px-4">
        <div className="max-w-4xl mx-auto text-center">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="card p-12 relative overflow-hidden"
          >
            <div className="absolute inset-0 bg-gradient-to-r from-primary-500/10 to-accent-500/10" />
            <div className="relative">
              <h2 className="text-4xl font-bold mb-4">Ready to Analyse?</h2>
              <p className="text-surface-400 mb-8 max-w-lg mx-auto">
                Upload your DICOM or NIfTI files and let MedGemma produce a
                structured aneurysm detection report in seconds.
              </p>
              <Link to="/medgemma">
                <motion.button
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                  className="btn-primary px-8 py-4 text-lg"
                >
                  Open MedGemma Analyser
                </motion.button>
              </Link>
            </div>
          </motion.div>
        </div>
      </section>

      {/* ── Footer ─────────────────────────────────────────────────────────── */}
      <footer className="border-t border-surface-800 py-8 px-4">
        <div className="max-w-6xl mx-auto flex flex-col md:flex-row items-center justify-between gap-4">
          <div className="text-surface-500 text-sm">
            © 2024 AneurysmAI · RSNA Intracranial Aneurysm Detection
          </div>
          <div className="text-surface-500 text-sm">
            Powered by Google MedGemma 4B · PyTorch · FastAPI
          </div>
        </div>
      </footer>
    </div>
  )
}
