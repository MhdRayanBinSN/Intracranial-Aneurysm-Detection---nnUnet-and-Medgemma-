import { useEffect, useState } from 'react'
import { motion } from 'framer-motion'
import {
  CircleStackIcon,
  UsersIcon,
  BeakerIcon,
  ChartBarIcon,
  CheckCircleIcon,
  ExclamationTriangleIcon,
  ClockIcon,
  MagnifyingGlassIcon,
  ServerStackIcon,
  CpuChipIcon,
} from '@heroicons/react/24/outline'
import api from '../api/client'

// ── Hardcoded fallback (shows even if backend is offline) ──────────────────
const FALLBACK = {
  overview: { total_series: 4348, positive_cases: 1864, negative_cases: 2484, positive_pct: 42.9, negative_pct: 57.1, total_locations: 13 },
  modality: { CTA: 1808, MRA: 1252, 'MRI T2': 983, 'MRI T1post': 305 },
  sex: { Female: 3005, Male: 1343 },
  age: {
    mean: 58.5, median: 59.0, min: 18, max: 89,
    buckets: [
      { label: '<30', count: 39 }, { label: '30-39', count: 217 }, { label: '40-49', count: 614 },
      { label: '50-59', count: 1165 }, { label: '60-69', count: 1321 }, { label: '70-79', count: 762 }, { label: '80+', count: 230 }
    ]
  },
  locations: [
    { name: 'Anterior Communicating Artery',              count: 363, pct: 19.5 },
    { name: 'Left Supraclinoid Internal Carotid Artery',  count: 331, pct: 17.8 },
    { name: 'Right Middle Cerebral Artery',               count: 294, pct: 15.8 },
    { name: 'Right Supraclinoid Internal Carotid Artery', count: 277, pct: 14.9 },
    { name: 'Left Middle Cerebral Artery',                count: 219, pct: 11.8 },
    { name: 'Other Posterior Circulation',                count: 113, pct:  6.1 },
    { name: 'Basilar Tip',                                count: 110, pct:  5.9 },
    { name: 'Right Posterior Communicating Artery',       count: 101, pct:  5.4 },
    { name: 'Right Infraclinoid Internal Carotid Artery', count:  98, pct:  5.3 },
    { name: 'Left Posterior Communicating Artery',        count:  86, pct:  4.6 },
    { name: 'Left Infraclinoid Internal Carotid Artery',  count:  78, pct:  4.2 },
    { name: 'Right Anterior Cerebral Artery',             count:  56, pct:  3.0 },
    { name: 'Left Anterior Cerebral Artery',              count:  46, pct:  2.5 },
  ],
  files: { zip_size_gb: 214.86 },
  competition: {
    name: 'RSNA 2025 Intracranial Aneurysm Detection',
    host: 'Radiological Society of North America (RSNA)',
    year: 2025, platform: 'Kaggle',
    task: 'Multi-label binary classification across 13 anatomical locations',
  }
}

// ── Sub-components ─────────────────────────────────────────────────────────

function StatCard({ icon: Icon, label, value, sub, color = 'text-blue-500 dark:text-blue-400', delay = 0 }) {
  return (
    <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay }}
      className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl p-5 hover:border-slate-400 dark:hover:border-slate-600 transition-colors"
    >
      <div className="w-9 h-9 rounded-lg bg-slate-100 dark:bg-slate-800 flex items-center justify-center mb-3">
        <Icon className={`w-5 h-5 ${color}`} />
      </div>
      <p className={`text-2xl font-bold font-mono ${color}`}>{value}</p>
      <p className="text-sm font-medium text-slate-700 dark:text-slate-300 mt-1">{label}</p>
      {sub && <p className="text-xs text-slate-500 mt-0.5">{sub}</p>}
    </motion.div>
  )
}

function HBar({ name, count, maxVal, pct, color }) {
  return (
    <div>
      <div className="flex items-center justify-between text-xs mb-1">
        <span className="text-slate-700 dark:text-slate-300 truncate max-w-[65%]">{name}</span>
        <span className="text-slate-600 dark:text-slate-400 font-mono">{count.toLocaleString()}{pct != null ? ` (${pct}%)` : ''}</span>
      </div>
      <div className="h-2 bg-slate-200 dark:bg-slate-800 rounded-full overflow-hidden">
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${(count / maxVal) * 100}%` }}
          transition={{ duration: 0.8, ease: 'easeOut' }}
          className={`h-full rounded-full ${color}`}
        />
      </div>
    </div>
  )
}

function DonutRing({ pct, label, color }) {
  const r = 34, circ = 2 * Math.PI * r, dash = (pct / 100) * circ
  return (
    <div className="flex flex-col items-center gap-2">
      <svg width="88" height="88" viewBox="0 0 88 88">
        <circle cx="44" cy="44" r={r} fill="none" className="stroke-slate-200 dark:stroke-slate-800" strokeWidth="9" />
        <motion.circle cx="44" cy="44" r={r} fill="none" stroke={color} strokeWidth="9"
          strokeLinecap="round" strokeDasharray={circ}
          initial={{ strokeDashoffset: circ }}
          animate={{ strokeDashoffset: circ - dash }}
          transition={{ duration: 1.2, ease: 'easeOut' }}
          style={{ transformOrigin: '44px 44px', transform: 'rotate(-90deg)' }}
        />
        <text x="44" y="49" textAnchor="middle" fontSize="13" fontWeight="bold" className="fill-slate-900 dark:fill-white" fontFamily="monospace">{pct}%</text>
      </svg>
      <p className="text-xs text-slate-500 dark:text-slate-400 text-center leading-snug whitespace-pre-line">{label}</p>
    </div>
  )
}

// ── Main page ──────────────────────────────────────────────────────────────
export default function Dataset() {
  const [data, setData]   = useState(FALLBACK)
  const [live, setLive]   = useState(false)

  useEffect(() => {
    api.get('/dataset/info')
      .then(r => { if (r.data?.available) { setData(r.data); setLive(true) } })
      .catch(() => {})
  }, [])

  const { overview, modality, sex, age, locations, files, competition } = data
  const maxLoc    = Math.max(...locations.map(l => l.count))
  const maxBucket = Math.max(...age.buckets.map(b => b.count))
  const modList   = Object.entries(modality).sort((a, b) => b[1] - a[1])
  const maxMod    = modList[0]?.[1] || 1

  return (
    <div className="min-h-screen py-12 px-4 bg-slate-50 text-slate-900 dark:bg-slate-950 dark:text-white">
      <div className="max-w-7xl mx-auto space-y-8">

        {/* ── Header ── */}
        <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} className="text-center">
          <div className="inline-flex items-center gap-3 mb-4">
            <div className="w-11 h-11 bg-blue-100 border border-blue-200 dark:bg-blue-600/20 dark:border-blue-500/30 rounded-lg flex items-center justify-center">
              <CircleStackIcon className="w-6 h-6 text-blue-600 dark:text-blue-400" />
            </div>
            <h1 className="text-3xl font-bold uppercase tracking-wide">
              Dataset <span className="text-blue-600 dark:text-blue-400">Information</span>
            </h1>
          </div>
          <p className="text-slate-600 dark:text-slate-400 max-w-xl mx-auto text-sm">
            Statistics for the <strong className="text-slate-900 dark:text-slate-200">RSNA 2023 Intracranial Aneurysm Detection</strong> dataset
            used to train and evaluate our nnU-Net detection model.
          </p>
        </motion.div>

        

        {/* ── Key Stats Row 1 ── */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <StatCard icon={CircleStackIcon}       label="Total Series"       value={overview.total_series.toLocaleString()} sub="Scan series"                  color="text-blue-600 dark:text-blue-400"   delay={0.08} />
          <StatCard icon={ExclamationTriangleIcon} label="Positive Cases"   value={overview.positive_cases.toLocaleString()} sub={`${overview.positive_pct}% of dataset`} color="text-red-600 dark:text-red-400"    delay={0.12} />
          <StatCard icon={CheckCircleIcon}        label="Negative Cases"    value={overview.negative_cases.toLocaleString()} sub={`${overview.negative_pct}% of dataset`} color="text-emerald-600 dark:text-emerald-400" delay={0.16} />
          <StatCard icon={ServerStackIcon}        label="Dataset Size"      value={`${files.zip_size_gb} GB`} sub="Raw ZIP archive"              color="text-cyan-600 dark:text-cyan-400"   delay={0.20} />
        </div>

        {/* ── Key Stats Row 2 ── */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <StatCard icon={UsersIcon}    label="Mean Patient Age"        value={`${age.mean} yrs`}  sub={`Range ${age.min}–${age.max}`}  color="text-amber-600 dark:text-amber-400"  delay={0.08} />
          <StatCard icon={BeakerIcon}   label="Imaging Modalities"      value={Object.keys(modality).length} sub="CTA · MRA · MRI"        color="text-violet-600 dark:text-violet-400" delay={0.12} />
          <StatCard icon={MagnifyingGlassIcon} label="Anatomical Zones" value={overview.total_locations} sub="RSNA standard locations"   color="text-sky-600 dark:text-sky-400"    delay={0.16} />
          <StatCard icon={CpuChipIcon}  label="Classification Labels"   value={overview.total_locations + 1} sub="13 locations + overall" color="text-pink-600 dark:text-pink-400"   delay={0.20} />
        </div>

        {/* ── Donut Row ── */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-5">

          {/* Class Balance */}
          <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.25 }}
            className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl p-6"
          >
            <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-400 flex items-center gap-2 mb-6">
              <ChartBarIcon className="w-4 h-4 text-blue-500 dark:text-blue-400" /> Class Balance
            </h2>
            <div className="flex justify-around">
              <DonutRing pct={overview.positive_pct} label={`Positive\n${overview.positive_cases.toLocaleString()} series`} color="#f87171" />
              <DonutRing pct={overview.negative_pct} label={`Negative\n${overview.negative_cases.toLocaleString()} series`} color="#34d399" />
            </div>
            <p className="text-xs text-slate-500 text-center mt-5">
              ⚠ Moderate imbalance — Focal Loss used during training
            </p>
          </motion.div>

          {/* Sex Distribution */}
          <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.28 }}
            className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl p-6"
          >
            <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-400 flex items-center gap-2 mb-6">
              <UsersIcon className="w-4 h-4 text-violet-500 dark:text-violet-400" /> Sex Distribution
            </h2>
            <div className="flex justify-around">
              {Object.entries(sex).map(([s, count]) => (
                <DonutRing
                  key={s}
                  pct={Math.round(count / overview.total_series * 100)}
                  label={`${s}\n${count.toLocaleString()} patients`}
                  color={s === 'Female' ? '#a78bfa' : '#60a5fa'}
                />
              ))}
            </div>
            <p className="text-xs text-slate-500 text-center mt-5">
              Female-predominant (consistent with IA epidemiology)
            </p>
          </motion.div>

          {/* Modality */}
          <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.31 }}
            className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl p-6"
          >
            <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-400 flex items-center gap-2 mb-5">
              <BeakerIcon className="w-4 h-4 text-cyan-500 dark:text-cyan-400" /> Imaging Modalities
            </h2>
            <div className="space-y-3">
              {modList.map(([name, count]) => (
                <HBar key={name} name={name} count={count}
                  maxVal={maxMod} pct={Math.round(count / overview.total_series * 100)}
                  color="bg-gradient-to-r from-cyan-600 to-blue-500"
                />
              ))}
            </div>
          </motion.div>
        </div>

        {/* ── Age Distribution ── */}
        <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.33 }}
          className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl p-6"
        >
          <div className="flex items-center justify-between flex-wrap gap-3 mb-6">
            <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-400 flex items-center gap-2">
              <ClockIcon className="w-4 h-4 text-amber-500 dark:text-amber-400" /> Age Distribution
            </h2>
            <div className="flex gap-5 text-xs font-mono">
              <span className="text-slate-500">Mean <span className="text-slate-900 dark:text-white">{age.mean}</span></span>
              <span className="text-slate-500">Median <span className="text-slate-900 dark:text-white">{age.median}</span></span>
              <span className="text-slate-500">Range <span className="text-slate-900 dark:text-white">{age.min}–{age.max}</span></span>
            </div>
          </div>
          <div className="flex items-end gap-2 h-36">
            {age.buckets.map((b, i) => {
              const h = Math.round((b.count / maxBucket) * 100)
              return (
                <div key={i} className="flex-1 flex flex-col items-center gap-1">
                  <p className="text-xs text-slate-500 font-mono">{b.count.toLocaleString()}</p>
                  <motion.div
                    initial={{ height: 0 }}
                    animate={{ height: `${h}%` }}
                    transition={{ duration: 0.8, delay: i * 0.07, ease: 'easeOut' }}
                    className="w-full rounded-t bg-gradient-to-t from-amber-700 to-amber-400 min-h-[3px]"
                    style={{ maxHeight: '100px' }}
                  />
                  <p className="text-xs text-slate-500">{b.label}</p>
                </div>
              )
            })}
          </div>
        </motion.div>

        {/* ── Location Distribution ── */}
        <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.36 }}
          className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl p-6"
        >
          <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-400 flex items-center gap-2 mb-2">
            <MagnifyingGlassIcon className="w-4 h-4 text-red-500 dark:text-red-400" /> Aneurysm Location Distribution
          </h2>
          <p className="text-xs text-slate-500 mb-5">
            Positive cases per anatomical zone among {overview.positive_cases.toLocaleString()} positive series.
            A series may have multiple aneurysm locations.
          </p>
          <div className="space-y-3">
            {locations.map((l) => (
              <HBar key={l.name} name={l.name} count={l.count} maxVal={maxLoc} pct={l.pct}
                color="bg-gradient-to-r from-red-700 to-red-500"
              />
            ))}
          </div>
        </motion.div>

        {/* ── Dataset Files ── */}
        <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.39 }}
          className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl p-6"
        >
          <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-400 flex items-center gap-2 mb-4">
            <ServerStackIcon className="w-4 h-4 text-cyan-500 dark:text-cyan-400" /> Dataset Files
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {[
              { label: 'Main Archive', path: 'rsna-intracranial-aneurysm-detection.zip', info: `${files.zip_size_gb} GB · ZIP archive`, color: 'text-cyan-400' },
              { label: 'Labels CSV',  path: 'train.csv',                                 info: '4,348 rows × 18 columns',               color: 'text-emerald-400' },
              { label: 'Root Path',   path: 'C:\\Users\\Rayan\\Desktop\\Main Project\\', info: 'Local storage',                         color: 'text-slate-400' },
            ].map((f, i) => (
              <div key={i} className="bg-slate-50 dark:bg-slate-800/50 rounded-lg p-4 border border-slate-200 dark:border-slate-700">
                <p className="text-xs text-slate-500 uppercase tracking-wider mb-1">{f.label}</p>
                <p className={`text-sm font-mono break-all ${f.color}`}>{f.path}</p>
                <p className="text-xs text-slate-500 mt-1">{f.info}</p>
              </div>
            ))}
          </div>
        </motion.div>

        {/* ── Disclaimer ── */}
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.42 }}
          className="border border-slate-200 dark:border-slate-800 rounded-xl p-4 bg-slate-100 dark:bg-slate-900/40"
        >
          <p className="text-xs text-slate-500 text-center">
            Data sourced from the <strong className="text-slate-400">RSNA 2023 Intracranial Aneurysm Detection</strong> Kaggle competition.
            Used solely for academic and research purposes. All patient data is de-identified per competition guidelines.
          </p>
        </motion.div>

      </div>
    </div>
  )
}
