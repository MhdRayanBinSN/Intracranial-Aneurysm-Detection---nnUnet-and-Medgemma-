import { useState, useEffect } from 'react'
import { Link, useLocation } from 'react-router-dom'
import { motion } from 'framer-motion'
import { Bars3Icon, XMarkIcon, SunIcon, MoonIcon } from '@heroicons/react/24/outline'
import clsx from 'clsx'
import { useTheme } from '../context/ThemeContext'

const navItems = [
  { name: 'Project Overview', path: '/' },
  { name: 'Architecture', path: '/architecture' },
  { name: 'Dataset', path: '/dataset' },
  { name: 'New Analysis', path: '/analysis' },
]

export default function Navbar() {
  const [isScrolled, setIsScrolled] = useState(false)
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false)
  const location = useLocation()
  const { theme, toggleTheme } = useTheme()
  const [sysStatus, setSysStatus] = useState('CHECKING')
  const [isOnline, setIsOnline] = useState(false)

  useEffect(() => {
    const checkStatus = async () => {
      try {
        const res = await fetch('http://localhost:8001/health')
        const data = await res.json()
        if (data.model_loaded) {
          setSysStatus('GPU ACTIVE')
          setIsOnline(true)
        } else {
          setSysStatus('CPU READY')
          setIsOnline(true)
        }
      } catch (err) {
        setSysStatus('OFFLINE')
        setIsOnline(false)
      }
    }
    checkStatus()
    const interval = setInterval(checkStatus, 15000)
    return () => clearInterval(interval)
  }, [])

  useEffect(() => {
    const handleScroll = () => {
      setIsScrolled(window.scrollY > 10)
    }
    window.addEventListener('scroll', handleScroll)
    return () => window.removeEventListener('scroll', handleScroll)
  }, [])

  return (
    <nav
      className={clsx(
        'fixed top-0 left-0 right-0 z-50 transition-all duration-300 border-b',
        isScrolled
          ? 'bg-white/95 dark:bg-slate-950/95 backdrop-blur-md shadow-sm border-slate-200 dark:border-white/5'
          : 'bg-white dark:bg-slate-950 border-transparent dark:border-transparent'
      )}
    >
      <div className="container-fluid">
        <div className="flex items-center justify-between h-16">
          {/* Logo / Project Title */}
          <Link to="/" className="flex items-center space-x-3 group">
              <div className="flex flex-col">
                <span className="text-sm font-bold text-slate-900 dark:text-slate-100 uppercase tracking-wide">
                  Intracranial Aneurysm Detection
                </span>
                <span className="text-xs text-slate-500 dark:text-slate-400 font-mono">
                  AI-Powered Medical Imaging
                </span>
              </div>
            </Link>

          {/* Desktop Navigation */}
          <div className="hidden md:flex items-center space-x-1">
            {navItems.map((item) => (
              <Link
                key={item.path}
                to={item.path}
                className={clsx(
                  'px-4 py-2 text-sm font-medium transition-colors duration-200 border-b-2',
                  location.pathname === item.path
                    ? 'border-blue-600 text-blue-600 dark:border-blue-500 dark:text-blue-400'
                    : 'border-transparent text-slate-600 hover:text-slate-900 dark:text-slate-400 dark:hover:text-slate-200'
                )}
              >
                {item.name}
              </Link>
            ))}
          </div>

          {/* Metadata / CTA */}
          <div className="hidden md:flex items-center gap-4">
             <div className="text-right hidden lg:block">
                <div className="text-xs text-slate-500 uppercase">System Status</div>
                <div className="flex items-center gap-2 justify-end">
                    <span className={`w-2 h-2 rounded-full ${isOnline ? 'bg-emerald-500 animate-pulse' : 'bg-red-500'}`}></span>
                    <span className={`text-xs font-mono w-24 text-right ${isOnline ? 'text-emerald-600 dark:text-emerald-400' : 'text-red-500'}`}>{sysStatus}</span>
                </div>
             </div>
             <div className="h-8 w-px bg-slate-300 dark:bg-slate-800 mx-2"></div>
             
             {/* Theme Toggle */}
             <button
               onClick={toggleTheme}
               className="p-2 rounded-full text-slate-500 hover:bg-slate-100 dark:text-slate-400 dark:hover:bg-slate-800 transition-colors"
               aria-label="Toggle theme"
             >
               {theme === 'light' ? (
                 <MoonIcon className="w-5 h-5" />
               ) : (
                 <SunIcon className="w-5 h-5" />
               )}
             </button>

             <Link to="/analysis">
              <button className="btn-primary">
                + New Scan
              </button>
            </Link>
          </div>

          {/* Mobile Menu Button */}
          <button
            onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
            className="md:hidden p-2 rounded text-slate-600 hover:text-slate-900 hover:bg-slate-100 dark:text-slate-400 dark:hover:text-white dark:hover:bg-slate-800"
          >
            {isMobileMenuOpen ? (
              <XMarkIcon className="w-6 h-6" />
            ) : (
              <Bars3Icon className="w-6 h-6" />
            )}
          </button>
        </div>
      </div>

      {/* Mobile Menu */}
      <motion.div
        initial={false}
        animate={{
          height: isMobileMenuOpen ? 'auto' : 0,
          opacity: isMobileMenuOpen ? 1 : 0,
        }}
        className="md:hidden overflow-hidden bg-slate-50 dark:bg-slate-900 border-b border-slate-200 dark:border-slate-800"
      >
        <div className="px-4 py-4 space-y-2">
          {navItems.map((item) => (
            <Link
              key={item.path}
              to={item.path}
              onClick={() => setIsMobileMenuOpen(false)}
              className={clsx(
                'block px-4 py-3 rounded text-sm font-medium',
                location.pathname === item.path
                  ? 'bg-slate-200 text-blue-600 dark:bg-slate-800 dark:text-blue-400'
                  : 'text-slate-600 hover:bg-slate-100 hover:text-slate-900 dark:text-slate-400 dark:hover:text-slate-200 dark:hover:bg-slate-800'
              )}
            >
              {item.name}
            </Link>
          ))}
        </div>
      </motion.div>
    </nav>
  )
}



