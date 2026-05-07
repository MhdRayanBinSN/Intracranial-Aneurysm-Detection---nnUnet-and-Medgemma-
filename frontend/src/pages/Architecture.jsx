import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Activity, Server, Brain, Layers, ArrowRight, CheckCircle, Workflow, FileText, BarChart3, Database, Shield, Zap, Globe, Cpu, Code } from 'lucide-react';

const Architecture = () => {
    const [selectedModule, setSelectedModule] = useState(null);

    const modules = {
        frontend: {
            id: 'frontend',
            title: 'Frontend Interface',
            icon: <Layers className="w-8 h-8 text-blue-400" />,
            color: 'blue',
            description: 'User-facing React application for secure DICOM uploads, real-time analysis tracking, and interactive 3D result visualization.',
            features: [
                'Drag-and-Drop DICOM Upload Manager',
                'WebSocket Client (Live Progress 0-100%)',
                '3D Slice Navigation & Heatmap Overlay',
                'Responsive Dashboard (TailwindCSS)',
                'Risk Assessment Badges (High/Mod/Low)'
            ],
            tech: ['React 18', 'Vite', 'TailwindCSS', 'Framer Motion', 'Axios']
        },
        backend: {
            id: 'backend',
            title: 'Backend API Gateway',
            icon: <Server className="w-8 h-8 text-green-400" />,
            color: 'green',
            description: 'High-performance FastAPI server managing asynchronous inference tasks, websocket connections, and secure file handling.',
            features: [
                'Async Request Handling (Uvicorn)',
                'Concurrent WebSocket Connections',
                'Temporary File Lifecycle Management',
                'CORS & Security Middleware',
                'JSON Response Formatting'
            ],
            tech: ['Python 3.10', 'FastAPI', 'Uvicorn', 'WebSockets', 'Pydantic']
        },
        ml_engine: {
            id: 'ml_engine',
            title: 'Deep Learning Inference Engine',
            icon: <Brain className="w-8 h-8 text-purple-400" />,
            color: 'purple',
            description: 'The core intelligence unit. An ensemble of 3D U-Net models trained on the RSNA 2025 dataset to detect intracranial aneurysms with high precision.',
            // High-detail pipeline for deep technical insight
            pipeline: [
                {
                    stage: '1. Data Ingestion & Volumetric Loading',
                    icon: <FileText className="w-5 h-5" />,
                    details: 'Parallel loading of raw DICOM series. Slices are sorted, oriented to RAS coordinate system, and stacked into 3D volumes.',
                    specs: [
                        { label: 'Input Format', value: 'DICOM Series / NIfTI' },
                        { label: 'Parallelism', value: 'Joblib (freq=cpu_count)' },
                        { label: 'Orientation', value: 'RAS (Right-Anterior-Superior)' },
                        { label: 'Library', value: 'SimpleITK + Pydicom' }
                    ]
                },
                {
                    stage: '2. Adaptive Preprocessing',
                    icon: <Workflow className="w-5 h-5" />,
                    details: 'Region-of-Interest (RoI) localization followed by intensity normalization specific to CTA Hounsfield Units.',
                    specs: [
                        { label: 'RoI Crop Size', value: '200mm (Z) × 160mm (Y) × 160mm (X)' },
                        { label: 'Resampling', value: 'Target Spacing from plans.json' },
                        { label: 'Normalization', value: 'Intensity Clipping → Z-score' },
                        { label: 'Output Shape', value: '(C=1, Z, Y, X) NumPy Array' }
                    ]
                },
                {
                    stage: '3. 3D Residual U-Net Inference',
                    icon: <Brain className="w-5 h-5" />,
                    details: 'Deep learning inference using the nnU-Net architecture with sliding window prediction to handle large volumes.',
                    specs: [
                        { label: 'Architecture', value: 'nnUNetResEncUNet M (3D Full-Res)' },
                        { label: 'Inference Mode', value: 'Sliding Window (50% Overlap)' },
                        { label: 'Precision', value: 'FP16 (Mixed Precision)' },
                        { label: 'Patch Size', value: '(128, 128, 128) Voxels' }
                    ]
                },
                {
                    stage: '4. Post-Processing & Probability Mapping',
                    icon: <Activity className="w-5 h-5" />,
                    details: 'Converting logits to probabilities, correcting known anatomical laterality errors, and extracting peak coordinates.',
                    specs: [
                        { label: 'Activation', value: 'Sigmoid (Multi-label Independent)' },
                        { label: 'Correction', value: 'L/R Anatomical Swap Fix' },
                        { label: 'Threshold', value: '> 0.5 (Binary Detection)' },
                        { label: 'Output', value: '13-Channel Probability Map' }
                    ]
                },
                {
                    stage: '5. Evaluation & Performance Metrics',
                    icon: <BarChart3 className="w-5 h-5" />,
                    details: 'Rigorous validation treating detection as a 3D segmentation task. Performance is measured against voxel-level ground truth.',
                    specs: [
                        { label: 'Primary Metric', value: 'Dice Similarity Coefficient (DSC)' },
                        { label: 'Overlap Metric', value: 'Intersection over Union (IoU)' },
                        { label: 'Error Metric', value: 'False Positive Rate (FPR)' },
                        { label: 'Validation', value: '5-Fold Cross-Validation' }
                    ]
                }
            ],
            tech: ['PyTorch', 'nnU-Net v2', 'SimpleITK', 'NumPy', 'SciPy']
        }
    };

    return (
        <div className="min-h-screen bg-slate-50 text-slate-900 dark:bg-gray-900 dark:text-white p-8 pt-24">
            <div className="max-w-7xl mx-auto">
                <div className="text-center mb-16">
                    <h1 className="text-4xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-500 via-purple-500 to-pink-500 dark:from-blue-400 dark:via-purple-500 dark:to-pink-500 mb-4">
                        System Architecture & Pipeline
                    </h1>
                    <p className="text-slate-600 dark:text-gray-400 max-w-2xl mx-auto">
                        Explore the end-to-end flow from DICOM ingestion to AI-powered diagnosis. Click on a module to inspect its technical specifications.
                    </p>
                </div>

                {/* Interactive Diagram */}
                <div className="relative h-[600px] bg-white border-slate-200 dark:bg-gray-800/30 rounded-3xl border dark:border-gray-700/50 backdrop-blur-xl overflow-hidden mb-12">
                    <div className="absolute inset-0 grid grid-cols-3 gap-8 p-12 items-center">

                        {/* Frontend Node */}
                        <ModuleNode
                            module={modules.frontend}
                            isSelected={selectedModule?.id === 'frontend'}
                            onClick={() => setSelectedModule(modules.frontend)}
                            position="left"
                        />

                        {/* Connection 1 */}
                        <ConnectionArrow active={true} />

                        {/* Backend Node */}
                        <ModuleNode
                            module={modules.backend}
                            isSelected={selectedModule?.id === 'backend'}
                            onClick={() => setSelectedModule(modules.backend)}
                            position="center"
                        />

                        {/* Connection 2 */}
                        <ConnectionArrow active={true} />

                        {/* ML Engine Node */}
                        <ModuleNode
                            module={modules.ml_engine}
                            isSelected={selectedModule?.id === 'ml_engine'}
                            onClick={() => setSelectedModule(modules.ml_engine)}
                            position="right"
                        />
                    </div>

                    {/* Background Grid Animation */}
                    <div className="absolute inset-0 bg-[linear-gradient(rgba(255,255,255,0.02)_1px,transparent_1px),linear-gradient(90deg,rgba(255,255,255,0.02)_1px,transparent_1px)] bg-[size:40px_40px] [mask-image:radial-gradient(ellipse_60%_60%_at_50%_50%,#000_70%,transparent_100%)] pointer-events-none" />
                </div>

                {/* Detail Panel */}
                <AnimatePresence mode="wait">
                    {selectedModule && (
                        <motion.div
                            key={selectedModule.id}
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            exit={{ opacity: 0, y: -20 }}
                            className="bg-white dark:bg-gray-800/50 backdrop-blur-md rounded-3xl border border-slate-200 dark:border-gray-700/50 overflow-hidden mb-24"
                        >
                            <div className={`h-2 w-full bg-gradient-to-r from-${selectedModule.color}-500 to-${selectedModule.color}-700`} />
                            
                            <div className="p-8 md:p-12">
                                {/* Header */}
                                <div className="flex flex-col md:flex-row gap-8 mb-12">
                                    <div className={`p-6 rounded-2xl bg-${selectedModule.color}-500/10 border border-${selectedModule.color}-500/20 w-fit h-fit`}>
                                        {selectedModule.icon}
                                    </div>
                                    <div>
                                        <h2 className="text-3xl font-bold mb-4">{selectedModule.title}</h2>
                                        <p className="text-slate-600 dark:text-gray-300 text-lg leading-relaxed max-w-3xl">
                                            {selectedModule.description}
                                        </p>
                                        
                                        {/* Tech Stack Chips */}
                                        <div className="flex flex-wrap gap-3 mt-6">
                                            {selectedModule.tech.map((tech, i) => (
                                                <span key={i} className={`px-3 py-1 rounded-full text-xs font-semibold bg-${selectedModule.color}-500/10 text-${selectedModule.color}-300 border border-${selectedModule.color}-500/20`}>
                                                    {tech}
                                                </span>
                                            ))}
                                        </div>
                                    </div>
                                </div>

                                {/* Content Section: Pipeline vs Feature List */}
                                {selectedModule.id === 'ml_engine' ? (
                                    <div className="space-y-8">
                                        <h3 className="text-xl font-bold flex items-center gap-2">
                                            <Workflow className="w-5 h-5 text-purple-400" />
                                            End-to-End ML Pipeline Execution
                                        </h3>
                                        <div className="grid gap-6">
                                            {selectedModule.pipeline.map((step, idx) => (
                                                <motion.div 
                                                    key={idx}
                                                    initial={{ opacity: 0, x: -20 }}
                                                    animate={{ opacity: 1, x: 0 }}
                                                    transition={{ delay: idx * 0.1 }}
                                                    className="relative pl-8 border-l-2 border-purple-500/20 group hover:border-purple-500/50 transition-colors"
                                                >
                                                    <div className="absolute -left-[9px] top-0 w-4 h-4 rounded-full bg-white dark:bg-gray-900 border-2 border-purple-500 group-hover:bg-purple-500 transition-colors" />
                                                    
                                                    <div className="bg-slate-50 border-slate-200 dark:bg-gray-900/50 rounded-xl p-6 border dark:border-gray-700/50 hover:border-purple-500/30 transition-all">
                                                        <div className="flex items-start justify-between mb-4">
                                                            <div>
                                                                <h4 className="text-lg font-semibold text-purple-600 dark:text-purple-300 mb-1">{step.stage}</h4>
                                                                <p className="text-slate-600 dark:text-gray-400 text-sm">{step.details}</p>
                                                            </div>
                                                            <div className="p-2 bg-purple-500/10 rounded-lg text-purple-400">
                                                                {step.icon}
                                                            </div>
                                                        </div>
                                                        
                                                        {/* Technical Specs Grid */}
                                                        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 bg-slate-100 dark:bg-black/20 rounded-lg p-4">
                                                            {step.specs.map((spec, sIdx) => (
                                                                <div key={sIdx}>
                                                                    <div className="text-xs text-slate-500 dark:text-gray-500 uppercase font-mono mb-1">{spec.label}</div>
                                                                    <div className="text-sm font-medium text-slate-900 dark:text-gray-200">{spec.value}</div>
                                                                </div>
                                                            ))}
                                                        </div>
                                                    </div>
                                                </motion.div>
                                            ))}
                                        </div>
                                    </div>
                                ) : (
                                    <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                                        <div>
                                            <h3 className="text-xl font-semibold mb-6 flex items-center gap-2">
                                                <Activity className="w-5 h-5 text-slate-400 dark:text-gray-400" />
                                                Key Functionalities
                                            </h3>
                                            <ul className="space-y-4">
                                                {selectedModule.features.map((feature, i) => (
                                                    <li key={i} className="flex items-start gap-3 text-slate-700 dark:text-gray-300">
                                                        <CheckCircle className={`w-5 h-5 text-${selectedModule.color}-500 shrink-0 mt-0.5`} />
                                                        <span>{feature}</span>
                                                    </li>
                                                ))}
                                            </ul>
                                        </div>
                                        {/* Visualization Area if needed, or keeping it clean */}
                                    </div>
                                )}
                            </div>
                        </motion.div>
                    )}
                </AnimatePresence>

                {!selectedModule && (
                    <div className="text-center text-slate-500 dark:text-gray-500 mt-12 mb-24 animate-pulse">
                        Select a module in the diagram above to view its detailed technical specifications
                    </div>
                )}
                
                {/* Modules & Functionalities Deck (PPT Style) */}
                <ModulesDeck />

            </div>
        </div>
    );
};

// --- Sub-components for Diagram ---

const ModuleNode = ({ module, isSelected, onClick, position }) => {
    return (
        <motion.div
            layout
            onClick={onClick}
            className={`relative z-10 group cursor-pointer flex flex-col items-center justify-center p-8 rounded-2xl transition-all duration-300
        ${isSelected
                    ? `bg-${module.color}-500/10 border-2 border-${module.color}-500 shadow-lg shadow-${module.color}-500/20`
                    : 'bg-white border-slate-200 hover:border-slate-400 hover:bg-slate-50 dark:bg-gray-800 border dark:border-gray-700 dark:hover:border-gray-500 dark:hover:bg-gray-700'
                }
      `}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
        >
            <div className={`p-4 rounded-xl mb-4 transition-colors
        ${isSelected ? `bg-${module.color}-500 text-white` : `bg-slate-100 text-slate-500 group-hover:text-${module.color}-600 group-hover:bg-slate-200 dark:bg-gray-700 dark:text-gray-400 dark:group-hover:text-${module.color}-400 dark:group-hover:bg-gray-600`}
      `}>
                {module.icon}
            </div>
            <h2 className={`text-xl font-bold ${isSelected ? 'text-slate-900 dark:text-white' : 'text-slate-700 dark:text-gray-300'}`}>
                {module.title}
            </h2>
            <p className="text-sm text-slate-500 dark:text-gray-500 mt-2 text-center max-w-[200px]">
                {position === 'left' ? 'Client-Side Interaction' :
                    position === 'center' ? 'Server-Side Logic' :
                        'AI Processing Unit'}
            </p>

            {/* Connection Handle */}
            <div className={`absolute ${position === 'left' ? '-right-2' : position === 'right' ? '-left-2' : 'hidden'} top-1/2 w-4 h-4 bg-slate-300 dark:bg-gray-600 rounded-full transform -translate-y-1/2`} />
        </motion.div>
    );
};

const ConnectionArrow = ({ active }) => {
    return (
        <div className="relative h-px bg-slate-300 dark:bg-gray-700 w-full">
            <motion.div
                className="absolute top-0 left-0 h-full bg-blue-500"
                initial={{ width: '0%' }}
                animate={{ width: '100%' }}
                transition={{ duration: 2, repeat: Infinity }}
            />
            <div className="absolute right-0 top-1/2 transform translate-x-1/2 -translate-y-1/2 text-slate-300 dark:text-gray-700">
                <ArrowRight className="w-6 h-6" />
            </div>
        </div>
    );
};

// --- PPT Style Modules Deck ---

const ModulesDeck = () => {
    const [activeTab, setActiveTab] = useState('frontend');

    const slides = {
        frontend: {
            title: "Frontend Module",
            subtitle: "User Experience & Visualization Layer",
            color: "blue",
            icon: <Globe className="w-6 h-6" />,
            content: [
                {
                    header: "Upload Manager",
                    items: [
                        "Validates .dcm/NIfTI file signatures",
                        "Secure drag-and-drop zone with visual feedback",
                        "Asynchronous file reading for large datasets"
                    ]
                },
                {
                    header: "3D Visualizer",
                    items: [
                        "WebGL-based slice rendering engine",
                        "real-time probability heatmap overlay",
                        "Interactive windowing (contrast/brightness) controls"
                    ]
                },
                {
                    header: "Result Logic",
                    items: [
                        "Dynamic Risk Badges (High/Moderate/Low)",
                        "Weighted Probability Risk Calculation",
                        "Graceful Error Handling & User Notifications"
                    ]
                }
            ]
        },
        backend: {
            title: "Backend Module",
            subtitle: "Orchestration & API Layer",
            color: "green",
            icon: <Server className="w-6 h-6" />,
            content: [
                {
                    header: "API Gateway",
                    items: [
                        "FastAPI Route Handling for /analyze endpoints",
                        "Pydantic Input Schema Validation",
                        "CORS Security Middleware & Rate Limiting"
                    ]
                },
                {
                    header: "Async Manager",
                    items: [
                        "Uvicorn-based concurrency model",
                        "Non-blocking inference task execution",
                        "WebSocket state management for live logs"
                    ]
                },
                {
                    header: "File Orchestrator",
                    items: [
                        "Temporary directory lifecycle management",
                        "Secure cleanup of patient data post-analysis",
                        "Disk I/O optimization for large volumes"
                    ]
                }
            ]
        },
        ml_engine: {
            title: "ML Engine Module",
            subtitle: "Deep Learning Intelligence Layer",
            color: "purple",
            icon: <Brain className="w-6 h-6" />,
            content: [
                {
                    header: "Input Pipeline",
                    items: [
                        "Multi-Modality Support (CTA, MRA, MRI)",
                        "Parallel DICOM Reading (Joblib Multi-processing)",
                        "RAS Orientation Standardization (SimpleITK)"
                    ]
                },
                {
                    header: "Deep Learning Core",
                    items: [
                        "3D Residual Encoder U-Net Architecture",
                        "Sliding Window Inference Strategy (50% Overlap)",
                        "Multi-label Sigmoid Activation Function"
                    ]
                },
                {
                    header: "Output Handler",
                    items: [
                        "Anatomical Left/Right Correction Logic",
                        "3D Coordinate Extraction & Mapping",
                        "Base64 Image Generation for Frontend"
                    ]
                }
            ]
        }
    };

    return (
        <div className="mt-24">
            <div className="text-center mb-12">
                <h2 className="text-3xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-slate-900 to-slate-500 dark:from-gray-100 dark:to-gray-400">
                    Modules & Functionalities
                </h2>
                <p className="text-slate-500 dark:text-gray-500 mt-2">Technical breakdown of system components</p>
            </div>

            <div className="max-w-5xl mx-auto bg-white border-slate-200 dark:bg-gray-800/30 rounded-3xl border dark:border-gray-700/50 backdrop-blur-xl overflow-hidden shadow-2xl">
                {/* Tab Navigation */}
                <div className="flex border-b border-slate-200 dark:border-gray-700/50">
                    {Object.entries(slides).map(([key, slide]) => (
                        <button
                            key={key}
                            onClick={() => setActiveTab(key)}
                            className={`flex-1 p-6 text-sm font-medium transition-all flex items-center justify-center gap-3
                                ${activeTab === key 
                                    ? `bg-${slide.color}-500/10 text-${slide.color}-600 dark:text-${slide.color}-400 border-b-2 border-${slide.color}-500` 
                                    : 'text-slate-500 hover:text-slate-900 hover:bg-slate-50 dark:text-gray-500 dark:hover:text-gray-300 dark:hover:bg-white/5'
                                }
                            `}
                        >
                            {slide.icon}
                            <span className="uppercase tracking-wider">{slide.title}</span>
                        </button>
                    ))}
                </div>

                {/* Slide Content */}
                <div className="p-12 min-h-[400px] relative">
                    <AnimatePresence mode="wait">
                        <motion.div
                            key={activeTab}
                            initial={{ opacity: 0, x: 20 }}
                            animate={{ opacity: 1, x: 0 }}
                            exit={{ opacity: 0, x: -20 }}
                            transition={{ duration: 0.3 }}
                        >
                            <div className="flex items-center gap-4 mb-8">
                                <div className={`p-3 rounded-lg bg-${slides[activeTab].color}-500/20 text-${slides[activeTab].color}-400`}>
                                    {slides[activeTab].icon}
                                </div>
                                <div>
                                    <h3 className="text-2xl font-bold text-slate-900 dark:text-white">{slides[activeTab].title}</h3>
                                    <p className={`text-${slides[activeTab].color}-600 dark:text-${slides[activeTab].color}-400/80 font-mono text-sm`}>
                                        {slides[activeTab].subtitle}
                                    </p>
                                </div>
                            </div>

                            <div className="grid md:grid-cols-3 gap-8">
                                {slides[activeTab].content.map((section, idx) => (
                                    <div key={idx} className="bg-slate-50 border-slate-200 hover:border-slate-300 dark:bg-gray-900/40 rounded-xl p-6 border dark:border-white/5 dark:hover:border-white/10 transition-colors">
                                        <h4 className="text-lg font-semibold text-slate-900 dark:text-gray-200 mb-4 pb-2 border-b border-slate-200 dark:border-gray-700/50">
                                            {section.header}
                                        </h4>
                                        <ul className="space-y-3">
                                            {section.items.map((item, i) => (
                                                <li key={i} className="flex items-start gap-2 text-sm text-slate-600 dark:text-gray-400 leading-relaxed">
                                                    <span className={`mt-1.5 w-1.5 h-1.5 rounded-full bg-${slides[activeTab].color}-500 shrink-0`} />
                                                    {item}
                                                </li>
                                            ))}
                                        </ul>
                                    </div>
                                ))}
                            </div>
                        </motion.div>
                    </AnimatePresence>
                </div>
            </div>
        </div>
    );
};

export default Architecture;
