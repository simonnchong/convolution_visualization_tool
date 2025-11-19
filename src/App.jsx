import React, { useState, useMemo, useRef, useCallback, useEffect } from 'react';
import { 
  Play, Pause, Microscope, 
  ArrowRight, Eraser, Activity, Grid, MousePointer2,
  Hash, Upload, Image as ImageIcon, Sun, Moon,
  Calculator
} from 'lucide-react';

// --- MATH & LOGIC ENGINE ---

// Helper to generate filters based on size
const generateFilters = (kSize) => {
  const center = Math.floor(kSize / 2);
  
  // 1. Horizontal Edge
  const horiz = Array(kSize).fill(0).map(() => Array(kSize).fill(0));
  for(let y=0; y<kSize; y++) {
    for(let x=0; x<kSize; x++) {
      if (y < center) horiz[y][x] = -1;
      else if (y > center) horiz[y][x] = -1;
      else horiz[y][x] = 2;
    }
  }

  // 2. Vertical Edge
  const vert = Array(kSize).fill(0).map(() => Array(kSize).fill(0));
  for(let y=0; y<kSize; y++) {
    for(let x=0; x<kSize; x++) {
      if (x < center) vert[y][x] = -1;
      else if (x > center) vert[y][x] = -1;
      else vert[y][x] = 2;
    }
  }

  // 3. Sharpen 
  const sharpen = Array(kSize).fill(0).map(() => Array(kSize).fill(0));
  sharpen[center][center] = kSize === 3 ? 5 : 9;
  if(kSize === 3) {
      sharpen[0][1] = -1; sharpen[1][0] = -1; sharpen[1][2] = -1; sharpen[2][1] = -1;
  }

  // 4. Emboss (3x3 only visual mostly)
  const emboss = Array(kSize).fill(0).map(() => Array(kSize).fill(0));
  if (kSize === 3) {
      emboss[0][0] = -2; emboss[0][1] = -1; emboss[0][2] = 0;
      emboss[1][0] = -1; emboss[1][1] = 1;  emboss[1][2] = 1;
      emboss[2][0] = 0;  emboss[2][1] = 1;  emboss[2][2] = 2;
  } else {
      // Simple identity fallback for larger
      emboss[center][center] = 1;
  }

  return {
    horizontal: { name: 'Horizontal Edge', kernel: horiz },
    vertical: { name: 'Vertical Edge', kernel: vert },
    sharpen: { name: 'Sharpen', kernel: sharpen },
    emboss: { name: 'Emboss (3x3)', kernel: emboss },
  };
};

// Activation Functions
const ACTIVATIONS = {
  relu: (x) => Math.max(0, x),
  sigmoid: (x) => 1 / (1 + Math.exp(-x)),
  tanh: (x) => Math.tanh(x),
  none: (x) => x
};

export default function DeepLabCNN() {
  // --- STATE ---
  
  // Theme State (Default to 'dark')
  const [theme, setTheme] = useState('dark');
  
  // Dimensions
  const [gridSize, setGridSize] = useState(14); 
  const [kernelSize, setKernelSize] = useState(3);

  // Data
  const [inputGrid, setInputGrid] = useState(new Float32Array(14 * 14).fill(0));
  const [isDrawing, setIsDrawing] = useState(false);
  
  // Model Parameters
  const [stride, setStride] = useState(1);
  const [padding, setPadding] = useState(0); // Always 0 (Valid)
  const [activation, setActivation] = useState('relu');
  
  // Filter State
  const [filters, setFilters] = useState(generateFilters(3));
  const [selectedFilterIdx, setSelectedFilterIdx] = useState(0); // Track which filter we are "inspecting" for math
  
  // Visual Settings
  const [showValues, setShowValues] = useState(true); 
  const [hoveredPixel, setHoveredPixel] = useState(null); 
  const fileInputRef = useRef(null);

  // Animation State
  const [isAnimating, setIsAnimating] = useState(false);

  // --- EFFECTS ---

  // When gridSize changes, resize inputGrid
  useEffect(() => {
    setInputGrid(new Float32Array(gridSize * gridSize).fill(0));
  }, [gridSize]);

  // When kernelSize changes, regenerate filters
  useEffect(() => {
    setFilters(generateFilters(kernelSize));
  }, [kernelSize]);

  // Animation Loop
  useEffect(() => {
    if (isAnimating) {
      let currentIdx = 0;
      // Calculate max output index based on current settings
      const kSize = kernelSize;
      const outDim = Math.floor((gridSize + 2 * padding - kSize) / stride) + 1;
      const totalPixels = outDim * outDim;

      const interval = setInterval(() => {
        if (currentIdx >= totalPixels) {
            currentIdx = 0; // Loop
        }
        
        const y = Math.floor(currentIdx / outDim);
        const x = currentIdx % outDim;
        
        setHoveredPixel({ layer: 'conv', x, y, filterIndex: selectedFilterIdx });
        currentIdx++;
      }, 200); 

      return () => clearInterval(interval);
    } else {
        setHoveredPixel(null);
    }
  }, [isAnimating, gridSize, kernelSize, stride, padding, selectedFilterIdx]);


  // --- LOGIC HANDLERS ---

  const handleDraw = useCallback((e) => {
    if(e.type === 'touchmove') { /* e.preventDefault(); */ }
    if (!isDrawing && e.type !== 'click' && e.type !== 'mousedown' && e.type !== 'touchstart') return;
    
    const area = document.getElementById('draw-area');
    if (!area) return;

    const rect = area.getBoundingClientRect();
    let clientX, clientY;
    if (e.touches && e.touches.length > 0) {
      clientX = e.touches[0].clientX;
      clientY = e.touches[0].clientY;
    } else {
      clientX = e.clientX;
      clientY = e.clientY;
    }

    const x = clientX - rect.left;
    const y = clientY - rect.top;
    
    const scale = gridSize / rect.width;
    const gridX = Math.floor(x * scale);
    const gridY = Math.floor(y * scale);

    if (gridX >= 0 && gridX < gridSize && gridY >= 0 && gridY < gridSize) {
      setInputGrid(prev => {
        const newGrid = new Float32Array(prev);
        // Brush logic
        const brush = [
          {dx:0, dy:0, val: 1.0},
          {dx:1, dy:0, val: 0.5}, {dx:-1, dy:0, val: 0.5},
          {dx:0, dy:1, val: 0.5}, {dx:0, dy:-1, val: 0.5}
        ];
        
        brush.forEach(({dx, dy, val}) => {
          const bx = gridX + dx;
          const by = gridY + dy;
          if (bx >= 0 && bx < gridSize && by >= 0 && by < gridSize) {
            const idx = by * gridSize + bx;
            newGrid[idx] = Math.min(1, newGrid[idx] + val);
          }
        });
        return newGrid;
      });
    }
  }, [isDrawing, gridSize]);

  const handleImageUpload = (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (event) => {
      const img = new Image();
      img.onload = () => {
        const canvas = document.createElement('canvas');
        canvas.width = gridSize;
        canvas.height = gridSize;
        const ctx = canvas.getContext('2d');
        ctx.fillStyle = 'black'; 
        ctx.fillRect(0, 0, gridSize, gridSize);
        ctx.drawImage(img, 0, 0, gridSize, gridSize);
        const imgData = ctx.getImageData(0, 0, gridSize, gridSize);
        const data = imgData.data;
        const newGrid = new Float32Array(gridSize * gridSize);
        for (let i = 0; i < data.length; i += 4) {
          const val = (data[i] + data[i + 1] + data[i + 2]) / 3 / 255; 
          newGrid[i / 4] = val;
        }
        setInputGrid(newGrid);
      };
      img.src = event.target.result;
    };
    reader.readAsDataURL(file);
  };

  // --- CNN ENGINE (MEMOIZED) ---
  
  // 1. Convolution
  const featureMaps = useMemo(() => {
    return Object.values(filters).map(filter => {
      const kSize = filter.kernel.length;
      const outDim = Math.floor((gridSize + 2 * padding - kSize) / stride) + 1;
      
      if (outDim <= 0) return { name: filter.name, data: new Float32Array(0), dim: 0 };

      const mapData = new Float32Array(outDim * outDim);

      for (let y = 0; y < outDim; y++) {
        for (let x = 0; x < outDim; x++) {
          let sum = 0;
          for (let ky = 0; ky < kSize; ky++) {
            for (let kx = 0; kx < kSize; kx++) {
              const iy = (y * stride) - padding + ky;
              const ix = (x * stride) - padding + kx;
              
              let inputVal = 0; // Default for padding
              if (iy >= 0 && iy < gridSize && ix >= 0 && ix < gridSize) {
                inputVal = inputGrid[iy * gridSize + ix];
              }

              const weight = filter.kernel[ky][kx];
              sum += inputVal * weight;
            }
          }
          const val = ACTIVATIONS[activation](sum);
          mapData[y * outDim + x] = val;
        }
      }
      return { name: filter.name, data: mapData, dim: outDim };
    });
  }, [inputGrid, stride, padding, activation, gridSize, filters]);

  // --- MATH CALCULATOR ---
  const getMathDetails = () => {
      if (!hoveredPixel || hoveredPixel.layer !== 'conv') return null;
      
      const { x, y, filterIndex } = hoveredPixel;
      const filterKey = Object.keys(filters)[filterIndex];
      const filter = filters[filterKey];
      if(!filter) return null;

      const kSize = filter.kernel.length;
      let calculations = [];
      let total = 0;

      for (let ky = 0; ky < kSize; ky++) {
        for (let kx = 0; kx < kSize; kx++) {
            const iy = (y * stride) - padding + ky;
            const ix = (x * stride) - padding + kx;
            
            let val = 0;
            let isPadding = true;
            if (iy >= 0 && iy < gridSize && ix >= 0 && ix < gridSize) {
                val = inputGrid[iy * gridSize + ix];
                isPadding = false;
            }
            
            const weight = filter.kernel[ky][kx];
            const product = val * weight;
            total += product;
            
            if (gridSize <= 14 || product !== 0) {
                 calculations.push({ val, weight, product, isPadding });
            }
        }
      }
      
      const activated = ACTIVATIONS[activation](total);
      
      return { calculations, total, activated, filterName: filter.name };
  };

  // --- RENDERERS ---

  const ActivationGraph = ({ type }) => {
      // Simple visualizer for activation function
      const pts = [];
      const range = 10;
      for(let i=-range; i<=range; i+=1) {
          const y = ACTIVATIONS[type](i);
          const px = (i + range) * (60 / (2*range));
          let py = 30;
          if(type === 'relu') py = 30 - (y * 3); // Scale down
          else if (type === 'sigmoid' || type === 'tanh') py = 30 - (y * 15);
          else py = 30 - (y * 3);

          pts.push(`${px},${py}`);
      }

      return (
          <div className="w-16 h-10 bg-slate-900 border border-slate-700 rounded relative overflow-hidden">
              <svg width="100%" height="100%" viewBox="0 0 60 40" className="text-blue-500">
                  <line x1="0" y1="30" x2="60" y2="30" stroke="#475569" strokeWidth="1" /> {/* X Axis */}
                  <line x1="30" y1="0" x2="30" y2="40" stroke="#475569" strokeWidth="1" /> {/* Y Axis */}
                  <polyline points={pts.join(' ')} fill="none" stroke="currentColor" strokeWidth="2" />
              </svg>
          </div>
      );
  };

  const GridVisualizer = ({ data, dim, label, highlightRegion, onHoverPixel, isInteractive = false, showNums = false, usePadding = 0 }) => {
    
    // Adjusted pixel size: 20 for interactive (bigger)
    const pixelSize = isInteractive ? 20 : 22; 

    const lightThemeGridBg = 'bg-slate-200';
    const darkThemeGridBg = 'bg-black';
    const gridBg = theme === 'light' ? lightThemeGridBg : darkThemeGridBg;
    const gridBorder = theme === 'light' ? 'border-slate-400' : 'border-slate-700';

    return (
      <div className="flex flex-col items-center gap-2 relative group/grid">
        <div className="relative p-px" style={{ 
            padding: usePadding > 0 ? `${(usePadding/dim)*100}%` : '0', 
            border: usePadding > 0 ? '1px dashed var(--color-border)' : 'none'
        }}>
            {usePadding > 0 && <div className="absolute -top-4 left-0 text-[8px] text-slate-500">Padding: {usePadding}px</div>}
            
            <div 
            className={`relative ${gridBg} ${gridBorder} border shadow-lg select-none overflow-hidden`}
            style={{ 
                width: dim * pixelSize + 'px', 
                height: dim * pixelSize + 'px',
                touchAction: 'none'
            }}
            onMouseLeave={() => onHoverPixel && onHoverPixel(null)}
            >
            <div 
                className="grid w-full h-full pointer-events-none"
                style={{ gridTemplateColumns: `repeat(${dim}, 1fr)` }}
            >
                {Array.from(data).map((val, idx) => {
                return (
                    <div 
                    key={idx}
                    onMouseEnter={(!isInteractive && onHoverPixel) ? () => {
                        const x = idx % dim;
                        const y = Math.floor(idx / dim);
                        onHoverPixel({ x, y });
                    } : undefined}
                    style={{ backgroundColor: `rgb(${val*255}, ${val*255}, ${val*255})` }}
                    className={`w-full h-full flex items-center justify-center border-[0.5px] border-slate-900/10 ${!isInteractive ? 'pointer-events-auto' : ''}`} 
                    >
                        {/* Show numbers if toggle is on */}
                        {showNums && (
                            <span className={`text-[7px] font-mono font-bold ${val > 0.5 ? 'text-black' : 'text-white'}`}>
                                {val.toFixed(1)}
                            </span>
                        )}
                    </div>
                );
                })}
            </div>
            {highlightRegion && (
                <div 
                className="absolute border-2 transition-all duration-75 ease-out z-10 shadow-[0_0_10px_rgba(59,130,246,0.5)]"
                style={{
                    borderColor: highlightRegion.color || '#ef4444',
                    left: `${(highlightRegion.x / dim) * 100}%`,
                    top: `${(highlightRegion.y / dim) * 100}%`,
                    width: `${(highlightRegion.w / dim) * 100}%`,
                    height: `${(highlightRegion.h / dim) * 100}%`,
                }}
                />
            )}
            </div>
        </div>
        
        <div className="text-center">
             <span className="text-[10px] font-mono text-slate-400 block">{label} ({dim}×{dim})</span>
        </div>
      </div>
    );
  };

  const getInputHighlight = () => {
    if (!hoveredPixel) return null;
    if (hoveredPixel.layer === 'conv') {
      const { x, y } = hoveredPixel;
      return {
        x: (x * stride) - padding, 
        y: (y * stride) - padding,
        w: kernelSize, h: kernelSize, color: '#3b82f6'
      };
    }
    return null;
  };

  const mathInfo = getMathDetails();
  
  const themeClasses = {
      dark: 'bg-slate-950 text-slate-200 border-slate-800',
      light: 'bg-white text-slate-900 border-slate-200',
  };
  const cardClasses = theme === 'dark' ? 'bg-slate-900 border-slate-800' : 'bg-slate-50 border-slate-200';
  const topBarClasses = theme === 'dark' ? 'bg-slate-900' : 'bg-slate-100';


  return (
    <div className={`min-h-screen font-sans selection:bg-blue-500 selection:text-white ${themeClasses[theme]}`}>
      
      {/* Top Bar */}
      <div className={`h-20 border-b ${theme === 'dark' ? 'border-slate-800' : 'border-slate-300'} flex items-center px-6 ${topBarClasses} sticky top-0 z-20 overflow-x-auto`}>
        <div className="flex items-center gap-3 mr-8 shrink-0">
          <div className="bg-blue-600 p-1.5 rounded-lg">
            <Microscope className="w-5 h-5 text-white" />
          </div>
          <h1 className="font-bold text-lg tracking-tight">DeepLab <span className="font-normal text-slate-500">v1.0 by Simon</span></h1>
        </div>
        
        <div className="flex-1 flex items-center justify-end gap-6 text-sm shrink-0">
            
             {/* Theme Toggle */}
             <button
                onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
                className={`flex items-center px-3 py-1.5 rounded-full border text-xs transition-all ${theme === 'dark' ? 'bg-slate-800 border-slate-700 text-yellow-400' : 'bg-slate-200 border-slate-300 text-indigo-600'}`}
                title="Toggle Theme"
             >
                {theme === 'dark' ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
             </button>

             {/* Values Toggle */}
             <button 
                onClick={() => setShowValues(!showValues)}
                className={`flex items-center gap-2 px-3 py-1.5 rounded-lg border text-xs font-bold transition-all ${showValues ? 'bg-indigo-600 border-indigo-500 text-white' : 'bg-slate-800 border-slate-700 text-slate-400 hover:text-white'}`}
                title="Toggle Pixel Values"
             >
                <Hash className="w-3 h-3" />
                {showValues ? 'Values ON' : 'Values OFF'}
             </button>

             <div className="w-px h-8 bg-slate-800"></div>

             {/* Animation Control */}
             <button 
                onClick={() => setIsAnimating(!isAnimating)}
                className={`flex items-center gap-2 px-4 py-1.5 rounded-full border text-xs font-bold transition-all ${isAnimating ? 'bg-amber-500 border-amber-600 text-white animate-pulse' : 'bg-emerald-600 border-emerald-500 text-white hover:bg-emerald-500'}`}
             >
                {isAnimating ? <Pause className="w-3 h-3" /> : <Play className="w-3 h-3" />}
                {isAnimating ? 'SCANNING...' : 'ANIMATE SCAN'}
             </button>

             <div className="w-px h-8 bg-slate-800"></div>
             
             {/* Grid Size Control */}
             <div className="flex flex-col items-center gap-1">
                 <span className="text-[10px] text-slate-500 uppercase font-bold">Input Size</span>
                 <div className="flex bg-slate-800 rounded p-0.5">
                  {[14].map(s => (
                    <button key={s} onClick={() => setGridSize(s)} className={`px-2 py-0.5 rounded text-xs ${gridSize === s ? 'bg-blue-600 text-white' : 'text-slate-400 hover:text-white'}`}>{s}</button>
                  ))}
                </div>
             </div>

             {/* Kernel Size Control */}
             <div className="flex flex-col items-center gap-1">
                <span className="text-[10px] text-slate-500 uppercase font-bold">Kernel</span>
                <div className="flex bg-slate-800 rounded p-0.5">
                  {[3, 5].map(s => (
                    <button key={s} onClick={() => setKernelSize(s)} className={`px-2 py-0.5 rounded text-xs ${kernelSize === s ? 'bg-emerald-600 text-white' : 'text-slate-400 hover:text-white'}`}>{s}x{s}</button>
                  ))}
                </div>
             </div>

             {/* Activation & Graph */}
             <div className={`flex items-center gap-3 p-1.5 rounded-lg border ${theme === 'dark' ? 'bg-slate-800/50 border-slate-800' : 'bg-slate-200 border-slate-300'}`}>
                 <div className="flex flex-col">
                    <span className="text-[10px] text-slate-500 uppercase font-bold mb-1">Activation</span>
                    <select 
                        value={activation} 
                        onChange={(e) => setActivation(e.target.value)}
                        className={`border ${theme === 'dark' ? 'bg-slate-900 border-slate-700 text-slate-300' : 'bg-white border-slate-300 text-slate-800'} text-xs rounded py-1 px-1 outline-none w-20`}
                    >
                        <option value="relu">ReLU</option>
                        <option value="sigmoid">Sigmoid</option>
                        <option value="tanh">Tanh</option>
                        <option value="none">No Act.</option>
                    </select>
                 </div>
                 <ActivationGraph type={activation} />
             </div>

        </div>
      </div>

      <div className="flex flex-col lg:flex-row h-[calc(100vh-80px)]">
        
        {/* LEFT COLUMN: INPUT & MATH */}
        <div className={`w-full lg:w-1/3 border-r ${theme === 'dark' ? 'border-slate-800 bg-slate-900' : 'border-slate-300 bg-slate-100'} flex flex-col overflow-y-auto`}>
          
          {/* Drawing Area */}
          <div className={`p-6 border-b ${theme === 'dark' ? 'border-slate-800' : 'border-slate-300'}`}>
            <div className="flex justify-between items-center mb-4">
                <h2 className={`font-semibold flex items-center gap-2 ${theme === 'dark' ? 'text-slate-100' : 'text-slate-800'}`}>
                    <ImageIcon className="w-4 h-4 text-blue-500" /> Input Layer
                </h2>
                <div className="flex gap-2 items-center">
                    <button onClick={() => fileInputRef.current.click()} className="text-xs flex items-center gap-1 bg-blue-900/30 hover:bg-blue-900/50 text-blue-300 px-2 py-1 rounded border border-blue-800 transition-colors">
                        <Upload className="w-3 h-3" /> Upload
                    </button>
                    <input type="file" ref={fileInputRef} className="hidden" accept="image/*" onChange={handleImageUpload} />
                    <button onClick={() => setInputGrid(new Float32Array(gridSize*gridSize).fill(0))} className="text-xs text-slate-400 hover:text-red-400 ml-1"><Eraser className="w-3 h-3" /></button>
                </div>
            </div>

            <div className="flex flex-col items-center justify-center gap-4 min-h-[300px]">
                <div 
                id="draw-area"
                className="relative shadow-2xl shadow-blue-900/20 cursor-crosshair group touch-none"
                onMouseDown={() => setIsDrawing(true)}
                onMouseUp={() => setIsDrawing(false)}
                onMouseLeave={() => setIsDrawing(false)}
                onMouseMove={handleDraw}
                onClick={handleDraw}
                onTouchStart={(e) => { setIsDrawing(true); handleDraw(e); }}
                onTouchEnd={() => setIsDrawing(false)}
                onTouchMove={handleDraw}
                >
                <GridVisualizer 
                    data={inputGrid} 
                    dim={gridSize} 
                    label={`${gridSize}x${gridSize} Input`}
                    highlightRegion={getInputHighlight()} 
                    isInteractive={true}
                    showNums={showValues} 
                    usePadding={0} 
                />
                 <div className="absolute inset-0 flex items-center justify-center pointer-events-none opacity-0 group-hover:opacity-100 transition-opacity duration-500">
                    {!inputGrid.some(x=>x>0) && !isDrawing && (
                        <div className={`flex flex-col items-center ${theme === 'dark' ? 'text-slate-500 bg-slate-900/90 border-slate-800' : 'text-slate-600 bg-slate-200/90 border-slate-300'} px-4 py-2 rounded-xl backdrop-blur border`}>
                            <MousePointer2 className="w-5 h-5 mb-1" />
                            <span className="font-mono text-xs">Draw Here</span>
                        </div>
                    )}
                </div>
                </div>
            </div>
          </div>

          {/* Math Explainer Panel */}
          <div className={`flex-1 p-6 ${theme === 'dark' ? 'bg-slate-900/50' : 'bg-slate-100/50'} overflow-y-auto`}>
             <h2 className={`font-semibold flex items-center gap-2 mb-4 ${theme === 'dark' ? 'text-slate-100' : 'text-slate-800'}`}>
                 <Calculator className="w-4 h-4 text-amber-500" /> Convolution Math
             </h2>
             
             {mathInfo ? (
                <div className="space-y-4 animate-in fade-in duration-300">
                    <div className={`text-xs p-2 rounded border ${theme === 'dark' ? 'text-slate-400 bg-slate-800 border-slate-700' : 'text-slate-600 bg-slate-200 border-slate-300'}`}>
                        Analyzing: <span className="text-amber-400 font-bold">{mathInfo.filterName}</span> at ({hoveredPixel.x}, {hoveredPixel.y})
                    </div>
                    
                    <div className="space-y-1">
                        {mathInfo.calculations.slice(0, 9).map((calc, i) => (
                            <div key={i} className="flex items-center text-xs font-mono">
                                <span className="text-slate-500 w-4">({i})</span>
                                <span className={`w-12 text-right ${calc.isPadding ? 'text-slate-600 italic' : (theme === 'dark' ? 'text-blue-400' : 'text-blue-700')}`}>
                                    {calc.isPadding ? '0 (pad)' : calc.val.toFixed(2)}
                                </span>
                                <span className="px-2 text-slate-600">×</span>
                                <span className={`w-8 text-right ${theme === 'dark' ? 'text-emerald-400' : 'text-emerald-700'}`}>{calc.weight}</span>
                                <span className="px-2 text-slate-600">=</span>
                                <span className={`${theme === 'dark' ? 'text-white' : 'text-slate-900'}`}>{calc.product.toFixed(2)}</span>
                            </div>
                        ))}
                        {mathInfo.calculations.length > 9 && <div className="text-xs text-slate-600 pl-4">...and {mathInfo.calculations.length - 9} more</div>}
                    </div>

                    <div className={`border-t ${theme === 'dark' ? 'border-slate-700' : 'border-slate-300'} pt-2 mt-2`}>
                        <div className="flex justify-between text-sm">
                            <span className="text-slate-400">Sum:</span>
                            <span className={`font-mono ${theme === 'dark' ? 'text-white' : 'text-slate-800'}`}>{mathInfo.total.toFixed(2)}</span>
                        </div>
                        <div className="flex justify-between text-sm font-bold text-amber-400 mt-1">
                            <span>{activation.toUpperCase()}(Sum):</span>
                            <span>{mathInfo.activated.toFixed(2)}</span>
                        </div>
                    </div>
                </div>
             ) : (
                <div className={`flex flex-col items-center justify-center h-32 text-xs text-center border-2 border-dashed rounded-xl ${theme === 'dark' ? 'text-slate-500 border-slate-800' : 'text-slate-400 border-slate-400'}`}>
                    <MousePointer2 className="w-6 h-6 mb-2 opacity-20" />
                    Hover over a feature map pixel<br/>to see the math.
                </div>
             )}
          </div>

        </div>

        {/* RIGHT COLUMN: PIPELINE */}
        <div className={`flex-1 overflow-y-auto ${theme === 'dark' ? 'bg-slate-950' : 'bg-slate-200'} p-6`}>
          
          {/* STEP 1: CONVOLUTION */}
          <section className="mb-10">
            <div className="flex items-center gap-3 mb-6">
              <div className="bg-emerald-600/20 p-2 rounded">
                <Activity className="w-5 h-5 text-emerald-500" />
              </div>
              <div>
                <h2 className={`text-lg font-medium ${theme === 'dark' ? 'text-slate-200' : 'text-slate-800'}`}>Convolution Layer</h2>
                <p className="text-xs text-slate-500 font-mono">
                  Kernel: {kernelSize}x{kernelSize} | Stride: {stride} | Pad: {padding}
                </p>
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-12">
              {Object.values(filters).map((filter, idx) => {
                 const map = featureMaps[idx];
                 return (
                    <div 
                        key={idx} 
                        onMouseEnter={() => setSelectedFilterIdx(idx)}
                        className={`border rounded-xl p-4 flex flex-col items-center transition-colors group cursor-crosshair ${theme === 'dark' ? 'bg-slate-900 border-slate-800 hover:border-slate-600' : 'bg-white border-slate-200 hover:border-slate-400'} ${hoveredPixel?.layer === 'conv' && hoveredPixel.filterIndex === idx ? 'border-blue-500 ring-1 ring-blue-500 bg-blue-900/10' : ''}`}
                    >
                        <div className="flex justify-between items-center w-full mb-3">
                             <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-400">{filter.name}</h3>
                             {/* Kernel Preview Mini */}
                             <div className={`border p-1 rounded shadow-sm transition-colors ${theme === 'dark' ? 'bg-slate-800 border-slate-700 group-hover:border-slate-500' : 'bg-slate-200 border-slate-300 group-hover:border-slate-500'}`} title="Kernel Weights">
                                  <div className={`grid gap-px ${theme === 'dark' ? 'bg-slate-600 border-slate-600' : 'bg-slate-400 border-slate-400'}`} style={{ gridTemplateColumns: `repeat(${filter.kernel.length}, 1fr)` }}>
                                      {filter.kernel.flat().map((k, i) => (
                                          <div key={i} className={`w-2 h-2 ${k > 0 ? 'bg-white' : k < 0 ? 'bg-black' : (theme === 'dark' ? 'bg-slate-400' : 'bg-slate-600')}`} />
                                      ))}
                                  </div>
                             </div>
                        </div>
                        <div className="relative">
                            <GridVisualizer 
                                data={map.data} 
                                dim={map.dim} 
                                label={`Feature Map ${idx+1}`}
                                highlightRegion={hoveredPixel?.layer === 'conv' && hoveredPixel.filterIndex === idx ? {x: hoveredPixel.x, y: hoveredPixel.y, w: 1, h: 1, color: 'white'} : null}
                                onHoverPixel={(px) => setHoveredPixel(px ? { layer: 'conv', ...px, filterIndex: idx } : null)}
                                showNums={showValues} // Use the toggle state
                            />
                        </div>
                    </div>
                 )
              })}
            </div>
          </section>

        </div>
      </div>
    </div>
  );
}
