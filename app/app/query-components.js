// "use client";
// Lat/Long inputs (location filter) removed from frontend.
// Kept QueryImageGrid and image components only.
import React from 'react';


function QueryImage({ id, src, selected, active, gridSize, onClick, onHover }) {
    const [imageError, setImageError] = React.useState(false);
    
    return (
        <div
            className={`flex m-2 bg-slate-100 overflow-hidden cursor-pointer hover:opacity-75 border-4 ` +
                      `${selected ? 'border-orange-400' : 'border-slate-200'} ` + 
                      `${active ? 'scale-105 p-0.5 opacity-80 ' : ''}`}
            // style={{'boxShadow': selected ? '0 0 10px 4px #94e7a6' : 'none',
            style={{'boxShadow': selected ? '0 0 7px 4px rgb(231, 188, 148)' : 'none',
                    'width':  `${gridSize ?? 10}rem`,
                    'height': `${gridSize ?? 10}rem`}}
            onClick={onClick} onMouseEnter={onHover}
            id={`im-${id}`}
            >
            {imageError ? (
                <div className='flex items-center justify-center w-full h-full bg-red-100 text-red-600 text-xs p-2 text-center'>
                    Failed to load image
                </div>
            ) : (
                <img 
                    src={src} 
                    className='object-cover w-full h-full border-2 border-white'
                    onError={(e) => {
                        console.error('Image failed to load:', src);
                        setImageError(true);
                    }}
                    alt={`Image ${id}`}
                />
            )}
        </div>
    )
  }
  
export function QueryImageGrid({ images, selectedIds, gridSize, hoveredImageIdx, onImageClick, onImageHover }) {
    return (
      <div className='p-3 flex justify-evenly flex-wrap overflow-scroll'>
          {images.map((im, key) => {
              const selected = selectedIds.includes(im.id);
              return (
                <QueryImage  key={im.id} id={im.id} src={im.src} selected={selected} active={key == hoveredImageIdx} gridSize={gridSize}
                  onClick={() => {onImageClick(im.id)}} 
                  onHover={() => {onImageHover(im.id)}}
                />
              )
          })}
      </div>
    )
  }