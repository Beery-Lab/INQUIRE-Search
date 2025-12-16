export function Button({children, onClick, className, size="md", buttonStyles="text-white bg-blue-500 hover:bg-blue-700"}) {
    let size_styles = {
        "md": "h-9 px-4 py-2 text-sm",
        "sm": "h-7 px-3 py-1 text-xs"
    }[size];

    return (
        <button 
            onClick={onClick || (() => {})} 
            className={`${size_styles} rounded-lg ${buttonStyles} ${className}`}
        >
            {children}
        </button>
    )
}

export function ButtonRed({ children, ...props }) {
    return <Button buttonStyles='text-white bg-red-500 hover:bg-red-600' {...props}>{children}</Button>
}

export function ButtonOutline({ children, ...props }) {
    return <Button buttonStyles='text-blue-600 border border-blue-600 hover:bg-slate-100' {...props}>{children}</Button>
}


export function BadgeGreen({ children, className }) {
  return (
    <div className={`inline-block px-4 py-0.5 rounded-full bg-green-600 text-white text-sm font-medium ${className}`}>
      {children}
    </div>
  )
}

export function BadgeRed({ children, className }) {
  return (
    <div className={`inline-block px-4 py-0.5 rounded-full bg-red-600 text-white text-sm font-medium ${className}`}>
      {children}
    </div>
  )
}