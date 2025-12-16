"use client";
import { redirect } from 'next/navigation'
import { useState, useEffect } from 'react';
import * as Icon from 'react-bootstrap-icons';

export default function LoginForm() {
    const [loggedIn, setLoggedIn] = useState(false);

    async function onSubmit(event) {
        event.preventDefault()
     
        console.log('submitted with value: ' + event.currentTarget.elements.namedItem('user_input').value)
    
        const formData = new FormData(event.currentTarget)
        const response = await fetch('/api/login', {
          method: 'POST',
          body: formData,
        })
    
        if (!response.ok) {
          console.error(`HTTP Error! status: ${response.status}`)
          return
        }
     
        const data = await response.json();
        console.log(data);

        if (data.success) {
            setLoggedIn(true);
        }
    }

    useEffect(()=>{
        if (loggedIn){
            redirect("/query", 'push')
        }
    }, [loggedIn])


    return (
        <form onSubmit={onSubmit} className='w-96 text-center mb-10'>
            <img src='/Coracias-garrulus.png' className='w-[80%] mb-6 inline'/>
            <div className='flex'>
                <input name="user_input" 
                    placeholder="Enter the key here..."
                    className='grow px-3 h-10 text-md outline-none bg-white rounded-lg border border-slate-200'/>
                <button type="submit" className='h-10 shrink-0 px-2 ms-2 flex items-center text-slate-600 rounded-lg border border-slate-200 bg-white hover:bg-slate-200 cursor-pointer'>
                    <Icon.ArrowRightShort className='w-8 h-8' />{/* Search */}
                </button>
            </div>
        </form>
    )
}