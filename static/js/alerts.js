if(!window.alerts){window.alerts=(function(){
  const LS='alerts'; let list=[]; let t;
  function push(a){ list.push({...a, ts: Date.now()}); if(list.length>50){ list.shift(); } save(); render(); }
  function save(){ try{ localStorage.setItem(LS, JSON.stringify(list)); }catch(e){} }
  function load(){ try{ list = JSON.parse(localStorage.getItem(LS)||'[]'); }catch(e){ list=[]; } }
  function debounce(fn, ms){ return (...args)=>{ clearTimeout(t); t=setTimeout(()=>fn(...args), ms); }; }
  function render(){ const el=document.getElementById('alerts-list'); if(!el) return; el.innerHTML=''; list.slice().reverse().forEach(it=>{
    const d=document.createElement('div'); d.className='card'; d.textContent = `${new Date(it.ts).toLocaleString()} — ${it.type}`; el.appendChild(d);
  });}
  document.addEventListener('DOMContentLoaded',()=>{load(); render();});
  return {push,load,render,debounce};
})();}