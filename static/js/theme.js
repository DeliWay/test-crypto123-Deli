if(!window.theme){window.theme=(function(){
  const LS='theme'; let th=localStorage.getItem(LS)||'dark';
  function apply(){
    document.documentElement.dataset.theme=th;
  }
  function toggle(){ th = (th==='dark'?'light':'dark'); localStorage.setItem(LS, th); apply(); }
  document.addEventListener('DOMContentLoaded', apply);
  return {toggle,apply};
})();}