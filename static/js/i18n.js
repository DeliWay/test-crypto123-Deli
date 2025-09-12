if(!window.i18n){window.i18n=(function(){
  const LS='lang'; let dict={}; let lang=localStorage.getItem(LS)||'ru';
  async function load(l){
    const r=await fetch(`/static/i18n/${l}.json`); dict=await r.json(); lang=l; localStorage.setItem(LS,l); apply();
  }
  function t(k){ return dict[k]||k; }
  function apply(){ document.querySelectorAll('[data-i18n]').forEach(el=>el.textContent=t(el.getAttribute('data-i18n'))); }
  document.addEventListener('DOMContentLoaded',()=>load(lang));
  return {t,load,apply};
})();}