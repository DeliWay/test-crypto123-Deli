if(!window.formatPrice){window.formatPrice=(v, opts={})=>{
  if(v==null || v==='') return '—';
  const n = Number(v);
  if(Number.isNaN(n)) return '—';
  const {max=8, min=2} = opts;
  const digits = Math.min(Math.max(min, (n<1? 6 : 2)), max);
  return n.toLocaleString(undefined, {minimumFractionDigits:digits, maximumFractionDigits:digits});
};}