if(!window.app){window.app=(function(){
  const s={symbol:'BTCUSDT', interval:'60'}; const subs=new Set();
  function set(p){ Object.assign(s,p); subs.forEach(fn=>fn(Object.freeze({...s}))); }
  function get(){ return Object.freeze({...s}); }
  function sub(fn){ subs.add(fn); try{ fn(get()); }catch(e){} return ()=>subs.delete(fn); }
  return {set,get,sub};
})();}