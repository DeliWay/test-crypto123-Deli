if(!window.guards){window.guards=(function(){
  function noDuplicateScripts(){
    const srcs=[...document.querySelectorAll('script[src]')].map(s=>s.src);
    const dups = srcs.filter((s,i)=>srcs.indexOf(s)!==i);
    if(dups.length){ console.warn("Duplicate <script> detected:", dups); }
    return dups;
  }
  return {noDuplicateScripts};
})();}