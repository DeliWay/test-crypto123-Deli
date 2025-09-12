if(!window.tv){window.tv=(function(){
  let widget;
  function mount(container){
    const st=window.app.get();
    try{
      widget=new TradingView.widget({
        symbol: st.symbol.replace('USDT','USD'),
        interval: st.interval,
        container_id: container,
        autosize:true, theme:"dark"
      });
    }catch(e){ console.error(e); }
    window.app.sub(({symbol,interval})=>{
      try{ widget && widget.setSymbol(symbol.replace('USDT','USD'), interval); }catch(e){}
    });
  }
  return {mount};
})();}
// Additive: sync interval from TradingView -> store, and observe price from TV DOM
(function(){
  function onReady(){
    try{
      const chart = widget && widget.activeChart && widget.activeChart();
      if(!chart) return;
      // TV -> store: interval
      chart.onIntervalChanged().subscribe(null, function(iv){
        try{ window.app && window.app.set({interval:String(iv)}); }catch(e){}
      });
      // TV -> store: symbol
      chart.onSymbolChanged().subscribe(null, function(sym){
        try{ if(sym && sym.symbol){ window.app.set({symbol: sym.symbol.replace('USD','USDT')}); } }catch(e){}
      });
    }catch(e){}
  }
  if(widget && widget.onChartReady){ widget.onChartReady(onReady); }
  // Observe price in DOM
  let priceCb=null, obs=null;
  function hookPrice(cb){
    priceCb = cb;
    try{
      const root = document.getElementById('tv_container') || document.body;
      if(obs){ obs.disconnect(); }
      obs = new MutationObserver(()=>{
        try{
          // heuristic: find nodes that look like last price in header
          const nodes = root.querySelectorAll('div,span');
          let best=null;
          for(const el of nodes){
            const t = el.textContent || '';
            if(/\d[\d\s\.,]*$/.test(t) && t.length<20 && el.closest('.chart-controls')==null){
              best = t.replace(/[^\d\.,]/g,'');
            }
          }
          if(best && priceCb){
            const num = Number(best.replace(/\s/g,'').replace(',','.'));
            if(!Number.isNaN(num)){ priceCb(num); }
          }
        }catch(e){}
      });
      obs.observe(root, {childList:true, subtree:true, characterData:true});
    }catch(e){}
  }
  if(!window.tv){ window.tv = {}; }
  window.tv.hookPrice = hookPrice;
})();