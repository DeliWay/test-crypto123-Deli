# crypto-trading-platform (merged)

Production-ready cleanup with additive changes only. See "Changes" below.

## Run (dev)
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
FLASK_APP=crypto-trading-platform/app.py FLASK_ENV=development flask run
```

## Run (prod)
Use a WSGI server (gunicorn/uwsgi) and a reverse proxy. Set `ENABLE_NOTIFICATIONS=false` unless configured.

## .env
Copy `.env.example` to `.env` and fill values.

## Self-check
```bash
python scripts/selfcheck.py
```

## Release Checklist
- RU/EN switch works, all keys exist.
- Theme switch updates background/text/panels.
- No duplicate `<script>`; no `already been declared` / `Unexpected identifier`.
- Prices show via `window.formatPrice` only.
- Bybit v5 fields strictly: `lastPrice, price24hPcnt, highPrice24h, lowPrice24h`.
- TradingView syncs to store symbol/interval.
- Calculator is adaptive.
- Alerts debounced, history visible; CSV export works.
- CSP allows TradingView and Google Fonts.
- Archive `crypto-trading-platform-merged.zip` present.


## Analysis V2

Set `ANALYSIS_V2=1` or open `/analyze?symbol=BTCUSDT&timeframe=60&v2=1` to use the redesigned analysis page.


### Legacy analysis template

- The previous analysis page was preserved as `templates/analysis_legacy.html`.
- `templates/analysis.html` now uses the redesigned layout (formerly `analysis_v2.html`).

