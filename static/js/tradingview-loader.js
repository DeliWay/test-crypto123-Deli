// static/js/tradingview-loader.js
class TradingViewLoader {
    static async load() {
        if (window.TradingView) {
            return true;
        }

        return new Promise((resolve, reject) => {
            // Создаем script элемент с nonce
            const script = document.createElement('script');
            script.nonce = document.currentScript.nonce; // Наследуем nonce
            script.src = 'https://s3.tradingview.com/tv.js';
            script.async = true;
            script.onload = () => resolve(true);
            script.onerror = () => reject(new Error('Failed to load TradingView'));

            document.head.appendChild(script);
        });
    }

    static initWidget(symbol, timeframe, theme, containerId = 'tradingview-chart') {
        if (typeof TradingView === 'undefined') {
            console.error('TradingView not loaded');
            return null;
        }

        try {
            return new TradingView.widget({
                symbol: `BINANCE:${symbol}`,
                interval: timeframe,
                container_id: containerId,
                locale: "ru",
                autosize: true,
                theme: theme,
                style: "1",
                toolbar_bg: "#f1f3f6",
                enable_publishing: false,
                hide_top_toolbar: false,
                hide_legend: false,
                save_image: false,
                studies: [
                    "RSI@tv-basicstudies",
                    "MACD@tv-basicstudies",
                    "StochasticRSI@tv-basicstudies",
                    "BB@tv-basicstudies"
                ],
                drawings_access: {
                    type: 'black',
                    tools: [
                        { name: "Regression Trend" },
                        { name: "Trend Angle" },
                        { name: "Trend Line" },
                        { name: "Horizontal Line" },
                        { name: "Vertical Line" },
                        { name: "Line" },
                        { name: "Arrow" }
                    ]
                }
            });
        } catch (error) {
            console.error('Error initializing TradingView widget:', error);
            return null;
        }
    }

    static showError(containerId = 'tradingview-chart') {
        const container = document.getElementById(containerId);
        if (container) {
            container.innerHTML = `
                <div class="tradingview-error">
                    <i class="fas fa-exclamation-triangle"></i>
                    <h4>Не удалось загрузить график</h4>
                    <p>Попробуйте обновить страницу или проверьте подключение к интернету</p>
                    <button class="btn btn-primary" id="reload-tradingview">
                        <i class="fas fa-sync-alt"></i> Обновить график
                    </button>
                </div>
            `;

            document.getElementById('reload-tradingview').addEventListener('click', () => {
                window.location.reload();
            });
        }
    }

    static showLoading(containerId = 'tradingview-chart') {
        const container = document.getElementById(containerId);
        if (container) {
            container.innerHTML = `
                <div class="chart-loading">
                    <div class="chart-loading-spinner"></div>
                    <p>Загрузка графика...</p>
                </div>
            `;
        }
    }
}