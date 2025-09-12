// static/js/analysis-core.js
class AnalysisCore {
    static config = {
        refreshInterval: 15000,
        apiTimeout: 10000,
        themeKey: 'cryptoanalyst_theme'
    };

    static state = {
        currentSymbol: '',
        currentTimeframe: '60',
        currentTheme: 'dark',
        autoRefresh: true,
        refreshTimer: null,
        countdown: 15,
        tradingViewWidget: null,
        currentData: null,
        marketStats: null
    };

    static init() {
        this.parseUrlParams();
        this.setupEventListeners();
        this.initTradingView();
        this.startAutoRefresh();
        this.refreshData();
        this.setupThemeHandler();
    }

    static setupThemeHandler() {
        const themeToggle = document.getElementById('theme-toggle');
        if (themeToggle) {
            themeToggle.addEventListener('click', () => {
                const currentTheme = document.documentElement.getAttribute('data-theme');
                const newTheme = currentTheme === 'dark' ? 'light' : 'dark';

                document.documentElement.setAttribute('data-theme', newTheme);
                localStorage.setItem('cryptoanalyst_theme', newTheme);

                const themeIcon = document.querySelector('#theme-toggle i');
                if (themeIcon) {
                    themeIcon.className = newTheme === 'dark' ? 'fas fa-sun' : 'fas fa-moon';
                }

                if (this.state.tradingViewWidget) {
                    try {
                        this.state.tradingViewWidget.changeTheme(newTheme);
                    } catch (error) {
                        console.warn('Ошибка обновления темы TradingView:', error);
                    }
                }
            });
        }
    }

    static parseUrlParams() {
        const urlParams = new URLSearchParams(window.location.search);
        this.state.currentSymbol = urlParams.get('symbol') || 'BTCUSDT';
        this.state.currentTimeframe = urlParams.get('timeframe') || '60';

        document.getElementById('symbol-header').textContent = `Анализ ${this.state.currentSymbol}`;
        document.getElementById('current-symbol').textContent = this.state.currentSymbol;
    }

    static setupEventListeners() {
        // Кнопка обновления
        document.getElementById('refresh-analysis').addEventListener('click', () => {
            this.refreshData();
        });

        // Переключение автообновления
        document.getElementById('toggle-refresh').addEventListener('click', () => {
            this.state.autoRefresh = !this.state.autoRefresh;
            this.toggleAutoRefresh();
        });

        // Кнопки таймфреймов
        document.querySelectorAll('.timeframe-buttons .btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                e.preventDefault();
                document.querySelectorAll('.timeframe-buttons .btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');

                this.state.currentTimeframe = btn.dataset.tf;
                this.refreshData();
                this.updateTradingView();
            });
        });

        // Калькулятор риска
        const riskSlider = document.getElementById('risk-percent');
        const riskValue = document.getElementById('risk-value');
        const depositInput = document.getElementById('deposit-size');

        riskSlider.addEventListener('input', () => {
            riskValue.textContent = `${riskSlider.value}%`;
            this.updateRiskCalculator();
        });

        depositInput.addEventListener('input', () => {
            this.updateRiskCalculator();
        });
    }

    static async refreshData() {
        try {
            this.showLoadingState();
            const [analysisData, marketStats] = await Promise.all([
                this.loadAnalysisData(this.state.currentSymbol, this.state.currentTimeframe),
                this.loadMarketStats(this.state.currentSymbol)
            ]);

            this.state.currentData = analysisData;
            this.state.marketStats = marketStats;
            this.updateUI(analysisData);
            this.updateAdditionalData(marketStats);
        } catch (error) {
            console.error('Ошибка обновления данных:', error);
            this.showError(`Ошибка загрузки данных: ${error.message}`);
        }
    }

    static async loadAnalysisData(symbol, timeframe) {
        try {
            const response = await fetch(`/api/analyze/${symbol}?timeframe=${timeframe}`);
            if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);

            const data = await response.json();
            if (!data.success) throw new Error(data.error || 'Ошибка анализа данных');

            return data;
        } catch (error) {
            console.error('Ошибка загрузки данных анализа:', error);
            throw error;
        }
    }

    static async loadMarketStats(symbol) {
        try {
            const response = await fetch(`/api/market-stats/${symbol}`);
            if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);

            const data = await response.json();
            if (!data.success) throw new Error(data.error || 'Ошибка загрузки статистики');

            return data;
        } catch (error) {
            console.error('Ошибка загрузки статистики:', error);
            return null;
        }
    }

    static updateUI(data) {
        if (!data) return;
        this.updatePriceData(data);
        this.updateRecommendation(data.analysis);
        this.updateIndicators(data.analysis);
        this.updatePatterns(data.patterns);
        this.updateProfitPotential(data.profit_potential);
        this.updateMultiTimeframeAnalysis();
        document.getElementById('update-timestamp').textContent = new Date().toLocaleTimeString();
    }

    static updatePriceData(data) {
        const price = data.analysis?.price || 0;
        const change = data.analysis?.price_change || 0;

        document.getElementById('current-price').textContent = `$${this.formatPrice(price)}`;

        const priceChangeElement = document.getElementById('price-change');
        priceChangeElement.textContent = `${change >= 0 ? '+' : ''}${change.toFixed(2)}%`;
        priceChangeElement.className = `price-change ${change >= 0 ? 'positive' : 'negative'}`;

        const changeIcon = priceChangeElement.querySelector('i');
        if (changeIcon) {
            changeIcon.className = change >= 0 ? 'fas fa-arrow-up' : 'fas fa-arrow-down';
        }
    }

    static updateRecommendation(analysis) {
        const recommendationElement = document.getElementById('recommendation');
        const confidenceElement = document.getElementById('confidence-value');
        const confidenceFill = document.getElementById('confidence-fill');

        if (!analysis) {
            recommendationElement.innerHTML = `
                <div class="error-state">
                    <i class="fas fa-exclamation-triangle"></i>
                    <p>Данные недоступны</p>
                </div>
            `;
            return;
        }

        const recommendation = analysis.recommendation || 'НЕЙТРАЛЬНО';
        const score = analysis.score || 50;
        const confidence = analysis.confidence || 'Низкая';

        recommendationElement.innerHTML = `
            <div class="recommendation ${recommendation.toLowerCase()}">
                ${recommendation}
            </div>
            <div class="recommendation-explanation">
                ${this.getRecommendationExplanation(recommendation, analysis)}
            </div>
        `;

        confidenceElement.textContent = `${score}%`;
        confidenceFill.style.width = `${score}%`;
        confidenceFill.style.background = this.getConfidenceColor(score);
    }

    static updateIndicators(analysis) {
        if (!analysis || !analysis.technical_indicators) return;

        const indicators = analysis.technical_indicators;

        // RSI
        if (indicators.rsi) {
            const rsiValue = parseFloat(indicators.rsi);
            document.getElementById('rsi-value').textContent = indicators.rsi;
            document.getElementById('rsi-bar').style.width = `${Math.min(100, Math.max(0, rsiValue))}%`;
            document.getElementById('rsi-bar').style.background = this.getRsiColor(rsiValue);
            document.getElementById('rsi-signal').textContent = this.getRsiSignal(rsiValue);
        }

        // MACD
        if (indicators.macd) {
            const macdValue = parseFloat(indicators.macd.value) || 0;
            const macdSignal = indicators.macd.signal || '';

            document.getElementById('macd-value').textContent = macdValue.toFixed(6);
            document.getElementById('macd-signal').textContent = macdSignal;
            document.getElementById('macd-hint').textContent = this.getMacdHint(macdValue, macdSignal);
        }

        // Bollinger Bands
        if (indicators.bollinger_bands) {
            const position = indicators.bollinger_bands.position || 'Нейтральный';

            document.getElementById('bb-value').textContent = position;
            document.getElementById('bb-signal').textContent = this.getBollingerSignal(position);
            document.getElementById('bb-hint').textContent = this.getBollingerHint(position);
        }

        // Stochastic
        if (indicators.stochastic) {
            const stochK = parseFloat(indicators.stochastic.k) || 0;
            const stochD = parseFloat(indicators.stochastic.d) || 0;

            document.getElementById('stoch-value').textContent = `K: ${stochK.toFixed(2)}, D: ${stochD.toFixed(2)}`;
            document.getElementById('stoch-signal').textContent = this.getStochasticSignal(stochK, stochD);
            document.getElementById('stoch-hint').textContent = this.getStochasticHint(stochK, stochD);
        }
    }

    static updatePatterns(patterns) {
        const container = document.getElementById('patterns-container');

        if (!patterns || patterns.length === 0) {
            container.innerHTML = `
                <div class="empty-state">
                    <i class="fas fa-search"></i>
                    <p>Паттерны не обнаружены</p>
                </div>
            `;
            return;
        }

        container.innerHTML = patterns.map(pattern => `
            <div class="pattern-card">
                <div class="pattern-header">
                    <h4>${pattern.name}</h4>
                    <span class="confidence-badge ${pattern.confidence.toLowerCase()}">
                        ${pattern.confidence}
                    </span>
                </div>
                <div class="pattern-type">Тип: ${pattern.type}</div>
                <p class="pattern-description">${pattern.description}</p>
                ${pattern.easter_egg ? `<div class="pattern-hint">${pattern.easter_egg}</div>` : ''}
            </div>
        `).join('');
    }

    static updateProfitPotential(profitData) {
        if (!profitData) return;

        document.getElementById('profit-value').textContent = profitData.potential_profit || '0%';
        document.getElementById('profit-confidence').textContent = profitData.confidence || 'Низкая';
        document.getElementById('risk-reward').textContent = profitData.risk_reward_ratio || '1:0';
        document.getElementById('stop-loss').textContent = profitData.stop_loss || '-';
        document.getElementById('take-profit').textContent = profitData.take_profit || '-';
        document.getElementById('volatility').textContent = profitData.volatility_based || '-';
    }

    static updateMultiTimeframeAnalysis() {
        // Загрузка анализа для разных таймфреймов
        const timeframes = ['60', '240', 'D', 'W'];
        timeframes.forEach(tf => {
            this.loadTimeframeAnalysis(this.state.currentSymbol, tf);
        });
    }

    static async loadTimeframeAnalysis(symbol, timeframe) {
        try {
            const response = await fetch(`/api/analyze/${symbol}?timeframe=${timeframe}`);
            if (!response.ok) return;

            const data = await response.json();
            if (!data.success) return;

            this.updateTimeframeCard(timeframe, data.analysis);
        } catch (error) {
            console.error(`Ошибка загрузки анализа для таймфрейма ${timeframe}:`, error);
        }
    }

    static updateTimeframeCard(timeframe, analysis) {
        const tfElement = document.getElementById(`tf-${timeframe}`);
        const tfConfElement = document.getElementById(`tf-${timeframe}-conf`);

        if (tfElement && analysis) {
            tfElement.textContent = analysis.recommendation || '-';
            tfElement.className = `tf-signal ${analysis.recommendation ? analysis.recommendation.toLowerCase() : 'neutral'}`;

            if (tfConfElement) {
                tfConfElement.textContent = analysis.confidence || '-';
            }
        }
    }

    static updateAdditionalData(marketStats) {
        if (!marketStats) return;

        // Обновление исторических данных
        if (marketStats.historical_data) {
            const historyBody = document.getElementById('history-data');
            historyBody.innerHTML = marketStats.historical_data.map(item => `
                <tr>
                    <td>${item.period}</td>
                    <td>$${this.formatPrice(item.price)}</td>
                    <td class="${item.change >= 0 ? 'positive' : 'negative'}">
                        ${item.change >= 0 ? '+' : ''}${item.change}%
                    </td>
                </tr>
            `).join('');
        }

        // Обновление статистики рынка
        if (marketStats.market_stats) {
            const stats = marketStats.market_stats;
            document.getElementById('volume-24h').textContent = this.formatNumber(stats.volume_24h) || '-';
            document.getElementById('high-24h').textContent = stats.high_24h ? `$${this.formatPrice(stats.high_24h)}` : '-';
            document.getElementById('low-24h').textContent = stats.low_24h ? `$${this.formatPrice(stats.low_24h)}` : '-';
            document.getElementById('change-24h').textContent = stats.change_24h ? `${stats.change_24h >= 0 ? '+' : ''}${stats.change_24h}%` : '-';
        }
    }

    static async initTradingView() {
        if (typeof TradingView === 'undefined') {
            console.error('TradingView widget not loaded');
            return;
        }

        try {
            this.state.tradingViewWidget = new TradingView.widget({
                symbol: `BINANCE:${this.state.currentSymbol}`,
                interval: this.state.currentTimeframe,
                container_id: "tradingview-chart",
                locale: "ru",
                autosize: true,
                theme: this.state.currentTheme,
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
            console.error('Ошибка инициализации TradingView:', error);
        }
    }

    static updateTradingView() {
        if (this.state.tradingViewWidget) {
            this.state.tradingViewWidget.setSymbol(`BINANCE:${this.state.currentSymbol}`);
            this.state.tradingViewWidget.setInterval(this.state.currentTimeframe);
        }
    }

    static startAutoRefresh() {
        if (this.state.refreshTimer) {
            clearInterval(this.state.refreshTimer);
        }

        this.state.refreshTimer = setInterval(() => {
            if (this.state.autoRefresh) {
                if (this.state.countdown > 0) {
                    this.state.countdown--;
                    document.getElementById('countdown').textContent = this.state.countdown;
                    document.getElementById('progress-fill').style.width =
                        `${(100 - (this.state.countdown / 15 * 100))}%`;
                } else {
                    this.state.countdown = 15;
                    this.refreshData();
                }
            }
        }, 1000);
    }

    static toggleAutoRefresh() {
        const toggleBtn = document.getElementById('toggle-refresh');
        const icon = toggleBtn.querySelector('i');

        if (this.state.autoRefresh) {
            toggleBtn.innerHTML = '<i class="fas fa-play"></i> Возобновить автообновление';
            toggleBtn.style.background = 'var(--accent-success)';
        } else {
            toggleBtn.innerHTML = '<i class="fas fa-pause"></i> Приостановить автообновление';
            toggleBtn.style.background = 'var(--accent-warning)';
        }
    }

    static showLoadingState() {
        document.querySelectorAll('.analysis-content, #recommendation, #patterns-container')
            .forEach(el => {
                el.innerHTML = `
                    <div class="loading-state">
                        <div class="loading-spinner"></div>
                        <p>Загрузка данных...</p>
                    </div>
                `;
            });
    }

    static showError(message) {
        const recommendationElement = document.getElementById('recommendation');
        recommendationElement.innerHTML = `
            <div class="error-state">
                <i class="fas fa-exclamation-triangle"></i>
                <p>${message}</p>
                <button class="btn btn-primary" onclick="AnalysisCore.refreshData()">
                    <i class="fas fa-sync-alt"></i> Попробовать снова
                </button>
            </div>
        `;
    }

    static formatPrice(price) {
        if (!price) return '0.00';
        if (price >= 1000) return price.toFixed(0);
        if (price >= 1) return price.toFixed(2);
        if (price >= 0.01) return price.toFixed(4);
        return price.toFixed(6);
    }

    static formatNumber(num) {
        if (!num) return '0';
        if (num >= 1000000) return (num / 1000000).toFixed(2) + 'M';
        if (num >= 1000) return (num / 1000).toFixed(2) + 'K';
        return num.toFixed(2);
    }

    static getRecommendationExplanation(recommendation, analysis) {
        const explanations = {
            'ПОКУПАТЬ': 'Сильные бычьи сигналы. Рекомендуется рассмотреть покупку.',
            'ПРОДАВАТЬ': 'Сильные медвежьи сигналы. Рекомендуется рассмотреть продажу.',
            'НЕЙТРАЛЬНО': 'Сигналы неоднозначны. Рекомендуется ожидание.'
        };

        return explanations[recommendation] || 'Анализ завершен.';
    }

    static getConfidenceColor(score) {
        if (score >= 70) return 'var(--accent-success)';
        if (score >= 50) return 'var(--accent-warning)';
        return 'var(--accent-error)';
    }

    static getRsiColor(value) {
        if (value > 70) return 'var(--accent-error)';
        if (value < 30) return 'var(--accent-success)';
        return 'var(--accent-warning)';
    }

    static getRsiSignal(value) {
        if (value > 70) return 'Перекупленность';
        if (value < 30) return 'Перепроданность';
        return 'Нейтральный';
    }

    static getMacdHint(value, signal) {
        if (value > 0 && signal === 'Бычий') return 'Сильный бычий сигнал';
        if (value < 0 && signal === 'Медвежий') return 'Сильный медвежий сигнал';
        return 'Сигнал неясен';
    }

    static getBollingerSignal(position) {
        if (position.includes('Перекупленность') || position.includes('Выше')) return 'Медвежий';
        if (position.includes('Перепроданность') || position.includes('Ниже')) return 'Бычий';
        return 'Нейтральный';
    }

    static getBollingerHint(position) {
        if (position.includes('Перекупленность') || position.includes('Выше')) return 'Возможна коррекция вниз';
        if (position.includes('Перепроданность') || position.includes('Ниже')) return 'Возможен отскок вверх';
        return 'Цена в нормальном диапазоне';
    }

    static getStochasticSignal(k, d) {
        if (k > 80 && d > 80) return 'Перекупленность';
        if (k < 20 && d < 20) return 'Перепроданность';
        if (k > d) return 'Бычий сигнал';
        if (k < d) return 'Медвежий сигнал';
        return 'Нейтральный';
    }

    static getStochasticHint(k, d) {
        if (k > 80 && d > 80) return 'Возможна коррекция';
        if (k < 20 && d < 20) return 'Возможен отскок';
        if (k > d) return 'Восходящий момент';
        if (k < d) return 'Нисходящий момент';
        return 'Момент неясен';
    }

    static updateRiskCalculator() {
        const deposit = parseFloat(document.getElementById('deposit-size').value) || 1000;
        const riskPercent = parseFloat(document.getElementById('risk-percent').value) || 2;
        const riskAmount = deposit * (riskPercent / 100);

        if (this.state.currentData && this.state.currentData.analysis) {
            const price = this.state.currentData.analysis.price;
            const stopLossPercent = 3; // Пример: 3% стоп-лосс
            const positionSize = riskAmount / (stopLossPercent / 100);

            document.getElementById('position-size').textContent = `$${positionSize.toFixed(2)}`;
            document.getElementById('max-loss').textContent = `$${riskAmount.toFixed(2)}`;

            // Пример расчета потенциальной прибыли
            const potentialGain = positionSize * 0.05; // 5% прибыль
            document.getElementById('potential-profit').textContent = `$${potentialGain.toFixed(2)}`;
            document.getElementById('rr-ratio').textContent = '1:1.67'; // Пример
        }
    }
}

// Инициализация при загрузке документа
document.addEventListener('DOMContentLoaded', function() {
    AnalysisCore.init();
});


//deepsuck
// Обработчик переключения языка
document.getElementById('langSwitcher')?.addEventListener('change', function(e) {
    const lang = e.target.value;
    // Здесь будет логика смены языка
    console.log('Selected language:', lang);
    // Можно добавить запрос к API для смены языка
});

// Обработчик темы в хедере
document.getElementById('header-theme-toggle')?.addEventListener('click', function() {
    const currentTheme = document.documentElement.getAttribute('data-theme');
    const newTheme = currentTheme === 'dark' ? 'light' : 'dark';

    document.documentElement.setAttribute('data-theme', newTheme);
    localStorage.setItem('cryptoanalyst_theme', newTheme);

    const themeIcon = this.querySelector('i');
    if (themeIcon) {
        themeIcon.className = newTheme === 'dark' ? 'fas fa-sun' : 'fas fa-moon';
    }

    // Обновление TradingView если есть
    if (window.tradingViewWidget) {
        try {
            window.tradingViewWidget.changeTheme(newTheme);
        } catch (error) {
            console.warn('Ошибка обновления темы TradingView:', error);
        }
    }
});

// Инициализация иконки темы
function initThemeIcon() {
    const themeToggle = document.getElementById('header-theme-toggle');
    if (!themeToggle) return;

    const themeIcon = themeToggle.querySelector('i');
    const currentTheme = document.documentElement.getAttribute('data-theme') || 'dark';

    if (themeIcon) {
        themeIcon.className = currentTheme === 'dark' ? 'fas fa-sun' : 'fas fa-moon';
    }
}

// Запуск при загрузке
document.addEventListener('DOMContentLoaded', initThemeIcon);