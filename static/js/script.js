// ===== КОНФИГУРАЦИЯ ПРИЛОЖЕНИЯ =====
const AppConfig = {
    refreshInterval: 5000, // 5 секунд для обновления данных (более безопасно)
    apiTimeout: 10000, // 10 секунд таймаут для API запросов
    maxRetries: 2, // Максимальное количество попыток переподключения
    themeKey: 'cryptoanalyst_theme', // Ключ для хранения темы в localStorage
    symbolsKey: 'cryptoanalyst_symbols', // Ключ для кэширования символов
    cacheTime: 300000, // 5 минут кэширования данных
};

// ===== ГЛОБАЛЬНОЕ СОСТОЯНИЕ ПРИЛОЖЕНИЯ =====
const AppState = {
    currentTheme: 'dark',
    isOnline: navigator.onLine,
    isLoading: false,
    lastUpdate: null,
    activeRequests: new Set(),
    refreshIntervals: new Map(),
    eventListeners: new Map()
};

// ===== КЛАСС ДЛЯ УПРАВЛЕНИЯ ТЕМОЙ =====
class ThemeManager {
    static init() {
        this.loadTheme();
        this.setupEventListeners();
        this.applyThemeToTradingView();
    }

    static loadTheme() {
        const savedTheme = localStorage.getItem(AppConfig.themeKey);
        const systemTheme = window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
        AppState.currentTheme = savedTheme || systemTheme;

        document.documentElement.setAttribute('data-theme', AppState.currentTheme);
        this.updateThemeIcon();
    }

    static toggleTheme() {
        AppState.currentTheme = AppState.currentTheme === 'dark' ? 'light' : 'dark';
        document.documentElement.setAttribute('data-theme', AppState.currentTheme);
        localStorage.setItem(AppConfig.themeKey, AppState.currentTheme);

        this.updateThemeIcon();
        this.applyThemeToTradingView();
        EventManager.dispatchEvent('themeChanged', { theme: AppState.currentTheme });
    }

    static updateThemeIcon() {
        const themeIcon = document.querySelector('#theme-toggle i');
        if (themeIcon) {
            themeIcon.className = AppState.currentTheme === 'dark' ? 'fas fa-sun' : 'fas fa-moon';
        }
    }

    static applyThemeToTradingView() {
        // Асинхронное обновление TradingView с задержкой для стабильности
        setTimeout(() => {
            if (window.tradingViewWidget && typeof window.tradingViewWidget.changeTheme === 'function') {
                try {
                    window.tradingViewWidget.changeTheme(AppState.currentTheme === 'dark' ? 'dark' : 'light');
                } catch (error) {
                    console.warn('Ошибка обновления темы TradingView:', error);
                }
            }
        }, 100);
    }

    static setupEventListeners() {
        const themeToggle = document.getElementById('theme-toggle');
        if (themeToggle) {
            // Исправлено: используем EventManager вместо this.addEventListener
            EventManager.addEventListener(themeToggle, 'click', () => this.toggleTheme());
        }
    }
}

// ===== КЛАСС ДЛЯ УПРАВЛЕНИЯ API ЗАПРОСАМИ =====
class ApiManager {
    static async request(endpoint, options = {}) {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), AppConfig.apiTimeout);

        const requestId = Symbol('request');
        AppState.activeRequests.add(requestId);

        try {
            const response = await fetch(endpoint, {
                ...options,
                signal: controller.signal,
                headers: {
                    'Content-Type': 'application/json',
                    ...options.headers
                }
            });

            clearTimeout(timeoutId);

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            return await response.json();
        } catch (error) {
            if (error.name === 'AbortError') {
                throw new Error('Таймаут запроса');
            }
            throw error;
        } finally {
            clearTimeout(timeoutId);
            AppState.activeRequests.delete(requestId);
        }
    }

    static async getSymbols() {
        const cacheKey = `${AppConfig.symbolsKey}_list`;
        const cached = this.getCachedData(cacheKey);

        if (cached) {
            return cached;
        }

        try {
            const data = await this.request('/api/symbols');
            this.cacheData(cacheKey, data, AppConfig.cacheTime);
            return data;
        } catch (error) {
            console.error('Ошибка загрузки символов:', error);
            return { symbols: ['BTCUSDT', 'ETHUSDT', 'BNBUSDT'] };
        }
    }

    static async getTopCryptos() {
        try {
            return await this.request('/api/top-cryptos');
        } catch (error) {
            console.error('Ошибка загрузки топовых крипто:', error);
            throw error;
        }
    }

    static cacheData(key, data, ttl) {
        const item = {
            data,
            expiry: Date.now() + ttl
        };
        localStorage.setItem(key, JSON.stringify(item));
    }

    static getCachedData(key) {
        const itemStr = localStorage.getItem(key);
        if (!itemStr) return null;

        const item = JSON.parse(itemStr);
        if (Date.now() > item.expiry) {
            localStorage.removeItem(key);
            return null;
        }

        return item.data;
    }
}

// ===== КЛАСС ДЛЯ УПРАВЛЕНИЯ ОБНОВЛЕНИЕМ ДАННЫХ =====
class DataRefreshManager {
    static intervals = new Map();

    static startRefresh(key, callback, interval = AppConfig.refreshInterval) {
        this.stopRefresh(key);

        const intervalId = setInterval(async () => {
            if (AppState.isLoading || !AppState.isOnline) return;

            try {
                await callback();
            } catch (error) {
                console.warn(`Ошибка обновления ${key}:`, error);
            }
        }, interval);

        this.intervals.set(key, intervalId);
        return intervalId;
    }

    static stopRefresh(key) {
        if (this.intervals.has(key)) {
            clearInterval(this.intervals.get(key));
            this.intervals.delete(key);
        }
    }

    static stopAll() {
        this.intervals.forEach((intervalId, key) => {
            clearInterval(intervalId);
            this.intervals.delete(key);
        });
    }
}

// ===== КЛАСС ДЛЯ УПРАВЛЕНИЯ УВЕДОМЛЕНИЯМИ =====
class NotificationManager {
    static show(message, type = 'info', duration = 5000) {
        this.removeExistingNotifications();

        const notification = this.createNotification(message, type);
        document.body.appendChild(notification);

        setTimeout(() => {
            notification.classList.add('show');
        }, 100);

        setTimeout(() => {
            this.hideNotification(notification);
        }, duration);

        return notification;
    }

    static createNotification(message, type) {
        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        notification.innerHTML = `
            <div class="notification-content">
                <i class="fas ${this.getNotificationIcon(type)}"></i>
                <span>${message}</span>
            </div>
            <button class="notification-close">
                <i class="fas fa-times"></i>
            </button>
        `;

        // Добавляем обработчик для кнопки закрытия
        const closeBtn = notification.querySelector('.notification-close');
        if (closeBtn) {
            EventManager.addEventListener(closeBtn, 'click', () => this.hideNotification(notification));
        }

        return notification;
    }

    static getNotificationIcon(type) {
        const icons = {
            success: 'fa-check-circle',
            error: 'fa-exclamation-circle',
            warning: 'fa-exclamation-triangle',
            info: 'fa-info-circle'
        };
        return icons[type] || 'fa-info-circle';
    }

    static hideNotification(notification) {
        notification.classList.remove('show');
        setTimeout(() => {
            if (notification.parentNode) {
                notification.parentNode.removeChild(notification);
            }
        }, 300);
    }

    static removeExistingNotifications() {
        document.querySelectorAll('.notification').forEach(notification => {
            this.hideNotification(notification);
        });
    }
}

// ===== КЛАСС ДЛЯ УПРАВЛЕНИЯ СОСТОЯНИЯМИ UI =====
class UIStateManager {
    static setLoading(state, element = null) {
        AppState.isLoading = state;

        if (element) {
            element.classList.toggle('loading', state);
            const buttons = element.querySelectorAll('button');
            buttons.forEach(btn => {
                btn.disabled = state;
                if (state) {
                    btn.dataset.originalText = btn.innerHTML;
                    btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Загрузка...';
                } else {
                    btn.innerHTML = btn.dataset.originalText || btn.innerHTML;
                }
            });
        }

        document.documentElement.classList.toggle('loading', state);
    }

    static showError(message, element) {
        if (element) {
            element.innerHTML = `
                <div class="error-state">
                    <i class="fas fa-exclamation-triangle"></i>
                    <p>${message}</p>
                    <button class="btn btn-secondary">
                        <i class="fas fa-redo"></i> Перезагрузить
                    </button>
                </div>
            `;

            // Добавляем обработчик для кнопки перезагрузки
            const reloadBtn = element.querySelector('.btn');
            if (reloadBtn) {
                EventManager.addEventListener(reloadBtn, 'click', () => location.reload());
            }
        }
        NotificationManager.show(message, 'error');
    }

    static updateLastUpdateTime() {
        AppState.lastUpdate = new Date();
        const timeElements = document.querySelectorAll('.last-update-time');
        timeElements.forEach(el => {
            el.textContent = AppState.lastUpdate.toLocaleTimeString();
        });
    }
}

// ===== КЛАСС ДЛЯ РАБОТЫ С КРИПТОДАННЫМИ =====
class CryptoDataManager {
    static async loadTopCryptos() {
        const container = document.getElementById('crypto-dashboard');
        if (!container) return;

        UIStateManager.setLoading(true, container);

        try {
            const data = await ApiManager.getTopCryptos();

            if (data && data.success) {
                this.renderCryptoDashboard(data.cryptos, container);
                UIStateManager.updateLastUpdateTime();
                NotificationManager.show('Данные обновлены', 'success', 2000);
            } else {
                throw new Error(data?.error || 'Ошибка загрузки данных');
            }
        } catch (error) {
            UIStateManager.showError('Ошибка загрузки данных: ' + error.message, container);
        } finally {
            UIStateManager.setLoading(false, container);
        }
    }

    static renderCryptoDashboard(cryptos, container) {
        if (!cryptos || cryptos.length === 0) {
            container.innerHTML = '<div class="empty-state">Нет данных для отображения</div>';
            return;
        }

        container.innerHTML = cryptos.map(crypto => `
            <div class="crypto-card" data-symbol="${crypto.symbol}">
                <div class="crypto-header">
                    <i class="${this.getCryptoIcon(crypto.symbol)}"></i>
                    <h4>${crypto.symbol.replace('USDT', '')}</h4>
                </div>
                <div class="crypto-price ${crypto.price_change >= 0 ? 'price-up' : 'price-down'}">
                    $${this.formatPrice(crypto.price)}
                </div>
                <div class="crypto-change ${crypto.price_change >= 0 ? 'change-up' : 'change-down'}">
                    <i class="fas ${crypto.price_change >= 0 ? 'fa-arrow-up' : 'fa-arrow-down'}"></i>
                    ${Math.abs(crypto.price_change).toFixed(2)}%
                </div>
                <a href="/analyze?symbol=${crypto.symbol}" class="btn btn-primary">
                    <i class="fas fa-chart-bar"></i> Анализ
                </a>
            </div>
        `).join('');

        this.setupCryptoCardEvents();
    }

    static getCryptoIcon(symbol) {
        const iconMap = {
            'BTC': 'fab fa-bitcoin',
            'ETH': 'fab fa-ethereum',
            'BNB': 'fas fa-coins',
            'SOL': 'fas fa-sun',
            'XRP': 'fas fa-bolt',
            'ADA': 'fas fa-chart-line',
            'DOGE': 'fas fa-dog',
            'MATIC': 'fas fa-shapes',
            'DOT': 'fas fa-circle',
            'LTC': 'fas fa-money-bill-wave'
        };
        const baseSymbol = symbol.replace('USDT', '');
        return iconMap[baseSymbol] || 'fas fa-coins';
    }

    static formatPrice(price) {
        if (price >= 1000) return price.toFixed(0);
        if (price >= 1) return price.toFixed(2);
        if (price >= 0.01) return price.toFixed(4);
        return price.toFixed(6);
    }

    static setupCryptoCardEvents() {
        document.querySelectorAll('.crypto-card').forEach(card => {
            EventManager.addEventListener(card, 'click', (e) => {
                if (!e.target.closest('a')) {
                    const symbol = card.dataset.symbol;
                    window.location.href = `/analyze?symbol=${symbol}`;
                }
            });
        });
    }
}

// ===== КЛАСС ДЛЯ УПРАВЛЕНИЯ СОБЫТИЯМИ =====
class EventManager {
    static addEventListener(element, event, handler, options = {}) {
        element.addEventListener(event, handler, options);
        const eventKey = `${event}-${Math.random().toString(36).substr(2, 9)}`;
        AppState.eventListeners.set(eventKey, { element, event, handler, options });
        return eventKey;
    }

    static removeEventListener(eventKey) {
        const listener = AppState.eventListeners.get(eventKey);
        if (listener) {
            listener.element.removeEventListener(listener.event, listener.handler, listener.options);
            AppState.eventListeners.delete(eventKey);
        }
    }

    static removeAllEventListeners() {
        AppState.eventListeners.forEach((listener, key) => {
            listener.element.removeEventListener(listener.event, listener.handler, listener.options);
            AppState.eventListeners.delete(key);
        });
    }

    static dispatchEvent(eventName, detail = {}) {
        const event = new CustomEvent(eventName, { detail });
        document.dispatchEvent(event);
    }
}

// ===== ОСНОВНАЯ ИНИЦИАЛИЗАЦИЯ ПРИЛОЖЕНИЯ =====
class AppInitializer {
    static async init() {
        try {
            console.log('🚀 Инициализация CryptoAnalyst Pro...');

            // 1. Инициализация темы
            ThemeManager.init();

            // 2. Проверка соединения
            this.checkConnection();

            // 3. Настройка обработчиков событий
            this.setupGlobalEventListeners();

            // 4. Загрузка начальных данных
            await this.loadInitialData();

            // 5. Запуск автоматического обновления
            this.startAutoRefresh();

            console.log('✅ Приложение успешно инициализировано');

        } catch (error) {
            console.error('❌ Ошибка инициализации приложения:', error);
            NotificationManager.show('Ошибка загрузки приложения', 'error');
        }
    }

    static checkConnection() {
        AppState.isOnline = navigator.onLine;
        document.documentElement.classList.toggle('online', AppState.isOnline);
        document.documentElement.classList.toggle('offline', !AppState.isOnline);

        if (!AppState.isOnline) {
            NotificationManager.show('Отсутствует интернет-соединение', 'warning');
        }
    }

    static setupGlobalEventListeners() {
        // События сети
        EventManager.addEventListener(window, 'online', () => {
            AppState.isOnline = true;
            document.documentElement.classList.add('online');
            document.documentElement.classList.remove('offline');
            NotificationManager.show('Соединение восстановлено', 'success');
            this.startAutoRefresh();
        });

        EventManager.addEventListener(window, 'offline', () => {
            AppState.isOnline = false;
            document.documentElement.classList.add('offline');
            document.documentElement.classList.remove('online');
            NotificationManager.show('Потеряно интернет-соединение', 'error');
            DataRefreshManager.stopAll();
        });

        // Глобальные горячие клавиши
        EventManager.addEventListener(document, 'keydown', (e) => {
            // Ctrl+R - обновление данных
            if (e.ctrlKey && e.key === 'r') {
                e.preventDefault();
                CryptoDataManager.loadTopCryptos();
            }

            // F1 - справка
            if (e.key === 'F1') {
                e.preventDefault();
                NotificationManager.show('CryptoAnalyst Pro - Платформа технического анализа', 'info');
            }
        });

        // Обработка видимости страницы
        EventManager.addEventListener(document, 'visibilitychange', () => {
            if (document.hidden) {
                DataRefreshManager.stopAll();
            } else {
                this.startAutoRefresh();
            }
        });
    }

    static async loadInitialData() {
        // Загрузка топовых криптовалют если есть контейнер
        if (document.getElementById('crypto-dashboard')) {
            await CryptoDataManager.loadTopCryptos();
        }
    }

    static startAutoRefresh() {
        // Обновление криптоданных каждые 5 секунд
        if (document.getElementById('crypto-dashboard')) {
            DataRefreshManager.startRefresh('cryptoData',
                () => CryptoDataManager.loadTopCryptos(),
                AppConfig.refreshInterval
            );
        }
    }
}

// ===== ГЛОБАЛЬНЫЕ ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ =====
const Utils = {
    debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    },

    throttle(func, limit) {
        let inThrottle;
        return function(...args) {
            if (!inThrottle) {
                func.apply(this, args);
                inThrottle = true;
                setTimeout(() => inThrottle = false, limit);
            }
        };
    },

    formatNumber(number, decimals = 2) {
        return new Intl.NumberFormat('ru-RU', {
            minimumFractionDigits: decimals,
            maximumFractionDigits: decimals
        }).format(number);
    },

    formatPercent(value) {
        return `${value >= 0 ? '+' : ''}${value.toFixed(2)}%`;
    }
};

// ===== ЭКСПОРТ ГЛОБАЛЬНЫХ ФУНКЦИЙ ДЛЯ HTML =====
window.App = {
    init: () => AppInitializer.init(),
    refreshData: () => CryptoDataManager.loadTopCryptos(),
    toggleTheme: () => ThemeManager.toggleTheme(),
    showNotification: (message, type) => NotificationManager.show(message, type),
    utils: Utils
};

// ===== ИНИЦИАЛИЗАЦИЯ ПРИ ЗАГРУЗКЕ ДОКУМЕНТА =====
document.addEventListener('DOMContentLoaded', () => {
    // Отложенная инициализация для полной загрузки DOM
    setTimeout(() => {
        AppInitializer.init().catch(console.error);
    }, 100);
});

// ===== ОБРАБОТКА ОШИБОК =====
EventManager.addEventListener(window, 'error', (event) => {
    console.error('Глобальная ошибка:', event.error);
    NotificationManager.show('Произошла непредвиденная ошибка', 'error');
});

EventManager.addEventListener(window, 'unhandledrejection', (event) => {
    console.error('Необработанный Promise:', event.reason);
    NotificationManager.show('Ошибка выполнения операции', 'error');
});

// ===== СОХРАНЕНИЕ СОСТОЯНИЯ ПРИ ЗАКРЫТИИ =====
EventManager.addEventListener(window, 'beforeunload', () => {
    DataRefreshManager.stopAll();
    EventManager.removeAllEventListeners();
});