// static/js/patterns.js
class PatternsManager {
    static init() {
        this.loadPatternsData();
        this.initTheme();
        this.initSearchAndFilters();
        this.initModal();
        this.debugImagePaths(); // Добавляем отладку
    }

    static debugImagePaths() {
    console.log('Current origin:', window.location.origin);

    const testPatterns = ['Двойное дно', 'Голова и плечи', 'Треугольник'];
    testPatterns.forEach(pattern => {
        const url = this.getPatternImage(pattern);
        console.log(`Pattern: ${pattern}, URL: ${url}`);

        // Проверяем доступность изображения
        fetch(url)
            .then(response => {
                console.log(`${pattern}: ${response.status} ${response.statusText}`);
            })
            .catch(error => {
                console.error(`${pattern}: Error -`, error);
            });
    });
}

    static async loadPatternsData() {
        try {
            const container = document.getElementById('patterns-grid');
            container.innerHTML = this.createLoadingState();

            const response = await fetch('/api/patterns');
            if (!response.ok) throw new Error('Ошибка загрузки данных');

            const data = await response.json();

            if (data.success) {
                this.renderPatterns(data.patterns);
            } else {
                throw new Error(data.error || 'Ошибка загрузки данных');
            }
        } catch (error) {
            console.error('Ошибка загрузки паттернов:', error);
            this.showErrorState('Не удалось загрузить данные паттернов');
        }
    }

    static createLoadingState() {
        return `
            <div class="loading-state">
                <div class="loading-spinner"></div>
                <p>Загрузка паттернов...</p>
            </div>
        `;
    }

    static renderPatterns(patterns) {
        const container = document.getElementById('patterns-grid');

        if (!patterns || patterns.length === 0) {
            container.innerHTML = this.createEmptyState('Паттерны не найдены');
            return;
        }

        container.innerHTML = patterns.map(pattern => this.createPatternCard(pattern)).join('');
        this.addPatternCardEventListeners();
    }

    // Добавим метод для получения пути к изображению
static getPatternImage(patternName) {
    const imageMap = {
        'Двойное дно': 'double_bottom.svg',
        'Двойная вершина': 'double_top.svg',
        'Голова и плечи': 'head_shoulders.svg',
        'Перевернутая голова и плечи': 'inverse_head_shoulders.svg',
        'Треугольник': 'triangle.svg',
        'Флаг': 'flag.svg',
        'Вымпел': 'pennant.svg',
        'Клин': 'wedge.svg',
        'Поглощение': 'engulfing.svg',
        'Молот': 'hammer.svg',
        'Падающая звезда': 'shooting_star.svg',
        'Утренняя звезда': 'morning_star.svg',
        'Вечерняя звезда': 'evening_star.svg',
        'Бриша': 'brisha.svg'
    };

    const filename = imageMap[patternName] || 'default_pattern.svg';
    // Используем абсолютный путь
    return `${window.location.origin}/static/images/patterns/${filename}`;
}

// Обновим создание карточки паттерна
static createPatternCard(pattern) {
    const imageUrl = this.getPatternImage(pattern.name);
    const patternData = JSON.stringify(pattern).replace(/"/g, '&quot;');

    return `
        <div class="pattern-card" data-type="${this.getPatternTypeClass(pattern.type)}" data-name="${pattern.name.toLowerCase()}">
            <div class="pattern-header">
                <div class="pattern-icon">
                    <i class="fas fa-chart-line"></i>
                </div>
                <div class="pattern-title">
                    <h3 class="pattern-name">${pattern.name}</h3>
                    <span class="pattern-type-badge ${this.getPatternTypeClass(pattern.type)}">
                        ${pattern.type}
                    </span>
                </div>
            </div>

            <div class="pattern-visual">
                <img src="${imageUrl}" alt="${pattern.name}"
                     onerror="this.style.display='none'; this.nextElementSibling.style.display='flex';"
                     class="pattern-image">
                <div class="image-fallback">
                    <i class="fas fa-image"></i>
                    <span>Изображение паттерна</span>
                </div>
            </div>

            <div class="pattern-body">
                <p class="pattern-description">${pattern.description}</p>

                <div class="pattern-meta">
                    <span class="pattern-reliability">
                        <i class="fas fa-star"></i>
                        <span class="reliability-text">${pattern.confidence}</span>
                    </span>
                    <span class="pattern-trend">
                        <i class="fas fa-arrow-up"></i>
                        <span class="trend-direction">${this.getTrendDirection(pattern.type)}</span>
                    </span>
                </div>
            </div>

            <div class="pattern-actions">
                <button class="pattern-details-btn" data-pattern='${patternData}'>
                    <i class="fas fa-info-circle"></i> Подробнее
                </button>
            </div>
        </div>
    `;
}

    static addPatternCardEventListeners() {
        setTimeout(() => {
            // Обработчик для всей карточки
            document.querySelectorAll('.pattern-card').forEach(card => {
                card.addEventListener('click', (e) => {
                    if (!e.target.closest('.pattern-details-btn')) {
                        const patternName = card.querySelector('.pattern-name').textContent;
                        const patternData = JSON.parse(document.getElementById('patterns-data').textContent);
                        const pattern = patternData.find(p => p.name === patternName);
                        if (pattern) {
                            this.showPatternDetails(pattern);
                        }
                    }
                });
            });

            // Обработчик для кнопок "Подробнее"
            document.querySelectorAll('.pattern-details-btn').forEach(btn => {
                btn.addEventListener('click', (e) => {
                    e.stopPropagation(); // Предотвращаем всплытие
                    const patternData = JSON.parse(btn.dataset.pattern.replace(/&quot;/g, '"'));
                    this.showPatternDetails(patternData);
                });
            });
        }, 100);
    }

    static showPatternDetails(pattern) {
        const modal = document.getElementById('pattern-modal');
        const modalTitle = document.getElementById('modal-title');
        const modalBody = document.getElementById('modal-body');

        modalTitle.textContent = pattern.name;
        modalBody.innerHTML = this.createPatternModalContent(pattern);

        modal.style.display = 'flex';
        document.body.style.overflow = 'hidden';

        setTimeout(() => {
            modal.style.opacity = '1';
            modal.querySelector('.modal-content').style.transform = 'translateY(0) scale(1)';
        }, 10);
    }

    static createPatternModalContent(pattern) {
    const imageUrl = this.getPatternImage(pattern.name);

    return `
        <div class="pattern-details">
            <div class="detail-section">
                <img src="${imageUrl}" alt="${pattern.name}"
                     onerror="this.src='/static/images/patterns/default_pattern.svg'"
                     class="modal-pattern-image">
            </div>

            <div class="detail-section">
                <h4><i class="fas fa-info-circle"></i> Описание</h4>
                <p class="detail-description">${pattern.description}</p>
            </div>

            <div class="detail-section">
                <h4><i class="fas fa-bullseye"></i> Торговые сигналы</h4>
                <div class="signals-grid">
                    <div class="signal-item">
                        <i class="fas fa-arrow-right"></i>
                        <strong>Вход:</strong> <span class="signal-entry">${this.getEntrySignal(pattern.type)}</span>
                    </div>
                    <div class="signal-item">
                        <i class="fas fa-shield-alt"></i>
                        <strong>Стоп-лосс:</strong> <span class="signal-stop">${this.getStopSignal(pattern.type)}</span>
                    </div>
                    <div class="signal-item">
                        <i class="fas fa-target"></i>
                        <strong>Цель:</strong> <span class="signal-target">${this.getTargetSignal(pattern.type)}</span>
                    </div>
                </div>
            </div>

            <div class="detail-section">
                <h4><i class="fas fa-chart-bar"></i> Особенности</h4>
                <ul class="pattern-features">
                    ${this.getPatternFeatures(pattern).map(feature => `<li>${feature}</li>`).join('')}
                </ul>
            </div>

            <div class="detail-section">
                <h4><i class="fas fa-lightbulb"></i> Советы по торговле</h4>
                <div class="trading-tips">
                    ${this.getTradingTips(pattern).map(tip => `<div>${tip}</div>`).join('')}
                </div>
            </div>

            <div class="detail-section">
                <h4><i class="fas fa-exclamation-triangle"></i> Что искать</h4>
                <div class="key-points">
                    ${this.getKeyPoints(pattern).map(point => `<div>${point}</div>`).join('')}
                </div>
            </div>

            ${pattern.easter_egg ? `
            <div class="detail-section">
                <h4><i class="fas fa-key"></i> Секрет успеха</h4>
                <div class="key-points">
                    <div>${pattern.easter_egg}</div>
                </div>
            </div>
            ` : ''}
        </div>
    `;
}

    // Все вспомогательные методы (getPatternTypeClass, getTrendDirection, и т.д.)
    static getPatternTypeClass(type) {
        const typeMap = {
            'Разворотный': 'reversal',
            'Продолжения': 'continuation',
            'Свечной': 'candlestick'
        };
        return typeMap[type] || 'reversal';
    }

    static getTrendDirection(type) {
        return type === 'Продолжения' ? 'Следование тренду' : 'Разворот тренда';
    }

    static getEntrySignal(type) {
        const signals = {
            'Разворотный': 'После подтверждения разворота и пробоя ключевого уровня',
            'Продолжения': 'На откате к поддержке/сопротивлению с подтверждением объема',
            'Свечной': 'После закрытия свечи подтверждения следующей свечой'
        };
        return signals[type] || 'После подтверждения паттерна и confluence факторов';
    }

    static getStopSignal(type) {
        const stops = {
            'Разворотный': '2-3% за ключевым уровнем поддержки/сопротивления',
            'Продолжения': '1-2% за локальным минимумом/максимумом',
            'Свечной': 'За экстремумом свечи подтверждения'
        };
        return stops[type] || '2-3% за ключевым уровнем';
    }

    static getTargetSignal(type) {
        const targets = {
            'Разворотный': '1:2-1:3 риск/вознаграждение, измерение от линии шеи',
            'Продолжения': '1:1.5-1:2 риск/вознаграждение, продолжение тренда',
            'Свечной': '1:1-1:1.5 риск/вознаграждение, ближайшие уровни'
        };
        return targets[type] || '1:1.5 риск/вознаграждение';
    }

    static getPatternFeatures(pattern) {
        return [
            `Надежность: ${pattern.confidence}`,
            `Тип: ${pattern.type}`,
            `Вероятность успеха: ${this.getSuccessProbability(pattern.confidence)}`,
            `Рекомендуемый таймфрейм: ${this.getRecommendedTimeframe(pattern.type)}`,
            pattern.easter_egg ? `Ключевой момент: ${pattern.easter_egg}` : 'Требует подтверждения объемом'
        ];
    }

    static getSuccessProbability(confidence) {
        const probabilities = {
            'Высокая': '70-80%',
            'Средняя': '50-60%',
            'Низкая': '30-40%'
        };
        return probabilities[confidence] || '50-60%';
    }

    static getRecommendedTimeframe(type) {
        const timeframes = {
            'Разворотный': '4H - Daily',
            'Продолжения': '1H - 4H',
            'Свечной': '15M - 1H'
        };
        return timeframes[type] || '1H - 4H';
    }

    static getTradingTips(pattern) {
        return [
            'Дождитесь полного формирования паттерна',
            'Ищите подтверждение объемами (увеличение на пробое)',
            'Используйте дополнительные индикаторы (RSI, MACD) для подтверждения',
            'Устанавливайте стоп-лосс сразу при входе в сделку',
            'Фиксируйте часть прибыли на первом целевом уровне',
            'Торгуйте только при наличии четких сигналов'
        ];
    }

    static getKeyPoints(pattern) {
        return [
            'Четкое графическое формирование по классическому описанию',
            'Соответствие текущему рыночному контексту и тренду',
            'Подтверждение объемом торгов на ключевых моментах',
            'Отсутствие конфликтующих сигналов от индикаторов',
            'Соответствие паттерна таймфрейму (большие TF = надежнее)'
        ];
    }

    static initSearchAndFilters() {
        const searchInput = document.getElementById('pattern-search');
        const filterButtons = document.querySelectorAll('.filter-btn');
        const viewButtons = document.querySelectorAll('.view-btn');

        searchInput.addEventListener('input', (e) => {
            this.filterPatterns(e.target.value.toLowerCase());
        });

        filterButtons.forEach(btn => {
            btn.addEventListener('click', () => {
                filterButtons.forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                this.applyFilter(btn.dataset.filter);
            });
        });

        viewButtons.forEach(btn => {
            btn.addEventListener('click', () => {
                viewButtons.forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                this.changeView(btn.dataset.view);
            });
        });
    }

    static filterPatterns(searchTerm) {
        const patterns = document.querySelectorAll('.pattern-card');
        let visibleCount = 0;

        patterns.forEach(pattern => {
            const patternName = pattern.querySelector('.pattern-name').textContent.toLowerCase();
            const patternDescription = pattern.querySelector('.pattern-description').textContent.toLowerCase();
            const isVisible = patternName.includes(searchTerm) ||
                             patternDescription.includes(searchTerm) ||
                             searchTerm === '';

            pattern.style.display = isVisible ? 'block' : 'none';
            if (isVisible) visibleCount++;
        });

        if (visibleCount === 0 && searchTerm !== '') {
            document.getElementById('patterns-grid').innerHTML = this.createEmptyState(`Ничего не найдено для "${searchTerm}"`);
        }
    }

    static applyFilter(filterType) {
        const patterns = document.querySelectorAll('.pattern-card');
        let visibleCount = 0;

        patterns.forEach(pattern => {
            const patternType = pattern.dataset.type;
            const isVisible = filterType === 'all' || patternType === filterType;

            pattern.style.display = isVisible ? 'block' : 'none';
            if (isVisible) visibleCount++;
        });

        if (visibleCount === 0 && filterType !== 'all') {
            document.getElementById('patterns-grid').innerHTML = this.createEmptyState(`Нет паттернов типа "${filterType}"`);
        }
    }

    static changeView(viewType) {
        const grid = document.getElementById('patterns-grid');
        grid.classList.toggle('list-view', viewType === 'list');
    }

    static initModal() {
        const modal = document.getElementById('pattern-modal');
        const closeBtn = document.getElementById('modal-close');

        modal.style.opacity = '0';
        modal.querySelector('.modal-content').style.transform = 'translateY(20px) scale(0.95)';
        modal.style.transition = 'opacity 0.3s ease';
        modal.querySelector('.modal-content').style.transition = 'transform 0.3s ease';

        closeBtn.addEventListener('click', () => this.closeModal());

        modal.addEventListener('click', (e) => {
            if (e.target === modal) this.closeModal();
        });

        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && modal.style.display === 'flex') {
                this.closeModal();
            }
        });
    }

    static closeModal() {
        const modal = document.getElementById('pattern-modal');

        modal.style.opacity = '0';
        modal.querySelector('.modal-content').style.transform = 'translateY(20px) scale(0.95)';

        setTimeout(() => {
            modal.style.display = 'none';
            document.body.style.overflow = 'auto';
        }, 300);
    }

    static createEmptyState(message) {
        return `
            <div class="empty-state">
                <i class="fas fa-search"></i>
                <p>${message}</p>
            </div>
        `;
    }

    static showErrorState(message) {
        const container = document.getElementById('patterns-grid');
        container.innerHTML = `
            <div class="error-state">
                <i class="fas fa-exclamation-triangle"></i>
                <p>${message}</p>
                <button class="btn btn-primary" onclick="PatternsManager.loadPatternsData()">
                    <i class="fas fa-sync-alt"></i> Попробовать снова
                </button>
            </div>
        `;
    }

    static initTheme() {
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
            });
        }

        // Загрузка сохраненной темы
        const savedTheme = localStorage.getItem('cryptoanalyst_theme') || 'dark';
        document.documentElement.setAttribute('data-theme', savedTheme);

        const themeIcon = document.querySelector('#theme-toggle i');
        if (themeIcon) {
            themeIcon.className = savedTheme === 'dark' ? 'fas fa-sun' : 'fas fa-moon';
        }
    }
}

// Инициализация после загрузки DOM
document.addEventListener('DOMContentLoaded', function() {
    PatternsManager.init();
});

// Делаем метод доступным глобально для обработки ошибок
window.showPatternDetails = function(patternData) {
    PatternsManager.showPatternDetails(patternData);
};