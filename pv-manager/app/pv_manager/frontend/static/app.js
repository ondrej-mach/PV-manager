'use strict';

let numberFmt1 = new Intl.NumberFormat(undefined, { maximumFractionDigits: 1, minimumFractionDigits: 0 });
let numberFmt2 = new Intl.NumberFormat(undefined, { maximumFractionDigits: 2, minimumFractionDigits: 0 });
let currencyFmt = new Intl.NumberFormat(undefined, { style: 'currency', currency: 'EUR', maximumFractionDigits: 2 });

let activeLocale = navigator.language || undefined;
let haTimeZone;
let timeFormatter;
let dateTimeFormatter;

let forecastChart;
let planChart;
let gridChart;
let deferrableLoadChart;
let priceImputedFlags = [];
let showingSettings = false;
let statisticsCatalog = [];
let haEntitiesSettings = null;
let haEntitiesBusy = false;
let entityCatalog = [];
let powerEntityCatalog = [];
let batteryEntityCatalog = [];
let entityLookup = Object.create(null);
let entityModalTarget = null;
let lastEntityTrigger = null;
let lastTrainingRunning = false;
let lastCycleRunning = false;
let autoRefreshTimer = null;
let controlSaveTimer = null;

function scheduleControlSave() {
    if (controlSaveTimer) clearTimeout(controlSaveTimer);
    setHAEntitiesMessage('Saving changes...', 'pending');
    controlSaveTimer = setTimeout(saveControlSettings, 1000);
}
let pendingForecastRefresh = null;
let lastForecastTimestamp = null; // optimizations: only fetch forecast when changed
let refreshAllPromise = null;
const TARIFF_MODES = ['constant', 'spot_plus_constant', 'dual_rate'];
const houseTrigger = document.getElementById('houseEntityTrigger');
const pvTrigger = document.getElementById('pvEntityTrigger');
const batteryTrigger = document.getElementById('batterySocTrigger');
const houseLabel = document.getElementById('houseEntityLabel');
const pvLabel = document.getElementById('pvEntityLabel');
const batteryLabel = document.getElementById('batterySocLabel');
const houseHint = document.getElementById('houseHint');
const pvHint = document.getElementById('pvHint');
const batteryHint = document.getElementById('batterySocHint');
const haEntitiesMessage = document.getElementById('controlMessage');
const batteryMessage = document.getElementById('batteryMessage');
const batteryWearInput = document.getElementById('batteryWearInput');
const batteryWearHint = document.getElementById('batteryWearHint');
const batterySocMinInput = document.getElementById('batterySocMinInput');
const batterySocMinHint = document.getElementById('batterySocMinHint');
const batterySocMaxInput = document.getElementById('batterySocMaxInput');
const batterySocMaxHint = document.getElementById('batterySocMaxHint');
const batteryCapacityInput = document.getElementById('batteryCapacityInput');
const batteryCapacityHint = document.getElementById('batteryCapacityHint');
const batteryPowerInput = document.getElementById('batteryPowerInput');
const batteryPowerHint = document.getElementById('batteryPowerHint');
const BATTERY_CAPACITY_HINT_TEXT = 'Defines how much energy the optimization can store in the battery.';
const BATTERY_POWER_HINT_TEXT = 'Set to the inverter\'s continuous charge or discharge power limit.';
const BATTERY_WEAR_HINT_TEXT = 'Applied as a throughput penalty in the optimization to limit excessive cycling.';
const BATTERY_SOC_MIN_HINT_TEXT = 'Keep at least this much charge reserved for resiliency.';
const BATTERY_SOC_MAX_HINT_TEXT = 'Upper bound on usable charge to limit battery wear.';
const batteryFieldConfigs = {
    capacity_kwh: {
        input: batteryCapacityInput,
        hintEl: batteryCapacityHint,
        hintText: BATTERY_CAPACITY_HINT_TEXT,
        min: 0.0,
        minInclusive: false,
        pendingMessage: 'Saving battery capacity…',
        invalidMessage: 'Battery capacity must be greater than zero.',
        invalidHint: 'Enter a capacity greater than zero to reflect your battery pack.',
    },
    power_limit_kw: {
        input: batteryPowerInput,
        hintEl: batteryPowerHint,
        hintText: BATTERY_POWER_HINT_TEXT,
        min: 0.0,
        minInclusive: false,
        pendingMessage: 'Saving power limit…',
        invalidMessage: 'Charge / discharge limit must be greater than zero.',
        invalidHint: 'Enter the inverter\'s continuous charge or discharge power in kW.',
    },
    wear_cost_eur_per_kwh: {
        input: batteryWearInput,
        hintEl: batteryWearHint,
        hintText: BATTERY_WEAR_HINT_TEXT,
        min: 0.0,
        minInclusive: true,
        pendingMessage: 'Saving wear cost…',
        invalidMessage: 'Battery wear cost must be a non-negative number.',
        invalidHint: 'Enter a non-negative number to penalize battery throughput.',
    },
    soc_min: {
        input: batterySocMinInput,
        hintEl: batterySocMinHint,
        hintText: BATTERY_SOC_MIN_HINT_TEXT,
        min: 0.0,
        minInclusive: true,
        max: 0.98,
        maxInclusive: true,
        formatPercent: true,
        epsilon: 0.001,
        pendingMessage: 'Saving reserve floor…',
        invalidMessage: 'Reserve floor must be between 0% and 98% and below the maximum.',
        invalidHint: 'Reserve floor must stay below the usable maximum and between 0% and 98%.',
        extraValidate: (value) => {
            ensureBatterySettings();
            if (value >= batterySettings.soc_max - 0.01) {
                return 'Reserve floor must stay below the maximum.';
            }
            return null;
        },
    },
    soc_max: {
        input: batterySocMaxInput,
        hintEl: batterySocMaxHint,
        hintText: BATTERY_SOC_MAX_HINT_TEXT,
        min: 0.02,
        minInclusive: true,
        max: 1.0,
        maxInclusive: true,
        formatPercent: true,
        epsilon: 0.001,
        pendingMessage: 'Saving maximum SoC…',
        invalidMessage: 'Maximum SoC must be between 2% and 100% and above the reserve floor.',
        invalidHint: 'Maximum SoC must be between 2% and 100% and stay above the reserve floor.',
        extraValidate: (value) => {
            ensureBatterySettings();
            if (value <= batterySettings.soc_min + 0.01) {
                return 'Maximum SoC must stay above the reserve floor.';
            }
            return null;
        },
    },
};
const pricingForm = document.getElementById('pricingForm');
const pricingMessage = document.getElementById('pricingMessage');
const importTariffModeSelect = document.getElementById('importTariffMode');
const exportTariffModeSelect = document.getElementById('exportTariffMode');
const importTariffHint = document.getElementById('importTariffHint');
const exportTariffHint = document.getElementById('exportTariffHint');
const importConstantInput = document.getElementById('importConstantPrice');
const importSpotOffsetInput = document.getElementById('importSpotOffset');
const importPeakRateInput = document.getElementById('importPeakRate');
const importOffpeakRateInput = document.getElementById('importOffpeakRate');
const importPeakStartInput = document.getElementById('importPeakStart');
const importPeakEndInput = document.getElementById('importPeakEnd');
const exportConstantInput = document.getElementById('exportConstantPrice');
const exportSpotOffsetInput = document.getElementById('exportSpotOffset');
const exportPeakRateInput = document.getElementById('exportPeakRate');
const exportOffpeakRateInput = document.getElementById('exportOffpeakRate');
const exportPeakStartInput = document.getElementById('exportPeakStart');
const exportPeakEndInput = document.getElementById('exportPeakEnd');
const tariffFieldBindings = {
    import: [
        { key: 'constant_eur_per_kwh', type: 'number', element: importConstantInput },
        { key: 'spot_offset_eur_per_kwh', type: 'number', element: importSpotOffsetInput },
        { key: 'dual_peak_eur_per_kwh', type: 'number', element: importPeakRateInput },
        { key: 'dual_offpeak_eur_per_kwh', type: 'number', element: importOffpeakRateInput },
        { key: 'dual_peak_start_local', type: 'time', element: importPeakStartInput },
        { key: 'dual_peak_end_local', type: 'time', element: importPeakEndInput },
    ],
    export: [
        { key: 'constant_eur_per_kwh', type: 'number', element: exportConstantInput },
        { key: 'spot_offset_eur_per_kwh', type: 'number', element: exportSpotOffsetInput },
        { key: 'dual_peak_eur_per_kwh', type: 'number', element: exportPeakRateInput },
        { key: 'dual_offpeak_eur_per_kwh', type: 'number', element: exportOffpeakRateInput },
        { key: 'dual_peak_start_local', type: 'time', element: exportPeakStartInput },
        { key: 'dual_peak_end_local', type: 'time', element: exportPeakEndInput },
    ],
};
const entityModal = document.getElementById('entityModal');
const entityModalBackdrop = document.getElementById('entityModalBackdrop');
const entityModalClose = document.getElementById('entityModalClose');
const entityModalCancel = document.getElementById('entityModalCancel');
const entitySearchInput = document.getElementById('entitySearch');
const entityListContainer = document.getElementById('entityList');
const exportLimitToggle = document.getElementById('exportLimitToggle');
let batterySettings = null;
let batteryBusy = false;
const messageTimers = Object.create(null);
let pricingSettings = null;
let pricingSaveTimer = null;
let pricingSaving = false;
let pricingPendingResave = false;
const PRICING_SAVE_DEBOUNCE_MS = 400;

function computeHourCycle(locale) {
    if (!locale) return undefined;
    return locale.toLowerCase().startsWith('cs') ? 'h23' : undefined;
}

function updateFormattingContext(locale, timeZone) {
    if (locale) {
        activeLocale = locale;
    } else if (!activeLocale) {
        activeLocale = navigator.language || undefined;
    }
    if (timeZone) {
        haTimeZone = timeZone;
    }
    const resolvedLocale = activeLocale;
    const hourCycle = computeHourCycle(resolvedLocale);
    const timeOptions = { hour: '2-digit', minute: '2-digit' };
    const dateTimeOptions = { weekday: 'short', day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit' };
    if (haTimeZone) {
        timeOptions.timeZone = haTimeZone;
        dateTimeOptions.timeZone = haTimeZone;
    }
    if (hourCycle) {
        timeOptions.hourCycle = hourCycle;
        dateTimeOptions.hourCycle = hourCycle;
    }
    timeFormatter = new Intl.DateTimeFormat(resolvedLocale, timeOptions);
    dateTimeFormatter = new Intl.DateTimeFormat(resolvedLocale, dateTimeOptions);
    numberFmt1 = new Intl.NumberFormat(resolvedLocale, { maximumFractionDigits: 1, minimumFractionDigits: 0 });
    numberFmt2 = new Intl.NumberFormat(resolvedLocale, { maximumFractionDigits: 2, minimumFractionDigits: 0 });
    currencyFmt = new Intl.NumberFormat(resolvedLocale, { style: 'currency', currency: 'EUR', maximumFractionDigits: 2 });
    const chartLocale = activeLocale || undefined;
    if (typeof Chart !== 'undefined' && Chart.defaults) {
        Chart.defaults.locale = chartLocale;
    }
    applyLocaleToCharts();
}

function ensureFormatters() {
    if (!timeFormatter || !dateTimeFormatter) {
        updateFormattingContext(activeLocale, haTimeZone);
    }
}

function applyLocaleToCharts() {
    const locale = activeLocale || undefined;
    if (forecastChart) {
        forecastChart.options.locale = locale;
    }
    if (planChart) {
        planChart.options.locale = locale;
    }
    if (gridChart) {
        gridChart.options.locale = locale;
    }
}

function formatDateTime(iso) {
    if (!iso) return '--';
    ensureFormatters();
    const dt = new Date(iso);
    if (Number.isNaN(dt.getTime())) return '--';
    return dateTimeFormatter.format(dt);
}

function formatTickLabel(iso) {
    if (!iso) return '';
    ensureFormatters();
    const dt = new Date(iso);
    if (Number.isNaN(dt.getTime())) return '';
    return timeFormatter.format(dt);
}

function sanitizeSeries(values, scale = 1) {
    return values.map((value) => {
        const num = Number(value);
        if (!Number.isFinite(num)) {
            return 0;
        }
        return num * scale;
    });
}

function updateViewMode() {
    const mainView = document.getElementById('mainView');
    const settingsView = document.getElementById('settingsView');
    const settingsBtn = document.getElementById('settingsBtn');
    if (!mainView || !settingsView || !settingsBtn) {
        return;
    }
    if (showingSettings) {
        mainView.classList.add('hidden');
        settingsView.classList.remove('hidden');
        settingsBtn.textContent = 'Back';
    } else {
        mainView.classList.remove('hidden');
        settingsView.classList.add('hidden');
        settingsBtn.textContent = 'Settings';
    }
}

function toggleSettings() {
    showingSettings = !showingSettings;
    if (showingSettings) {
        if (autoRefreshTimer) {
            clearTimeout(autoRefreshTimer);
            autoRefreshTimer = null;
        }
        if (pendingForecastRefresh) {
            clearTimeout(pendingForecastRefresh);
            pendingForecastRefresh = null;
        }
    }
    updateViewMode();
    if (showingSettings) {
        activateSettingsTab('control');
        loadSettingsData();
        loadControlSettings();
    } else {
        refreshAll();
    }
}

function activateSettingsTab(tab) {
    const buttons = document.querySelectorAll('.settings-nav button');
    const panels = document.querySelectorAll('.settings-panel');
    buttons.forEach((btn) => {
        if (btn.dataset.settingsTab === tab) {
            btn.classList.add('active');
        } else {
            btn.classList.remove('active');
        }
    });
    panels.forEach((panel) => {
        if (panel.dataset.settingsPanel === tab) {
            panel.classList.add('active');
        } else {
            panel.classList.remove('active');
        }
    });
}

function renderIntervention(intervention, fallbackText) {
    ensureFormatters();
    const el = document.getElementById('interventionBanner');
    if (!el) {
        return;
    }
    const classes = ['none', 'charge', 'discharge', 'limit'];
    classes.forEach((cls) => el.classList.remove(cls));

    if (!intervention || !intervention.type) {
        const text = fallbackText || 'Current intervention: Not available';
        el.classList.add('none');
        el.innerHTML = text;
        el.style.display = 'block';
        return;
    }

    const type = intervention.type;
    const description = intervention.description || '';

    let mainText = `Current intervention: ${type}`;
    let modeClass = 'none';

    // Map intervention types to CSS classes
    if (type === 'Charge from Grid') {
        modeClass = 'charge';
    } else if (type === 'Discharge to Grid') {
        modeClass = 'discharge';
    } else if (type === 'Cover Load from Battery') {
        modeClass = 'discharge'; // Use same color as discharge
    } else if (type === 'Disable Battery') {
        modeClass = 'none';
    }

    el.classList.add(modeClass);
    el.style.display = 'block';

    if (description) {
        el.innerHTML = `${mainText}<span class="detail">${description}</span>`;
    } else {
        el.innerHTML = mainText;
    }
}

async function fetchJson(url, options) {
    const response = await fetch(url, options);
    if (!response.ok) {
        const text = await response.text();
        try {
            const parsed = JSON.parse(text);
            const detail = parsed.detail || parsed.message;
            throw new Error(detail || `HTTP ${response.status}`);
        } catch (_) {
            throw new Error(text || `HTTP ${response.status}`);
        }
    }
    return response.json();
}

function statisticSupportsMean(item) {
    if (!item || typeof item !== 'object') return false;
    if (item.has_mean) return true;
    const supported = item.supported_statistics;
    if (Array.isArray(supported)) {
        return supported.some((entry) => String(entry).toLowerCase() === 'mean');
    }
    return false;
}

function statisticLabel(item) {
    if (!item) return '';
    const name = item.display_name || item.statistic_id || '';
    return item.unit ? `${name} (${item.unit})` : name;
}

function setMessage(element, text, kind = 'info', autoHideMs = 0, timerKey) {
    if (!element) return;
    if (timerKey && messageTimers[timerKey]) {
        clearTimeout(messageTimers[timerKey]);
        messageTimers[timerKey] = null;
    }
    if (!text) {
        element.style.display = 'none';
        element.classList.remove('error');
        element.textContent = '';
        return;
    }
    element.style.display = 'block';
    element.textContent = text;
    if (kind === 'error') {
        element.classList.add('error');
    } else {
        element.classList.remove('error');
    }
    if (timerKey && text && kind !== 'error' && autoHideMs > 0) {
        messageTimers[timerKey] = setTimeout(() => {
            messageTimers[timerKey] = null;
            setMessage(element, '');
        }, autoHideMs);
    }
}

function setHAEntitiesMessage(text, kind = 'info') {
    setMessage(haEntitiesMessage, text, kind);
}

function setBatteryMessage(text, kind = 'info', autoHideMs = 0) {
    setMessage(batteryMessage, text, kind, autoHideMs, 'battery');
}

function bindClick(element, handler) {
    if (!element || typeof handler !== 'function') return;
    element.addEventListener('click', handler);
}

function renderHAEntitiesHints() {
    if (!haEntitiesSettings) return;
    const catalogMap = Object.create(null);
    statisticsCatalog.forEach((item) => {
        if (item && item.statistic_id) {
            catalogMap[item.statistic_id] = item;
        }
    });
    const house = haEntitiesSettings.house_consumption || {};
    const pv = haEntitiesSettings.pv_power || {};
    const applyHint = (el, selection) => {
        if (!el || !selection) return;
        const meta = catalogMap[selection.entity_id];
        const status = selection.recorder_status || (meta ? (statisticSupportsMean(meta) ? 'ok' : 'no_mean') : 'missing');
        el.classList.remove('error');
        el.textContent = '';
        if (status === 'missing') {
            const warning = document.createElement('span');
            warning.className = 'error-text';
            warning.textContent = 'Recorder entry missing. Refresh statistics if this looks wrong.';
            el.appendChild(warning);
            el.classList.add('error');
        } else if (status === 'no_mean') {
            const note = document.createElement('span');
            note.className = 'warning-text';
            note.textContent = ' Recorder metadata does not list mean values. Forecasts will still attempt to use the statistic.';
            el.appendChild(note);
        }
    };
    applyHint(houseHint, house);
    applyHint(pvHint, pv);
}

function refreshEntityCatalogs(list) {
    entityCatalog = Array.isArray(list) ? list : [];
    entityLookup = Object.create(null);
    entityCatalog.forEach((item) => {
        if (item && item.entity_id) {
            entityLookup[item.entity_id] = item;
        }
    });
    powerEntityCatalog = filterPowerEntities(entityCatalog);
    batteryEntityCatalog = filterBatteryEntities(entityCatalog);
}

function sanitizeTimeValue(value, fallback) {
    if (typeof value !== 'string') {
        return fallback;
    }
    const trimmed = value.trim();
    if (!trimmed) {
        return fallback;
    }
    return /^([01]\d|2[0-3]):[0-5]\d$/.test(trimmed) ? trimmed : fallback;
}

function defaultTariffConfig(scope) {
    const isImport = scope === 'import';
    return {
        mode: 'spot_plus_constant',
        constant_eur_per_kwh: null,
        spot_offset_eur_per_kwh: isImport ? 0.14 : 0.0,
        dual_peak_eur_per_kwh: null,
        dual_offpeak_eur_per_kwh: null,
        dual_peak_start_local: '07:00',
        dual_peak_end_local: '21:00',
    };
}

function defaultPricingSettings() {
    return {
        import: defaultTariffConfig('import'),
        export: defaultTariffConfig('export'),
        export_enabled: true,
        export_limit_kw: null,
    };
}

function defaultBatterySettings() {
    return {
        soc_sensor: null,
        wear_cost_eur_per_kwh: 0.1,
        capacity_kwh: 10,
        power_limit_kw: 3,
        soc_min: 0.1,
        soc_max: 0.9,
    };
}

function normalizeBatterySettings(raw) {
    const defaults = defaultBatterySettings();
    if (!raw || typeof raw !== 'object') {
        return defaults;
    }
    const normalized = { ...defaults };
    if (raw.soc_sensor && typeof raw.soc_sensor === 'object' && raw.soc_sensor.entity_id) {
        normalized.soc_sensor = {
            entity_id: raw.soc_sensor.entity_id,
            unit: raw.soc_sensor.unit || null,
        };
    }
    const wear = Number(raw.wear_cost_eur_per_kwh);
    normalized.wear_cost_eur_per_kwh = Number.isFinite(wear) && wear >= 0 ? wear : defaults.wear_cost_eur_per_kwh;
    const capacity = Number(raw.capacity_kwh);
    normalized.capacity_kwh = Number.isFinite(capacity) && capacity > 0 ? capacity : defaults.capacity_kwh;
    const powerLimit = Number(raw.power_limit_kw);
    normalized.power_limit_kw = Number.isFinite(powerLimit) && powerLimit > 0 ? powerLimit : defaults.power_limit_kw;
    const socMin = Number(raw.soc_min);
    const socMax = Number(raw.soc_max);
    normalized.soc_min = Number.isFinite(socMin) && socMin >= 0 && socMin <= 0.98 ? socMin : defaults.soc_min;
    normalized.soc_max = Number.isFinite(socMax) && socMax > 0 && socMax <= 1 ? socMax : defaults.soc_max;
    normalized.soc_min = Math.min(Math.max(normalized.soc_min, 0), 0.98);
    normalized.soc_max = Math.min(Math.max(normalized.soc_max, 0.02), 1);
    if (normalized.soc_max <= normalized.soc_min + 0.005) {
        normalized.soc_max = Math.min(1, Math.max(normalized.soc_min + 0.05, defaults.soc_max));
    }
    return normalized;
}

function ensureBatterySettings() {
    if (!batterySettings || typeof batterySettings !== 'object') {
        batterySettings = defaultBatterySettings();
    }
    if (!batterySettings.soc_sensor || typeof batterySettings.soc_sensor !== 'object') {
        batterySettings.soc_sensor = null;
    }
    if (!Number.isFinite(Number(batterySettings.wear_cost_eur_per_kwh)) || Number(batterySettings.wear_cost_eur_per_kwh) < 0) {
        batterySettings.wear_cost_eur_per_kwh = 0.1;
    }
    if (!Number.isFinite(Number(batterySettings.capacity_kwh)) || Number(batterySettings.capacity_kwh) <= 0) {
        batterySettings.capacity_kwh = 10;
    }
    if (!Number.isFinite(Number(batterySettings.power_limit_kw)) || Number(batterySettings.power_limit_kw) <= 0) {
        batterySettings.power_limit_kw = 3;
    }
    const defaults = defaultBatterySettings();
    const socMin = Number(batterySettings.soc_min);
    batterySettings.soc_min = Number.isFinite(socMin) && socMin >= 0 ? Math.min(socMin, 0.98) : defaults.soc_min;
    const socMax = Number(batterySettings.soc_max);
    batterySettings.soc_max = Number.isFinite(socMax) && socMax > 0 ? Math.min(Math.max(socMax, batterySettings.soc_min + 0.01), 1) : defaults.soc_max;
    if (batterySettings.soc_max <= batterySettings.soc_min + 0.005) {
        batterySettings.soc_max = Math.min(1, batterySettings.soc_min + 0.05);
    }
    return batterySettings;
}

function normalizeTariffConfig(raw, scope) {
    const defaults = defaultTariffConfig(scope);
    const cfg = { ...defaults };
    if (!raw || typeof raw !== 'object') {
        return cfg;
    }
    if (typeof raw.mode === 'string') {
        const normalizedMode = raw.mode.trim();
        if (TARIFF_MODES.includes(normalizedMode)) {
            cfg.mode = normalizedMode;
        }
    }
    if (raw.constant_eur_per_kwh !== undefined) {
        const val = Number(raw.constant_eur_per_kwh);
        cfg.constant_eur_per_kwh = Number.isFinite(val) ? val : defaults.constant_eur_per_kwh;
    }
    if (raw.spot_offset_eur_per_kwh !== undefined) {
        const val = Number(raw.spot_offset_eur_per_kwh);
        cfg.spot_offset_eur_per_kwh = Number.isFinite(val) ? val : defaults.spot_offset_eur_per_kwh;
    }
    if (raw.dual_peak_eur_per_kwh !== undefined) {
        const val = Number(raw.dual_peak_eur_per_kwh);
        cfg.dual_peak_eur_per_kwh = Number.isFinite(val) ? val : defaults.dual_peak_eur_per_kwh;
    }
    if (raw.dual_offpeak_eur_per_kwh !== undefined) {
        const val = Number(raw.dual_offpeak_eur_per_kwh);
        cfg.dual_offpeak_eur_per_kwh = Number.isFinite(val) ? val : defaults.dual_offpeak_eur_per_kwh;
    }
    if (raw.dual_peak_start_local !== undefined) {
        cfg.dual_peak_start_local = sanitizeTimeValue(raw.dual_peak_start_local, defaults.dual_peak_start_local);
    }
    if (raw.dual_peak_end_local !== undefined) {
        cfg.dual_peak_end_local = sanitizeTimeValue(raw.dual_peak_end_local, defaults.dual_peak_end_local);
    }
    return cfg;
}

function normalizePricingSettings(raw) {
    const settings = defaultPricingSettings();
    if (!raw || typeof raw !== 'object') {
        return settings;
    }
    if (raw.import) {
        settings.import = normalizeTariffConfig(raw.import, 'import');
    }
    if (raw.export) {
        settings.export = normalizeTariffConfig(raw.export, 'export');
    }
    if (typeof raw.export_enabled === 'boolean') {
        settings.export_enabled = raw.export_enabled;
    }
    if (raw.export_limit_kw !== undefined && raw.export_limit_kw !== null) {
        const val = Number(raw.export_limit_kw);
        settings.export_limit_kw = Number.isFinite(val) && val >= 0 ? val : null;
    }
    return settings;
}

function resolveTariffConfig(scope) {
    ensurePricingSettings();
    const normalized = normalizeTariffConfig(pricingSettings[scope], scope);
    pricingSettings[scope] = normalized;
    return normalized;
}

function applySettingsPayload(payload) {
    if (Array.isArray(payload.statistics)) {
        statisticsCatalog = payload.statistics;
    }
    if (Array.isArray(payload.entities)) {
        refreshEntityCatalogs(payload.entities);
    }
    if (payload.settings) {
        if (payload.settings.ha_entities && !haEntitiesPendingResave) {
            haEntitiesSettings = payload.settings.ha_entities;
        }
        if (payload.settings.battery) {
            batterySettings = normalizeBatterySettings(payload.settings.battery);
        } else {
            batterySettings = normalizeBatterySettings(null);
        }
        if (payload.settings.pricing) {
            pricingSettings = normalizePricingSettings(payload.settings.pricing);
        }
    }
    renderHAEntitiesForm();
    renderBatteryForm();
    renderPricingForm();
    renderDeferrableLoads();
    updateChartVisibility();
}

function updateChartVisibility() {
    if (!planChart || !pricingSettings) return;
    const exportEnabled = pricingSettings.export_enabled !== false;
    const ds = planChart.data.datasets.find(d => d.id === 'exportPrice');
    if (ds) {
        ds.hidden = !exportEnabled;
        planChart.update();
    }
}

function filterPowerEntities(list) {
    const allowedUnits = new Set(['w', 'kw', 'mw', 'wh', 'kwh', 'mwh']);
    return list.filter((item) => {
        if (!item || !item.entity_id) return false;
        if (item.category && String(item.category).toLowerCase() === 'power') return true;
        if (item.category && String(item.category).toLowerCase() !== 'battery' && item.category !== undefined) return false;
        if (item.domain && item.domain !== 'sensor') return false;
        const deviceClass = (item.device_class || '').toLowerCase();
        const unit = (item.unit || '').toLowerCase();
        if (deviceClass === 'power' || deviceClass === 'energy') {
            return true;
        }
        return allowedUnits.has(unit);
    });
}

function filterBatteryEntities(list) {
    const allowedUnits = new Set(['%', 'percent', 'percentage']);
    return list.filter((item) => {
        if (!item || !item.entity_id) return false;
        const category = (item.category || '').toLowerCase();
        if (category === 'battery') return true;
        if (item.domain && !['sensor', 'number', 'input_number'].includes(item.domain)) return false;
        const deviceClass = (item.device_class || '').toLowerCase();
        const unit = (item.unit || '').toLowerCase();
        if (deviceClass === 'battery') {
            return true;
        }
        return allowedUnits.has(unit);
    });
}

function getEntityMeta(entityId) {
    return (entityId && entityLookup[entityId]) || null;
}

function entityButtonLabel(selection, fallback) {
    if (!selection) return fallback;
    const meta = getEntityMeta(selection.entity_id);
    if (meta) {
        const unit = meta.unit ? ` (${meta.unit})` : '';
        return `${meta.name}${unit}`;
    }
    if (selection.entity_id) {
        return selection.entity_id;
    }
    return fallback;
}

function updateEntityButton(button, labelEl, selection, fallback, busy, hasChoices, allowEmpty = false) {
    if (!button || !labelEl) return;
    labelEl.textContent = entityButtonLabel(selection, fallback);
    const shouldDisable = busy || (!hasChoices && !allowEmpty);
    if (shouldDisable) {
        button.setAttribute('disabled', 'disabled');
    } else {
        button.removeAttribute('disabled');
    }
    button.dataset.entityId = selection && selection.entity_id ? selection.entity_id : '';
}

function renderHAEntitiesForm() {
    if (!haEntitiesSettings) return;

    renderHAEntitiesTable();

    if (exportLimitToggle) {
        setSafeCheck(exportLimitToggle, haEntitiesSettings.export_power_limited);
        if (haEntitiesBusy) {
            exportLimitToggle.setAttribute('disabled', 'disabled');
        } else {
            exportLimitToggle.removeAttribute('disabled');
        }
    }

    const recorderStatuses = [
        haEntitiesSettings.house_consumption && haEntitiesSettings.house_consumption.recorder_status,
        haEntitiesSettings.pv_power && haEntitiesSettings.pv_power.recorder_status,
    ];
    if (recorderStatuses.every((value) => value === 'missing')) {
        setHAEntitiesMessage('No Home Assistant statistics were returned for the selected sensors. Verify long-term statistics are enabled.', 'error');
    } else if (!powerEntityCatalog.length) {
        setHAEntitiesMessage('No power sensors detected. Expose Home Assistant sensors with power readings to proceed.', 'error');
    } else if (!batteryEntityCatalog.length) {
        setHAEntitiesMessage('No battery sensors detected. Expose a sensor with % unit to proceed.', 'error');
    }
}

function renderBatteryForm() {
    ensureBatterySettings();
    Object.keys(batteryFieldConfigs).forEach((fieldKey) => {
        renderBatteryFieldInput(fieldKey);
    });
}

function getBatteryFieldConfig(fieldKey) {
    return batteryFieldConfigs[fieldKey] || null;
}

function formatBatteryFieldDisplayValue(config, rawValue) {
    if (!config) return '';
    const numeric = Number(rawValue);
    if (!Number.isFinite(numeric)) {
        return '';
    }
    const scaled = config.formatPercent ? numeric * 100 : numeric;
    const rounded = Math.round((scaled + Number.EPSILON) * 1000) / 1000;
    if (!Number.isFinite(rounded)) {
        return '';
    }
    return rounded.toString();
}

function renderBatteryFieldInput(fieldKey, options = {}) {
    const config = getBatteryFieldConfig(fieldKey);
    if (!config || !config.input) {
        return;
    }
    const value = batterySettings ? batterySettings[fieldKey] : null;
    setSafeValue(config.input, formatBatteryFieldDisplayValue(config, value));
    if (batteryBusy) {
        config.input.setAttribute('disabled', 'disabled');
    } else {
        config.input.removeAttribute('disabled');
    }
    if (config.hintEl) {
        if (options.resetHint) {
            config.hintEl.classList.remove('error');
            config.hintEl.textContent = config.hintText;
        } else if (!config.hintEl.classList.contains('error')) {
            config.hintEl.textContent = config.hintText;
        }
    }
}

function showBatteryFieldError(config, message, hintOverride) {
    const fallback = message || 'Enter a valid number.';
    if (config && config.hintEl) {
        config.hintEl.classList.add('error');
        config.hintEl.textContent = hintOverride || config.invalidHint || fallback;
    }
    setBatteryMessage(message || config.invalidMessage || 'Enter a valid number.', 'error');
}

function ensurePricingSettings() {
    if (!pricingSettings || typeof pricingSettings !== 'object') {
        pricingSettings = defaultPricingSettings();
    }
    if (!pricingSettings.import) {
        pricingSettings.import = defaultTariffConfig('import');
    }
    if (!pricingSettings.export) {
        pricingSettings.export = defaultTariffConfig('export');
    }
    return pricingSettings;
}

function setPricingMessage(text, kind = 'info', autoHideMs = 0) {
    setMessage(pricingMessage, text, kind, autoHideMs, 'pricing');
}

function setSafeValue(input, value) {
    if (!input) return;
    if (document.activeElement === input) return;
    const strVal = (value === null || value === undefined) ? '' : String(value);
    if (input.value !== strVal) {
        input.value = strVal;
    }
}

function setSafeCheck(input, value) {
    if (!input) return;
    if (document.activeElement === input) return;
    const boolVal = Boolean(value);
    if (input.checked !== boolVal) input.checked = boolVal;
}

function setNumericInputValue(input, value) {
    if (!input) return;
    let valStr = '';
    if (Number.isFinite(value)) {
        valStr = value.toString();
    } else if (value !== null && value !== undefined) {
        const parsed = Number(value);
        if (Number.isFinite(parsed)) valStr = parsed.toString();
    }
    setSafeValue(input, valStr);
}

function setTimeInputValue(input, value, fallback) {
    if (!input) return;
    const val = sanitizeTimeValue(value, fallback);
    setSafeValue(input, val);
}

function updateTariffVisibility(scope) {
    const cfg = resolveTariffConfig(scope);
    const mode = cfg.mode && TARIFF_MODES.includes(cfg.mode) ? cfg.mode : defaultTariffConfig(scope).mode;
    const nodes = document.querySelectorAll(`[data-tariff-scope="${scope}"]`);
    nodes.forEach((node) => {
        const modesRaw = node.dataset.tariffMode || '';
        const modeList = modesRaw.split(',').map((entry) => entry.trim()).filter(Boolean);
        if (!modeList.length || modeList.includes(mode)) {
            node.style.display = '';
        } else {
            node.style.display = 'none';
        }
    });
}

function updateTariffHint(scope) {
    const hintEl = scope === 'import' ? importTariffHint : exportTariffHint;
    if (!hintEl) return;
    const cfg = resolveTariffConfig(scope);
    const mode = cfg.mode && TARIFF_MODES.includes(cfg.mode) ? cfg.mode : defaultTariffConfig(scope).mode;
    const label = scope === 'import' ? 'Import' : 'Export';
    let text = `${label} price configuration pending.`;
    if (mode === 'constant') {
        text = `${label} price stays fixed for every interval.`;
    } else if (mode === 'spot_plus_constant') {
        text = `${label} price follows the day-ahead signal adjusted by the offset below.`;
    } else if (mode === 'dual_rate') {
        text = `${label} price switches between peak and off-peak windows in local time.`;
    }
    hintEl.textContent = text;
}

function renderPricingForm() {
    if (!pricingForm) return;
    ensurePricingSettings();

    const exportEnabledToggle = document.getElementById('exportEnabledToggle');
    const exportSettingsContainer = document.getElementById('exportSettingsContainer');
    if (exportEnabledToggle) {
        setSafeCheck(exportEnabledToggle, pricingSettings.export_enabled);
    }
    if (exportSettingsContainer) {
        exportSettingsContainer.style.display = pricingSettings.export_enabled ? 'block' : 'none';
    }

    const exportPowerLimitInput = document.getElementById('exportPowerLimit');
    if (exportPowerLimitInput) {
        setNumericInputValue(exportPowerLimitInput, pricingSettings.export_limit_kw);
    }

    ['import', 'export'].forEach((scope) => {
        const config = resolveTariffConfig(scope);
        const defaults = defaultTariffConfig(scope);
        const select = scope === 'import' ? importTariffModeSelect : exportTariffModeSelect;
        if (select) {
            select.value = config.mode;
        }
        const bindings = tariffFieldBindings[scope] || [];
        bindings.forEach(({ key, type, element }) => {
            if (!element) return;
            if (type === 'time') {
                setTimeInputValue(element, config[key], defaults[key]);
            } else {
                setNumericInputValue(element, config[key]);
            }
        });
        updateTariffVisibility(scope);
        updateTariffHint(scope);
    });
}

function handleTariffModeChange(scope, mode) {
    const config = resolveTariffConfig(scope);
    const normalized = TARIFF_MODES.includes(mode) ? mode : defaultTariffConfig(scope).mode;
    config.mode = normalized;
    renderPricingForm();
    schedulePricingSave();
}





function schedulePricingSave() {
    if (!pricingForm) return;
    if (pricingSaving) {
        pricingPendingResave = true;
        return;
    }
    if (pricingSaveTimer) {
        clearTimeout(pricingSaveTimer);
    }
    pricingSaveTimer = setTimeout(() => {
        pricingSaveTimer = null;
        void savePricingSettings(false);
    }, PRICING_SAVE_DEBOUNCE_MS);
}

async function savePricingSettings(manual = false) {
    ensurePricingSettings();
    if (pricingSaving) {
        pricingPendingResave = true;
        return;
    }
    if (pricingSaveTimer) {
        clearTimeout(pricingSaveTimer);
        pricingSaveTimer = null;
    }
    pricingSaving = true;
    if (manual) {
        setPricingMessage('Saving pricing…');
    } else {
        setPricingMessage('');
    }
    try {
        const payload = normalizePricingSettings(pricingSettings);
        pricingSettings = payload;
        const response = await fetchJson('api/settings', {
            method: 'PATCH',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ pricing: payload }),
        });
        applySettingsPayload(response);
        if (manual) {
            setPricingMessage('Pricing saved.', 'info', 2500);
        } else {
            setPricingMessage('');
        }
    } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        setPricingMessage(message, 'error');
    } finally {
        pricingSaving = false;
        if (pricingPendingResave) {
            pricingPendingResave = false;
            schedulePricingSave();
        }
    }
}

function bindTariffInput(input, scope, key, type) {
    if (!input) return;
    input.addEventListener('change', (event) => {
        ensurePricingSettings();
        if (type === 'time') {
            const fallback = defaultTariffConfig(scope)[key];
            const sanitized = sanitizeTimeValue(event.target.value, fallback);
            pricingSettings[scope][key] = sanitized;
            event.target.value = sanitized;
        } else {
            const raw = typeof event.target.value === 'string' ? event.target.value.trim() : '';
            if (!raw) {
                pricingSettings[scope][key] = null;
                schedulePricingSave();
                return;
            }
            const parsed = Number.parseFloat(raw);
            if (Number.isFinite(parsed)) {
                pricingSettings[scope][key] = parsed;
            } else {
                pricingSettings[scope][key] = null;
                event.target.value = '';
            }
        }
        schedulePricingSave();
    });
}

function bindTariffFields() {
    Object.entries(tariffFieldBindings).forEach(([scope, bindings]) => {
        bindings.forEach(({ element, key, type }) => {
            bindTariffInput(element, scope, key, type);
        });
    });
}

async function loadSettingsData() {
    try {
        const payload = await fetchJson('api/settings');
        applySettingsPayload(payload);
        const recorderStatuses = [
            haEntitiesSettings && haEntitiesSettings.house_consumption && haEntitiesSettings.house_consumption.recorder_status,
            haEntitiesSettings && haEntitiesSettings.pv_power && haEntitiesSettings.pv_power.recorder_status,
        ];
        if (recorderStatuses.some((value) => value !== 'missing') && powerEntityCatalog.length) {
            setHAEntitiesMessage('');
        }
        if (batteryEntityCatalog.length) {
            setBatteryMessage('');
        }

        const autoTrainingToggle = document.getElementById('autoTrainingToggle');
        if (autoTrainingToggle) {
            const enabled = payload.settings ? payload.settings.auto_training_enabled : false;
            setSafeCheck(autoTrainingToggle, enabled);
        }
    } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        setHAEntitiesMessage(message, 'error');
        setBatteryMessage(message, 'error');
    }
}

async function updateAutoTrainingSetting(enabled) {
    const toggle = document.getElementById('autoTrainingToggle');
    if (toggle) toggle.disabled = true;

    // Use controlMessage (haEntitiesMessage) for feedback
    setHAEntitiesMessage('Saving auto-training preference…');

    try {
        const payload = await fetchJson('api/settings', {
            method: 'PATCH',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ auto_training_enabled: Boolean(enabled) }),
        });
        applySettingsPayload(payload);
        setHAEntitiesMessage('Saved auto-training preference.', 'success');
    } catch (err) {
        setHAEntitiesMessage(err instanceof Error ? err.message : String(err), 'error');
        if (toggle) {
            toggle.checked = !enabled; // Revert on error
        }
    } finally {
        if (toggle) toggle.disabled = false;
    }
}

function setEntityTriggersDisabled(disabled) {
    [houseTrigger, pvTrigger, batteryTrigger].forEach((btn) => {
        if (!btn) return;
        if (disabled) {
            btn.setAttribute('disabled', 'disabled');
        } else {
            btn.removeAttribute('disabled');
        }
    });

    if (batteryWearInput) {
        if (disabled || batteryBusy) {
            batteryWearInput.setAttribute('disabled', 'disabled');
        } else {
            batteryWearInput.removeAttribute('disabled');
        }
    }
    if (exportLimitToggle) {
        if (disabled) {
            exportLimitToggle.setAttribute('disabled', 'disabled');
        } else {
            exportLimitToggle.removeAttribute('disabled');
        }
    }
    renderBatteryForm();
    renderHAEntitiesForm();
}

async function updateHAEntitySelection(key, entityId) {
    if (!haEntitiesSettings || haEntitiesBusy) return;
    const current = haEntitiesSettings[key] ? haEntitiesSettings[key].entity_id : null;
    if (entityId === current) return;
    haEntitiesBusy = true;
    setEntityTriggersDisabled(true);
    setHAEntitiesMessage('Saving selection…');
    try {
        const payload = await fetchJson('api/settings', {
            method: 'PATCH',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ ha_entities: { [key]: { entity_id: entityId } } }),
        });
        applySettingsPayload(payload);
        setHAEntitiesMessage('Saved. Re-run training to refresh the models with the new inputs.', 'success');
    } catch (err) {
        setHAEntitiesMessage(err instanceof Error ? err.message : String(err), 'error');
    } finally {
        haEntitiesBusy = false;
        setEntityTriggersDisabled(false);
        if (haEntitiesSettings) {
            renderHAEntitiesForm();
        }
    }
}

async function updateExportLimitSetting(enabled) {
    if (!haEntitiesSettings || haEntitiesBusy) return;
    haEntitiesBusy = true;
    setEntityTriggersDisabled(true);
    setHAEntitiesMessage('Saving selection…');
    try {
        const payload = await fetchJson('api/settings', {
            method: 'PATCH',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ ha_entities: { export_power_limited: Boolean(enabled) } }),
        });
        applySettingsPayload(payload);
        setHAEntitiesMessage('Saved export limit preference. Re-run training to incorporate the change.', 'success');
    } catch (err) {
        setHAEntitiesMessage(err instanceof Error ? err.message : String(err), 'error');
        if (exportLimitToggle) {
            exportLimitToggle.checked = Boolean(haEntitiesSettings.export_power_limited);
        }
    } finally {
        haEntitiesBusy = false;
        setEntityTriggersDisabled(false);
        if (haEntitiesSettings) {
            renderHAEntitiesForm();
        }
    }
}

async function updateBatterySelection(entityId) {
    return updateHAEntitySelection('soc_sensor', entityId);
}

async function submitBatteryConfig(partial, options = {}) {
    ensureBatterySettings();
    if (batteryBusy) return false;
    batteryBusy = true;
    renderBatteryForm();
    if (options.pendingMessage !== false) {
        const pendingText = typeof options.pendingMessage === 'string' ? options.pendingMessage : 'Saving battery settings…';
        setBatteryMessage(pendingText);
    }
    try {
        const payload = await fetchJson('api/settings', {
            method: 'PATCH',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ battery: partial }),
        });
        applySettingsPayload(payload);
        if (typeof options.onSuccess === 'function') {
            options.onSuccess();
        }
        if (options.successMessage) {
            setBatteryMessage(options.successMessage, 'success', options.successAutoHideMs || 2500);
        } else if (options.pendingMessage !== false) {
            setBatteryMessage('');
        }
        return true;
    } catch (err) {
        setBatteryMessage(err instanceof Error ? err.message : String(err), 'error');
        return false;
    } finally {
        batteryBusy = false;
        renderBatteryForm();
    }
}

function handleBatteryNumericInput(fieldKey) {
    ensureBatterySettings();
    const config = getBatteryFieldConfig(fieldKey);
    if (!config || !config.input) {
        return;
    }
    if (batteryBusy) {
        return;
    }
    if (config.hintEl && config.hintText && !config.hintEl.classList.contains('error')) {
        config.hintEl.textContent = config.hintText;
    }
    const raw = typeof config.input.value === 'string' ? config.input.value.trim() : '';
    if (!raw) {
        renderBatteryFieldInput(fieldKey);
        return;
    }
    const parsed = Number.parseFloat(raw);
    if (!Number.isFinite(parsed)) {
        showBatteryFieldError(config, config.invalidMessage, config.invalidHint);
        renderBatteryFieldInput(fieldKey);
        return;
    }
    let normalized = parsed;
    if (config.formatPercent) {
        normalized = parsed / 100;
    }
    if (!Number.isFinite(normalized)) {
        showBatteryFieldError(config, config.invalidMessage, config.invalidHint);
        renderBatteryFieldInput(fieldKey);
        return;
    }
    const min = typeof config.min === 'number' ? config.min : -Infinity;
    const max = typeof config.max === 'number' ? config.max : Infinity;
    const minInclusive = config.minInclusive !== false;
    const maxInclusive = config.maxInclusive !== false;
    const belowMin = minInclusive ? normalized < min : normalized <= min;
    const aboveMax = maxInclusive ? normalized > max : normalized >= max;
    if (belowMin || aboveMax) {
        showBatteryFieldError(config, config.invalidMessage, config.invalidHint);
        renderBatteryFieldInput(fieldKey);
        return;
    }
    if (typeof config.extraValidate === 'function') {
        const extraError = config.extraValidate(normalized);
        if (extraError) {
            showBatteryFieldError(config, extraError, extraError);
            renderBatteryFieldInput(fieldKey);
            return;
        }
    }
    const current = Number(batterySettings[fieldKey]);
    const epsilon = typeof config.epsilon === 'number' ? config.epsilon : 1e-6;
    if (Number.isFinite(current) && Math.abs(current - normalized) <= epsilon) {
        renderBatteryFieldInput(fieldKey);
        return;
    }
    submitBatteryConfig(
        { [fieldKey]: normalized },
        {
            pendingMessage: config.pendingMessage,
            successMessage: config.successMessage,
            onSuccess: () => {
                renderBatteryFieldInput(fieldKey, { resetHint: true });
            },
        },
    );
}

function openEntityModal(target) {
    if (!entityModal) return;
    entityModalTarget = target;
    if (target === 'soc_sensor') {
        if (!batteryEntityCatalog.length) {
            setBatteryMessage('No battery sensors detected. Expose a sensor with % unit to proceed.', 'error');
        } else {
            setBatteryMessage('');
        }
        lastEntityTrigger = batteryTrigger;
    } else {
        if (!powerEntityCatalog.length) {
            setHAEntitiesMessage('No power sensors detected. Expose Home Assistant sensors with power readings to proceed.', 'error');
            return;
        }
        lastEntityTrigger = target === 'pv_power' ? pvTrigger : houseTrigger;
    }
    const modalTitle = document.getElementById('entityModalTitle');
    if (modalTitle) {
        if (target === 'pv_power' || target === 'GENERIC_POWER') {
            modalTitle.textContent = 'Select Power Sensor';
        } else if (target === 'soc_sensor') {
            modalTitle.textContent = 'Select Battery SoC Sensor';
        } else if (target === 'GENERIC_SWITCH') {
            modalTitle.textContent = 'Select Control Switch';
        } else {
            modalTitle.textContent = 'Select Sensor';
        }
    }
    if (entitySearchInput) {
        entitySearchInput.value = '';
    }
    renderEntityList('');
    entityModal.classList.add('open');
    document.body.classList.add('modal-open');
    if (entitySearchInput) {
        setTimeout(() => entitySearchInput.focus(), 0);
    }
}

function closeEntityModal() {
    if (!entityModal) return;
    entityModal.classList.remove('open');
    document.body.classList.remove('modal-open');
    const trigger = lastEntityTrigger;
    entityModalTarget = null;
    lastEntityTrigger = null;
    if (trigger) {
        trigger.focus();
    }
}

function matchesEntity(item, term) {
    if (!term) return true;
    const haystack = `${item.name || ''} ${item.entity_id}`.toLowerCase();
    return haystack.includes(term);
}

function renderEntityList(searchTerm) {
    if (!entityListContainer) return;
    const term = (searchTerm || '').trim().toLowerCase();
    let catalog = [];
    let filterByDomain = null;

    if (entityModalTarget === 'soc_sensor') {
        catalog = batteryEntityCatalog;
    } else if (entityModalTarget === 'DRIVER_CONFIG') {
        catalog = entityCatalog;
        filterByDomain = activeDriverEntityDomain;
    } else if (entityModalTarget === 'GENERIC_SWITCH') {
        // Filter catalog for switch-like entities
        catalog = entityCatalog.filter(e =>
            e.entity_id.startsWith('switch.') ||
            e.entity_id.startsWith('light.') ||
            e.entity_id.startsWith('input_boolean.')
        );
    } else if (entityModalTarget === 'GENERIC_POWER') {
        catalog = powerEntityCatalog;
    } else {
        catalog = powerEntityCatalog;
    }

    let items = catalog.filter((item) => matchesEntity(item, term));
    if (filterByDomain) {
        items = items.filter((item) => item.entity_id.startsWith(filterByDomain + '.'));
    }

    entityListContainer.textContent = '';
    if (!items.length) {
        const empty = document.createElement('div');
        empty.className = 'entity-empty';
        if (term) {
            empty.textContent = `No matching ${filterByDomain || 'entities'} found.`;
        } else if (entityModalTarget === 'soc_sensor') {
            empty.textContent = 'No battery sensors available.';
        } else if (filterByDomain) {
            empty.textContent = `No ${filterByDomain} entities available.`;
        } else {
            empty.textContent = 'No power sensors available.';
        }
        entityListContainer.appendChild(empty);
        return;
    }
    let current = null;

    const noneOption = document.createElement('button');
    noneOption.type = 'button';
    noneOption.className = 'entity-option';
    noneOption.setAttribute('role', 'option');

    if (entityModalTarget === 'DRIVER_CONFIG' && activeDriverEntityDefault) {
        noneOption.innerHTML = `
            <span class="entity-name">Use Default</span>
            <span class="entity-meta">${activeDriverEntityDefault}</span>
        `;
    } else {
        noneOption.innerHTML = `
            <span class="entity-name">None</span>
            <span class="entity-meta">Clear selection</span>
        `;
    }

    noneOption.addEventListener('click', () => {
        const targetKey = entityModalTarget;
        closeEntityModal();
        if (targetKey === 'DRIVER_CONFIG' || targetKey === 'GENERIC_POWER' || targetKey === 'GENERIC_SWITCH') {
            if (activeDriverEntityCallback) activeDriverEntityCallback(null);
        } else if (targetKey) {
            updateHAEntitySelection(targetKey, null);
        }
    });
    entityListContainer.appendChild(noneOption);

    if (entityModalTarget && haEntitiesSettings && haEntitiesSettings[entityModalTarget]) {
        current = haEntitiesSettings[entityModalTarget].entity_id;
    }
    items.forEach((item) => {
        const option = document.createElement('button');
        option.type = 'button';
        option.className = 'entity-option';
        option.setAttribute('role', 'option');
        option.dataset.entityId = item.entity_id;
        option.innerHTML = `
            <span class="entity-name">${item.name}</span>
            <span class="entity-meta">${item.entity_id}${item.unit ? ` • ${item.unit}` : ''}</span>
        `;
        if (item.entity_id === current) {
            option.classList.add('selected');
        }
        option.addEventListener('click', () => {
            const targetKey = entityModalTarget;
            closeEntityModal();
            if (targetKey === 'DRIVER_CONFIG' || targetKey === 'GENERIC_POWER' || targetKey === 'GENERIC_SWITCH') {
                if (activeDriverEntityCallback) activeDriverEntityCallback(item.entity_id);
            } else if (targetKey) {
                updateHAEntitySelection(targetKey, item.entity_id);
            }
        });
        entityListContainer.appendChild(option);
    });
}

function updateSummary(summary) {
    const mapping = [
        { key: 'import_kwh', id: 'summary-import', format: (v) => numberFmt1.format(v) },
        { key: 'export_kwh', id: 'summary-export', format: (v) => numberFmt1.format(v) },
        { key: 'net_cost_eur', id: 'summary-net', format: (v) => currencyFmt.format(v) },
        { key: 'pv_energy_kwh', id: 'summary-pv-energy', format: (v) => numberFmt1.format(v) },
        { key: 'consumption_kwh', id: 'summary-load', format: (v) => numberFmt1.format(v) },
    ];
    if (!summary) {
        mapping.forEach((item) => {
            const el = document.getElementById(item.id);
            el.textContent = '--';
        });
        return;
    }
    mapping.forEach((item) => {
        const el = document.getElementById(item.id);
        const value = Number(summary[item.key]);
        el.textContent = Number.isFinite(value) ? item.format(value) : '--';
    });
}

function applyStatus(payload) {
    const errorEl = document.getElementById('error');
    const trainingStateEl = document.getElementById('trainingState');
    const trainingErrorEl = document.getElementById('trainingError');
    const trainBtn = document.getElementById('trainBtn');

    // Fallback if payload missing (e.g. partial update)
    const haError = payload.home_assistant_error || null;
    const cycleBtn = document.getElementById('cycleBtn');
    const cycleMessageEl = document.getElementById('cycleMessage');
    const summaryWindowEl = document.getElementById('summaryWindow');

    const controlSwitch = document.getElementById('controlSwitch');

    // Check if driver is configured
    // payload.driver_configured might be undefined in older backends, assume true if so to avoid breaking validation
    const hasDriver = payload.driver_configured !== false;

    if (controlSwitch) {
        controlSwitch.setAttribute('aria-checked', String(Boolean(payload.control_active)));

        if (haError) {
            controlSwitch.setAttribute('disabled', 'disabled');
            controlSwitch.setAttribute('title', 'Cannot enable: Home Assistant not connected');
        } else if (!hasDriver) {
            controlSwitch.setAttribute('disabled', 'disabled');
            controlSwitch.setAttribute('title', 'Cannot enable: No inverter driver configured. Go to Settings > Inverter Control to configure.');
        } else {
            controlSwitch.removeAttribute('disabled');
            controlSwitch.setAttribute('title', 'Toggle automatic inverter control');
        }
    }

    if (!haError && payload.snapshot_available) {
        // Status OK
    } else if (!haError) {
        if (summaryWindowEl) summaryWindowEl.textContent = '';
        if (payload.last_cycle_error) {
            renderIntervention(null, `Forecast error: ${payload.last_cycle_error}`);
        } else {
            renderIntervention(null, 'Current intervention: Not available');
        }
    } else {
        if (summaryWindowEl) summaryWindowEl.textContent = '';
        renderIntervention(null, 'Current intervention unavailable until Home Assistant connects');
    }

    // --- Error Reporting ---
    const errorMessages = [];
    if (haError) errorMessages.push(`Home Assistant error: ${haError}`);
    if (!hasDriver) errorMessages.push("Inverter not configured. Please go to Settings > Inverter Control.");

    // Training status
    const training = payload.training || {};
    const trainingRunning = Boolean(training.running);
    const cycleRunning = Boolean(payload.cycle_running);

    if (training.error) errorMessages.push(`Training failed: ${training.error}`);
    if (payload.last_cycle_error) errorMessages.push(`Optimization cycle failed: ${payload.last_cycle_error}`);

    // Update main error element
    if (errorMessages.length > 0) {
        // Join with <br> for readability
        errorEl.innerHTML = errorMessages.join('<br>');
        errorEl.style.display = 'block';
    } else {
        errorEl.textContent = '';
        errorEl.style.display = 'none';
    }

    // --- Button States & Labels ---

    // Default labels
    const trainLabel = 'Train predictors';
    const cycleLabel = 'Recompute Plan';

    // Verify buttons exist
    if (trainBtn && cycleBtn) {
        // 1. Critical blocks
        if (haError || !hasDriver) {
            trainBtn.disabled = true;
            cycleBtn.disabled = true;
            trainBtn.textContent = trainLabel;
            cycleBtn.textContent = cycleLabel;

            trainingStateEl.textContent = haError ? 'System unavailable' : 'Inverter not configured';
            trainingStateEl.classList.remove('running');
        } else {
            // Clear status text if no critical error (User requested "Display only if critical")
            trainingStateEl.textContent = '';
            trainingStateEl.classList.remove('running');

            // 2. Running states handling (Mutual Exclusion)
            if (trainingRunning) {
                // Training is active
                trainBtn.disabled = true;
                trainBtn.textContent = 'Training…';

                cycleBtn.disabled = true; // Gray out other button
                cycleBtn.textContent = cycleLabel;
            } else if (cycleRunning) {
                // Cycle is active
                cycleBtn.disabled = true;
                cycleBtn.textContent = 'Recomputing…';

                trainBtn.disabled = true; // Gray out other button
                trainBtn.textContent = trainLabel;
            } else {
                // Idle
                trainBtn.disabled = false;
                trainBtn.textContent = trainLabel;

                cycleBtn.disabled = false;
                cycleBtn.textContent = cycleLabel;
            }
        }
    }

    // Hide legacy error boxes if we are using the main one
    if (trainingErrorEl) trainingErrorEl.style.display = 'none';

    // Remove "Optimization cycle running..." message (User requested removal)
    if (cycleMessageEl) {
        cycleMessageEl.style.display = 'none';
        delete cycleMessageEl.dataset.auto;
    }

    updateSummary(payload.summary);

    // Redundant refresh logic removed (global 1s interval handles real-time updates)
}

function ensureForecastChart() {
    if (forecastChart) return;
    const ctx = document.getElementById('forecastChart');
    forecastChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                {
                    label: 'PV forecast (kW)',
                    data: [],
                    borderColor: '#22c55e',
                    backgroundColor: 'rgba(34, 197, 94, 0.12)',
                    fill: true,
                    tension: 0.25,
                    pointRadius: 0,
                    pointHitRadius: 6,
                },
                {
                    label: 'Load forecast (kW)',
                    data: [],
                    borderColor: '#ef4444',
                    backgroundColor: 'rgba(239, 68, 68, 0.08)',
                    fill: false,
                    tension: 0.25,
                    pointRadius: 0,
                    pointHitRadius: 6,
                },
            ],
        },
        options: {
            locale: activeLocale || undefined,
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: 'index', intersect: false },
            scales: {
                x: {
                    ticks: { maxRotation: 0, autoSkip: true },
                },
                y: {
                    title: { display: true, text: 'Power (kW)' },
                    beginAtZero: true,
                },
            },
            plugins: {
                legend: { display: true },
            },
        },
    });
}

function ensurePlanChart() {
    if (planChart) return;
    const ctx = document.getElementById('planChart');
    planChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                {
                    id: 'exportPrice',
                    type: 'line',
                    label: 'Export price (€/kWh)',
                    data: [],
                    borderColor: '#f97316',
                    tension: 0.18,
                    fill: false,
                    yAxisID: 'price',
                    pointRadius: 0,
                    pointHitRadius: 8,
                    spanGaps: true,
                    borderWidth: 2,
                },
                {
                    id: 'importPrice',
                    type: 'line',
                    label: 'Import price (€/kWh)',
                    data: [],
                    borderColor: '#2563eb',
                    tension: 0.18,
                    fill: {
                        target: '-1',
                        above: 'rgba(37, 99, 235, 0.12)',
                        below: 'rgba(249, 115, 22, 0.12)',
                    },
                    yAxisID: 'price',
                    pointRadius: 0,
                    pointHitRadius: 8,
                    spanGaps: true,
                    borderWidth: 2,
                },
                {
                    type: 'line',
                    label: 'Battery SoC (%)',
                    data: [],
                    borderColor: '#a855f7',
                    backgroundColor: 'rgba(168, 85, 247, 0.08)',
                    tension: 0.2,
                    fill: false,
                    yAxisID: 'soc',
                    pointRadius: 0,
                    pointHitRadius: 6,
                    spanGaps: true,
                },
            ],
        },
        options: {
            locale: activeLocale || undefined,
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: { ticks: { maxRotation: 0, autoSkip: true } },
                soc: {
                    position: 'left',
                    min: 0,
                    max: 100,
                    grid: { drawOnChartArea: false },
                    offset: true,
                    title: { display: true, text: 'State of charge (%)' },
                },
                price: {
                    position: 'right',
                    grid: { drawOnChartArea: false },
                    title: { display: true, text: 'Price (€/kWh)' },
                    offset: true,
                    ticks: {
                        callback(value) {
                            return currencyFmt.format(value);
                        },
                    },
                },
            },
            plugins: {
                legend: {
                    display: true,
                    labels: {
                        filter: function (item, chart) {
                            // Hide datasets that are hidden
                            const ds = chart.datasets[item.datasetIndex];
                            if (ds.hidden) return false;
                            return true;
                        }
                    }
                },
                tooltip: {
                    callbacks: {
                        afterLabel(context) {
                            if (!priceImputedFlags.length) return '';
                            const dataset = context.dataset || {};
                            if (dataset.yAxisID !== 'price') return '';
                            return priceImputedFlags[context.dataIndex] ? 'Imputed from previous day' : '';
                        },
                    },
                },
            },
        },
    });
}

function ensureGridChart() {
    if (gridChart) return;
    const ctx = document.getElementById('gridChart');
    gridChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: [],
            datasets: [
                {
                    label: 'Grid import (kW)',
                    data: [],
                    backgroundColor: 'rgba(37, 99, 235, 0.75)',
                    borderRadius: 2,
                },
                {
                    label: 'Grid export (kW)',
                    data: [],
                    backgroundColor: 'rgba(249, 115, 22, 0.75)',
                    borderRadius: 2,
                },
            ],
        },
        options: {
            locale: activeLocale || undefined,
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    ticks: { maxRotation: 0, autoSkip: true },
                    stacked: true, // Stack columns to maximize width (user requested overlap/full width)
                },
                y: {
                    stacked: true,
                    title: { display: true, text: 'Grid power (kW)' },
                    ticks: {
                        callback(value) {
                            return `${value >= 0 ? '' : '-'}${Math.abs(value)}`;
                        },
                    },
                    grid: {
                        color(context) {
                            if (context.tick && context.tick.value === 0) {
                                return 'rgba(148, 163, 184, 0.6)';
                            }
                            return 'rgba(148, 163, 184, 0.18)';
                        },
                    },
                },
                yRight: {
                    position: 'right', // Align with the Price axis of the chart above
                    display: true,
                    grid: { drawOnChartArea: false },
                    title: {
                        display: true,
                        text: 'Price (€/kWh)', // Same text length as upper chart 
                        color: 'transparent' // Invisible
                    },
                    ticks: {
                        display: true,
                        color: 'transparent',
                        callback: function () { return ' .00 '; } // Approximate tick width
                    }
                },
            },
            plugins: {
                legend: { display: true },
                tooltip: {
                    callbacks: {
                        label(context) {
                            const value = Math.abs(Number(context.parsed.y || 0));
                            const label = context.dataset && context.dataset.label ? context.dataset.label : '';
                            return `${label}: ${numberFmt2.format(value)} kW`;
                        },
                    },
                },
            },
        },
    });
}

function ensureDeferrableLoadChart() {
    if (deferrableLoadChart) return;
    const ctx = document.getElementById('deferrableLoadChart');
    if (!ctx) return; // Element might not exist if HTML mismatch

    deferrableLoadChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: [],
            datasets: [],
        },
        options: {
            locale: activeLocale || undefined,
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    ticks: { maxRotation: 0, autoSkip: true },
                    stacked: true,
                },
                y: {
                    stacked: true,
                    title: { display: true, text: 'Power (kW)' },
                    beginAtZero: true,
                },
            },
            plugins: {
                legend: { display: true },
                tooltip: {
                    callbacks: {
                        label(context) {
                            const value = Number(context.parsed.y || 0);
                            const label = context.dataset.label || '';
                            return `${label}: ${numberFmt2.format(value)} kW`;
                        }
                    }
                }
            },
        },
    });
}

function applyForecast(payload) {
    const series = payload.series || {};
    const forecastMessage = document.getElementById('forecastMessage');
    updateFormattingContext(payload.locale, payload.timezone);
    document.getElementById('lastUpdated').textContent = formatDateTime(payload.timestamp);

    ensureForecastChart();
    ensurePlanChart();
    ensureGridChart();

    const timestamps = Array.isArray(series.timestamps) ? series.timestamps : [];
    const labels = timestamps.map(formatTickLabel);
    const zeros = new Array(labels.length).fill(0);

    forecastChart.data.labels = labels;
    forecastChart.data.datasets[0].data = sanitizeSeries(series.pv_kw || zeros);
    forecastChart.data.datasets[1].data = sanitizeSeries(series.load_kw || zeros);
    const priceImputedRaw = Array.isArray(series.price_imputed) ? series.price_imputed : [];
    priceImputedFlags = labels.map((_, idx) => Boolean(priceImputedRaw[idx]));
    forecastChart.update();

    const importKw = Array.isArray(series.grid_import_kw) ? series.grid_import_kw : zeros;
    const exportKw = Array.isArray(series.grid_export_kw) ? series.grid_export_kw : zeros;
    const socSeries = Array.isArray(series.soc) ? series.soc : zeros;
    const importPriceRaw = Array.isArray(series.price_import_eur_per_kwh)
        ? series.price_import_eur_per_kwh
        : Array.isArray(series.price_eur_per_kwh)
            ? series.price_eur_per_kwh
            : zeros;
    const exportPriceRaw = Array.isArray(series.price_export_eur_per_kwh)
        ? series.price_export_eur_per_kwh
        : Array.isArray(series.price_spot_eur_per_kwh)
            ? series.price_spot_eur_per_kwh
            : zeros;

    const sanitizedImport = sanitizeSeries(importKw);
    const sanitizedExport = sanitizeSeries(exportKw);
    const sanitizedSoc = sanitizeSeries(socSeries, 100);
    const sanitizedImportPrice = sanitizeSeries(importPriceRaw);
    const sanitizedExportPrice = sanitizeSeries(exportPriceRaw);
    const segmentImputed = (ctx) => {
        if (!ctx) {
            return false;
        }
        const leftIdx = typeof ctx.p0DataIndex === 'number' ? ctx.p0DataIndex : 0;
        const rightIdx = typeof ctx.p1DataIndex === 'number' ? ctx.p1DataIndex : leftIdx;
        return Boolean(priceImputedFlags[leftIdx] || priceImputedFlags[rightIdx]);
    };

    planChart.data.labels = labels;
    planChart.data.datasets[0].data = sanitizedExportPrice;
    planChart.data.datasets[1].data = sanitizedImportPrice;
    planChart.data.datasets[2].data = sanitizedSoc;

    // Hide export price if export is disabled
    const exportEnabled = pricingSettings && pricingSettings.export_enabled !== false;
    planChart.data.datasets[0].hidden = !exportEnabled;

    planChart.data.datasets[0].segment = {
        borderDash: (ctx) => (segmentImputed(ctx) ? [6, 4] : []),
    };
    planChart.data.datasets[1].segment = {
        borderDash: (ctx) => (segmentImputed(ctx) ? [6, 4] : []),
    };
    const imputedRadiusValues = labels.map((_, idx) => (priceImputedFlags[idx] ? 2 : 0));
    planChart.data.datasets[0].pointRadius = 0;
    planChart.data.datasets[1].pointRadius = 0;
    planChart.data.datasets[0].pointHoverRadius = 0;
    planChart.data.datasets[1].pointHoverRadius = 0;
    planChart.data.datasets[0].pointBackgroundColor = labels.map(() => '#f97316');
    planChart.data.datasets[1].pointBackgroundColor = labels.map(() => '#2563eb');
    planChart.data.datasets[2].pointRadius = 0;
    planChart.data.datasets[2].pointBackgroundColor = '#a855f7';
    planChart.update();

    gridChart.data.labels = labels;
    gridChart.data.datasets[0].data = sanitizedImport;
    gridChart.data.datasets[1].data = sanitizedExport;
    gridChart.update();

    // -- Update Deferrable Load Chart --
    ensureDeferrableLoadChart();
    if (deferrableLoadChart) {
        deferrableLoadChart.data.labels = labels;

        // Define palette
        const palette = ['#a855f7', '#06b6d4', '#eab308', '#ec4899', '#84cc16'];

        // Build datasets
        const datasets = [];
        const loads = (haEntitiesSettings && Array.isArray(haEntitiesSettings.deferrable_loads))
            ? haEntitiesSettings.deferrable_loads
            : [];

        // We iterate based on potential columns `def_load_0`, `def_load_1`... 
        // We assume up to 10 loads or use the settings length
        const maxCheck = loads.length > 0 ? loads.length : 5;

        for (let i = 0; i < maxCheck; i++) {
            const load = loads[i];
            // Robust check for disabled state
            if (load && (load.enabled === false || String(load.enabled) === 'false')) continue;

            let colName = `deferrable_${i}_kw`; // Fallback

            if (load && load.name) {
                // Match Python: "".join(c if c.isalnum() else "_" for c in dload.name)
                const safeName = load.name.replace(/[^a-z0-9]/gi, '_');
                colName = `load_${safeName}_kw`;
            }

            const rawData = series[colName];

            // Only add dataset if data exists in the payload (implies it was part of optimization)
            if (!rawData) continue;

            const dataValues = sanitizeSeries(rawData || zeros);
            const name = (load && load.name) ? load.name : `Load ${i + 1}`;

            datasets.push({
                label: name,
                data: dataValues,
                backgroundColor: palette[i % palette.length],
                borderRadius: 2,
            });
        }

        deferrableLoadChart.data.datasets = datasets;
        deferrableLoadChart.update();
    }

    updateSummary(payload.summary);
    forecastMessage.textContent = '';
    forecastMessage.classList.remove('error');

    const summaryWindowEl = document.getElementById('summaryWindow');
    if (summaryWindowEl) {
        summaryWindowEl.textContent = ''; // Removed horizon text per user request
    }

    renderIntervention(payload.intervention || null);
}

async function refreshStatus() {
    try {
        const payload = await fetchJson('api/status');
        applyStatus(payload);
        return payload; // Return payload for upstream logic
    } catch (err) {
        const dot = document.querySelector('#status .dot');
        const label = document.getElementById('statusLabel');
        const errorEl = document.getElementById('error'); // Use global error element

        if (dot) dot.classList.remove('ok');
        if (label) label.textContent = 'Failed to reach API';
        // Only show connectivity error if dot/label exist, or updating errorEl if found
        if (errorEl) errorEl.textContent = err instanceof Error ? err.message : String(err);
        return null;
    }
}

// --- Inverter Control Logic ---

let availableDrivers = [];
let currentDriverConfig = {};

async function loadControlSettings() {
    try {
        const [driversPayload, configPayload] = await Promise.all([
            fetchJson('api/drivers'),
            fetchJson('api/settings/inverter-driver')
        ]);
        availableDrivers = driversPayload.drivers;
        currentDriverConfig = configPayload || {};
        renderControlForm();
    } catch (err) {
        console.error("Failed to load control settings:", err);
        const msg = document.getElementById('controlMessage');
        if (msg) {
            msg.textContent = "Failed to load settings: " + err.message;
            msg.style.display = 'block';
            msg.classList.add('error');
        }
    }
}

function renderControlForm() {
    const select = document.getElementById('driverSelect');
    if (!select) return;

    select.innerHTML = '<option value="">Select a driver...</option>';
    availableDrivers.forEach(d => {
        const opt = document.createElement('option');
        opt.value = d.id;
        opt.textContent = d.name;
        select.appendChild(opt);
    });

    if (currentDriverConfig.driver_id) {
        select.value = currentDriverConfig.driver_id;
    }

    renderHAEntitiesTable();

    select.removeEventListener('change', renderHAEntitiesTable);
    select.addEventListener('change', () => {
        renderHAEntitiesTable();
        scheduleControlSave();
    });
}

function renderHAEntitiesTable() {
    const select = document.getElementById('driverSelect');
    const container = document.getElementById('haEntitiesTableContainer');
    if (!select || !container) return;

    const driverId = select.value;

    // 1. Structure Setup (Lazy Creation)
    let table = container.querySelector('.driver-entity-table');
    let tbody;

    // Ensure Table
    if (!table) {
        if (!driverId) {
            container.innerHTML = '';
            return;
        }

        table = document.createElement('table');
        table.className = 'driver-entity-table';
        table.innerHTML = `
            <thead>
                <tr>
                    <th>Entity</th>
                    <th>Domain</th>
                    <th>Entity ID</th>
                </tr>
            </thead>
            <tbody></tbody>
        `;
        if (container.firstChild) container.insertBefore(table, container.firstChild);
        else container.appendChild(table);
    }
    tbody = table.querySelector('tbody');

    // 2. Prepare Data
    let driver = null;
    let requiredEntities = [];
    if (driverId) {
        driver = availableDrivers.find(d => d.id === driverId);
        if (driver && driver.required_entities) {
            requiredEntities = driver.required_entities;
        }
    } else {
        container.innerHTML = '';
        return;
    }

    // 3. Reconcile Table Rows
    Array.from(tbody.children).forEach((row, i) => {
        if (i >= requiredEntities.length) row.remove();
    });

    requiredEntities.forEach((entityObj, index) => {
        let row = tbody.children[index];
        if (!row) {
            row = document.createElement('tr');
            row.innerHTML = '<td></td><td class="driver-entity-domain"></td><td><button type="button" class="driver-entity-button"></button></td>';
            const btn = row.querySelector('button');
            btn.addEventListener('click', () => {
                if (haEntitiesBusy) return;
                openEntityModalForDriver(entityObj.key, entityObj.domain, entityObj.default, (selectedId) => {
                    if (!currentDriverConfig.entity_map) currentDriverConfig.entity_map = {};
                    currentDriverConfig.entity_map[entityObj.key] = selectedId;
                    renderHAEntitiesTable();
                    scheduleControlSave();
                });
            });
            tbody.appendChild(row);
        }

        // Update Content
        const curLabel = entityObj.label || entityObj.key.replace(/_/g, ' ');
        if (row.children[0].textContent !== curLabel) row.children[0].textContent = curLabel;
        if (row.children[1].textContent !== entityObj.domain) row.children[1].textContent = entityObj.domain;

        const button = row.querySelector('button');
        if (haEntitiesBusy) button.disabled = true;
        else button.disabled = false;

        let currentVal = currentDriverConfig.entity_map ? currentDriverConfig.entity_map[entityObj.key] : null;

        if (!currentVal && ['house_consumption', 'pv_power', 'soc_sensor'].includes(entityObj.key)) {
            if (haEntitiesSettings && haEntitiesSettings[entityObj.key]) {
                currentVal = haEntitiesSettings[entityObj.key].entity_id;
                if (!currentDriverConfig.entity_map) currentDriverConfig.entity_map = {};
                currentDriverConfig.entity_map[entityObj.key] = currentVal;
            }
        }
        if (!currentVal && entityObj.default) {
            currentVal = entityObj.default;
            if (!currentDriverConfig.entity_map) currentDriverConfig.entity_map = {};
            currentDriverConfig.entity_map[entityObj.key] = currentVal;
        }

        const displayValue = currentVal || 'Click to select...';
        if (button.textContent !== displayValue) button.textContent = displayValue;

        button.classList.remove('valid-default', 'valid-custom', 'invalid');
        if (!currentVal || currentVal === 'Click to select...') {
            button.classList.add('invalid');
        } else {
            const exists = entityCatalog.some(e => e.entity_id === currentVal);
            const matchesDomain = currentVal.startsWith(entityObj.domain + '.');
            if (exists && matchesDomain) {
                const isDefault = currentVal === entityObj.default;
                button.classList.add(isDefault ? 'valid-default' : 'valid-custom');
            } else {
                button.classList.add('invalid');
            }
        }
    });

    // 4. Driver Config Reconciliation
    let configHeader = container.querySelector('h4');
    let grid = container.querySelector('.driver-config-grid');

    const hasConfig = driver && driver.config_schema && Object.keys(driver.config_schema).length > 0;

    if (!hasConfig) {
        if (configHeader) configHeader.remove();
        if (grid) grid.remove();
    } else {
        if (!configHeader) {
            configHeader = document.createElement('h4');
            configHeader.textContent = "Driver Configuration";
            configHeader.style.marginTop = "1.5rem";
            configHeader.style.marginBottom = "0.5rem";
            container.appendChild(configHeader);
        }
        if (!grid) {
            grid = document.createElement('div');
            grid.className = 'driver-config-grid';
            container.appendChild(grid);
        }

        const schemaItems = driver.config_schema;

        Array.from(grid.children).forEach((field, i) => {
            if (i >= schemaItems.length) field.remove();
        });

        schemaItems.forEach((schemaItem, index) => {
            const key = schemaItem.key;
            const type = schemaItem.type;

            let field = grid.children[index];
            if (!field) {
                field = document.createElement('div');
                field.className = 'field';

                const label = document.createElement('label');
                field.appendChild(label);

                const input = document.createElement('input');
                if (type === 'float' || type === 'int') {
                    input.type = 'number';
                    input.step = type === 'float' ? '0.1' : '1';
                } else {
                    input.type = 'text';
                }

                input.onchange = (e) => {
                    if (!currentDriverConfig.config) currentDriverConfig.config = {};
                    let v = e.target.value;
                    if (type === 'float') v = parseFloat(v);
                    if (type === 'int') v = parseInt(v, 10);
                    currentDriverConfig.config[key] = v;
                    scheduleControlSave();
                };

                field.appendChild(input);
                grid.appendChild(field);
            }

            const labelEl = field.querySelector('label');
            const properLabel = schemaItem.label || key.replace(/_/g, ' ').replace(/\w/g, l => l.toUpperCase());
            if (labelEl.textContent !== properLabel) labelEl.textContent = properLabel;

            const input = field.querySelector('input');
            const val = currentDriverConfig.config ? currentDriverConfig.config[key] : undefined;
            const effectiveVal = val !== undefined && val !== '' ? val : (schemaItem.default !== undefined ? schemaItem.default : '');

            setSafeValue(input, effectiveVal);

            if (val === undefined && schemaItem.default !== undefined) {
                if (!currentDriverConfig.config) currentDriverConfig.config = {};
                currentDriverConfig.config[key] = schemaItem.default;
            }
        });
    }
}

// Helper to bridge the existing entity modal
let activeDriverEntityCallback = null;
let activeDriverEntityDomain = null;
let activeDriverEntityDefault = null;

function openEntityModalForDriver(entityKey, domain, defaultValue, callback) {
    activeDriverEntityCallback = callback;
    activeDriverEntityDomain = domain; // Store the domain filter
    activeDriverEntityDefault = defaultValue; // Store default value
    openEntityModal('DRIVER_CONFIG');
}

// We need to hook into `selectEntity` or `updateEntitySelection` to handle our callback.
// Since we can't easily modify those without replacing them, let's look at where they are.
// They are not in the viewed range. I will assume I can modify `selectEntity` or `updateEntitySelection` later.
// For now, let's assume I can add a check in `updateEntitySelection`.

async function saveControlSettings() {
    const select = document.getElementById('driverSelect');
    const driverId = select.value;
    if (!driverId) return;

    // Use a quieter indicator since we auto-save
    // If message is 'Saving changes...', keep it. If empty/null, maybe set it.

    try {
        // Sync special entities to global settings if present in driver config
        const specialKeys = ['house_consumption', 'pv_power', 'soc_sensor'];
        const globalUpdates = {};
        let hasGlobalUpdates = false;

        const driver = availableDrivers.find(d => d.id === driverId);

        specialKeys.forEach(key => {
            let valToSync = null;

            // 1. Check explicit override in map
            if (currentDriverConfig.entity_map && currentDriverConfig.entity_map[key]) {
                valToSync = currentDriverConfig.entity_map[key];
            }
            // 2. If no override, check default from driver (if available)
            // This ensures that if we clear an override, we revert the global setting to the default,
            // preventing the global setting from "sticking" to the old custom value.
            else if (driver && driver.required_entities) {
                const ent = driver.required_entities.find(e => e.key === key);
                if (ent && ent.default) {
                    valToSync = ent.default;
                }
            }

            if (valToSync) {
                globalUpdates[key] = { entity_id: valToSync };
                hasGlobalUpdates = true;
            }
        });

        if (hasGlobalUpdates) {
            await fetchJson('api/settings', {
                method: 'PATCH',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ ha_entities: globalUpdates }),
            });
        }

        await fetchJson('api/settings/inverter-driver', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                driver_id: driverId,
                entity_map: currentDriverConfig.entity_map || {},
                config: currentDriverConfig.config || {}
            })
        });
        setHAEntitiesMessage('Saved configuration.', 'success');
        setTimeout(() => {
            // Clear success message after a bit to keep UI clean
            const msgEl = document.getElementById('controlMessage');
            if (msgEl && msgEl.textContent === 'Saved configuration.') {
                msgEl.style.display = 'none';
                msgEl.textContent = '';
            }
        }, 3000);
    } catch (err) {
        setHAEntitiesMessage('Failed to save: ' + err.message, 'error');
    } finally {
        controlSaveTimer = null;
    }
}

async function refreshForecast() {
    const forecastMessage = document.getElementById('forecastMessage');
    // Removed "Loading latest forecast..." per user request to avoid blinking
    if (forecastMessage) forecastMessage.classList.remove('error');
    try {
        const payload = await fetchJson('api/forecast');
        applyForecast(payload);
    } catch (err) {
        let msg = err instanceof Error ? err.message : String(err);

        // Try to parse JSON error (fastapi returns {"detail": "..."})
        if (msg.startsWith('{')) {
            try {
                const parsed = JSON.parse(msg);
                if (parsed.detail) msg = parsed.detail;
            } catch (e) { /* ignore */ }
        }

        // Special handling for "Forecast not ready" (503) during startup
        // Avoid "Current intervention unavailable" scary red text, just show info.
        if (msg.includes('Forecast not ready')) {
            if (forecastMessage) {
                forecastMessage.textContent = 'Forecast is being generated...';
                forecastMessage.classList.remove('error');
            }
            // Don't flash intervention banner with error
            renderIntervention(null, 'Waiting for forecast...');
            // Optionally add .none class to banner to make it gray instead of red/orange
            const banner = document.getElementById('interventionBanner');
            if (banner) {
                banner.className = 'intervention-banner none';
            }
        } else {
            if (forecastMessage) {
                forecastMessage.textContent = msg;
                forecastMessage.classList.add('error');
            }
            renderIntervention(null, `Current intervention unavailable: ${msg}`);
        }
    }
}

async function refreshAll() {
    if (refreshAllPromise) {
        return refreshAllPromise;
    }

    refreshAllPromise = (async () => {
        try {
            // 1. Fetch Status first
            const statusPayload = await refreshStatus();

            // 2. Check if we need to fetch forecast
            let shouldFetchForecast = false;

            if (statusPayload && statusPayload.snapshot_available && statusPayload.last_updated) {
                // Backend reports a specific timestamp
                if (statusPayload.last_updated !== lastForecastTimestamp) {
                    shouldFetchForecast = true;
                    lastForecastTimestamp = statusPayload.last_updated;
                }
            } else {
                // Forecast not available. Do NOT poll /api/forecast as it will return 503.
                shouldFetchForecast = false;

                // If we are in this state, ensuring no error is displayed
                const forecastMessage = document.getElementById('forecastMessage');
                if (forecastMessage && (!statusPayload || !statusPayload.snapshot_available)) {
                    // Update text only if it was showing an error
                    if (forecastMessage.classList.contains('error')) {
                        forecastMessage.classList.remove('error');
                        forecastMessage.textContent = 'Results will appear here after training.';
                    }
                }
            }

            if (shouldFetchForecast) {
                await refreshForecast();
            }
        } catch (err) {
            console.error("Refresh sequence failed:", err);
        } finally {
            refreshAllPromise = null;
        }
    })();

    return refreshAllPromise;
}

async function triggerTraining() {
    const button = document.getElementById('trainBtn');
    button.disabled = true;
    button.textContent = 'Starting…';
    try {
        await fetchJson('api/training', { method: 'POST' });
    } catch (err) {
        const errorEl = document.getElementById('trainingError');
        errorEl.style.display = 'block';
        errorEl.textContent = err instanceof Error ? `Failed to start training: ${err.message}` : String(err);
    }
    await refreshAll();
}

async function triggerCycle() {
    const button = document.getElementById('cycleBtn');
    const messageEl = document.getElementById('cycleMessage');
    if (messageEl) {
        messageEl.style.display = 'none';
        messageEl.classList.remove('error');
        messageEl.textContent = '';
        delete messageEl.dataset.auto;
    }
    if (button) {
        button.disabled = true;
        button.textContent = 'Starting…';
    }
    try {
        await fetchJson('api/cycle', { method: 'POST' });
        if (messageEl) {
            messageEl.style.display = 'block';
            messageEl.textContent = 'Optimization started…';
        }
    } catch (err) {
        if (messageEl) {
            messageEl.style.display = 'block';
            messageEl.classList.add('error');
            messageEl.textContent = err instanceof Error ? err.message : String(err);
        }
        if (button) {
            // applyStatus will handle state, but ensure it's not "Starting..." if we failed
            button.disabled = false;
        }
    }
    await refreshAll();
}

// --------------------------------------------------------------------------
// Deferrable Loads Logic
// --------------------------------------------------------------------------

function initDeferrableLoads() {
    const addBtn = document.getElementById('addLoadBtn');
    const container = document.getElementById('deferrableLoadsList');

    if (addBtn) {
        addBtn.addEventListener('click', () => {
            if (!haEntitiesSettings) haEntitiesSettings = {};
            if (!haEntitiesSettings.deferrable_loads) haEntitiesSettings.deferrable_loads = [];
            haEntitiesSettings.deferrable_loads.push({
                name: '',
                nominal_power_kw: 0.0,
                switch_entity: '',
                power_entity: null,
                opt_mode: 'smart_dump',
                opt_value_start_eur: 0.05,
                opt_saturation_kwh: 10.0,
                opt_saturation_kwh: 10.0,
                opt_prevent_overshoot: false,
                enabled: true
            });
            renderDeferrableLoads();

            // Expand the new load automatically
            const container = document.getElementById('deferrableLoadsList');
            if (container && container.lastElementChild) {
                container.lastElementChild.classList.add('expanded');
                const input = container.lastElementChild.querySelector('.load-name');
                if (input) input.focus();
            }

            scheduleSaveHAEntities();
        });
    }



    if (!container) return;

    // --- Unified Event Delegation ---

    // 0. Toggle Expansion
    container.addEventListener('click', (e) => {
        // Check if clicked header or toggle
        if (e.target.closest('.load-header') &&
            !e.target.closest('.remove-load-btn') &&
            !e.target.closest('.toggle-switch-mini')) {
            const row = e.target.closest('.load-item');
            if (row) {
                row.classList.toggle('expanded');
            }
        }
    });

    // 1. Inputs (Text/Number): Handle immediate changes
    container.addEventListener('input', (e) => {
        const row = e.target.closest('.load-item');
        if (!row) return;
        const index = parseInt(row.getAttribute('data-index'));
        if (isNaN(index)) return;

        const load = haEntitiesSettings.deferrable_loads[index];
        if (!load) return;

        if (e.target.classList.contains('load-name')) {
            load.name = e.target.value;
            // Update Header Title immediately
            const titleEl = row.querySelector('.load-header-title');
            if (titleEl) titleEl.textContent = load.name || 'New Load';

            if (!load.name) e.target.classList.add('error');
            else e.target.classList.remove('error');
        }

        else if (e.target.classList.contains('load-nominal-power')) {
            load.nominal_power_kw = parseFloat(e.target.value) || 0;
            if (!e.target.value || parseFloat(e.target.value) <= 0) e.target.classList.add('error');
            else e.target.classList.remove('error');
        }
        else if (e.target.classList.contains('load-opt-start-value')) {
            load.opt_value_start_eur = parseFloat(e.target.value) || 0;
        }
        else if (e.target.classList.contains('load-opt-saturation')) {
            load.opt_saturation_kwh = parseFloat(e.target.value) || 0;
        }

        // Debounced save
        scheduleSaveHAEntities();
    });

    // 2. Selects / Checkboxes ('change' event)
    container.addEventListener('change', (e) => {
        const row = e.target.closest('.load-item');
        if (!row) return;
        const index = parseInt(row.getAttribute('data-index'));
        if (isNaN(index)) return;

        const load = haEntitiesSettings.deferrable_loads[index];
        if (!load) return;

        if (e.target.classList.contains('load-opt-mode')) {
            load.opt_mode = e.target.value;
            // Update UI visibility immediately
            const customSettings = row.querySelector('.custom-curve-settings');
            if (customSettings) {
                customSettings.style.display = load.opt_mode === 'custom_curve' ? 'block' : 'none';
            }
        }
        else if (e.target.classList.contains('load-opt-overshoot')) {
            load.opt_prevent_overshoot = e.target.checked;
        }
        else if (e.target.classList.contains('load-enabled-toggle')) {
            load.enabled = e.target.checked;
        }

        scheduleSaveHAEntities();
    });

    // 3. Clicks (Buttons: Remove, Entity Select)
    container.addEventListener('click', (e) => {
        const row = e.target.closest('.load-item');
        if (!row) return;
        const index = parseInt(row.getAttribute('data-index'));
        if (isNaN(index)) return;

        // Remove
        if (e.target.closest('.remove-load-btn')) {
            if (confirm('Remove this load?')) {
                haEntitiesSettings.deferrable_loads.splice(index, 1);
                renderDeferrableLoads(true); // Force re-render/cleanup
                scheduleSaveHAEntities();
            }
            return;
        }

        const load = haEntitiesSettings.deferrable_loads[index];
        if (!load) return;

        // Switch Entity Select
        if (e.target.closest('.load-switch-entity')) {
            openEntityModal('GENERIC_SWITCH');
            activeDriverEntityCallback = (entityId) => {
                // Fetch fresh load object
                const currentLoad = haEntitiesSettings.deferrable_loads[index];
                if (currentLoad) {
                    currentLoad.switch_entity = entityId || '';
                    renderDeferrableLoads(); // Update button text
                    scheduleSaveHAEntities();
                }
            };
        }
        // Power Entity Select
        else if (e.target.closest('.load-power-entity')) {
            openEntityModal('GENERIC_POWER');
            activeDriverEntityCallback = (entityId) => {
                const currentLoad = haEntitiesSettings.deferrable_loads[index];
                if (currentLoad) {
                    if (entityId) currentLoad.power_entity = { entity_id: entityId, unit: null };
                    else currentLoad.power_entity = null;
                    renderDeferrableLoads();
                    scheduleSaveHAEntities();
                }
            };
        }
    });
}


function resolveEntityLabel(entityId) {
    if (!entityId) return '';
    // Check all catalogs
    const all = [].concat(entityCatalog || [], powerEntityCatalog || [], batteryEntityCatalog || []);
    const match = all.find(e => e.entity_id === entityId);
    if (match && match.name) {
        return `${match.name} (${entityId})`;
    }
    return entityId;
}

function renderDeferrableLoads(force = false) {
    const container = document.getElementById('deferrableLoadsList');
    if (!container) return;

    if (force) {
        container.innerHTML = '';
    }

    const loads = (haEntitiesSettings && haEntitiesSettings.deferrable_loads) ? haEntitiesSettings.deferrable_loads : [];
    const template = document.getElementById('loadItemTemplate');
    if (!template) return;

    // Handle empty state
    if (loads.length === 0) {
        if (container.children.length === 0 || container.querySelector('.message')) {
            container.innerHTML = '<div class="message">No loads defined.</div>';
        } else {
            container.innerHTML = '<div class="message">No loads defined.</div>';
        }
        return;
    }

    // Remove "No loads defined" message if present
    const msg = container.querySelector('.message');
    if (msg) msg.remove();

    // 1. Reconcile Rows (Remove extra rows)
    Array.from(container.children).forEach((child, i) => {
        if (i >= loads.length) {
            child.remove();
        }
    });

    // 2. Update/Create Rows
    loads.forEach((load, index) => {
        let row = container.children[index];

        // Create if missing
        if (!row) {
            const clone = template.content.cloneNode(true);
            row = clone.querySelector('.load-item');
            if (!row) return;

            row.setAttribute('data-index', index);
            container.appendChild(clone);
            // Get live reference
            row = container.children[index];
        } else {
            // Update index attribute in case of shifts
            row.setAttribute('data-index', index);
        }

        // --- Update Header Title ---
        const titleEl = row.querySelector('.load-header-title');
        if (titleEl) {
            const titleText = load.name || 'New Load';
            if (titleEl.textContent !== titleText) titleEl.textContent = titleText;
        }

        // --- Update Functions (Safety Checks) ---
        const updateInput = (selector, val) => {
            const el = row.querySelector(selector);
            if (el && document.activeElement !== el) {
                if (el.value != val) el.value = val;
            }
            if (el && selector === '.load-name') {
                if (!val) el.classList.add('error');
                else el.classList.remove('error');
            }
            if (el && selector === '.load-nominal-power') {
                if (!val || parseFloat(val) <= 0) el.classList.add('error');
                else el.classList.remove('error');
            }
        };

        const updateCheck = (selector, val) => {
            const el = row.querySelector(selector);
            if (el && document.activeElement !== el) el.checked = Boolean(val);
        };



        // --- Apply Values ---
        updateCheck('.load-enabled-toggle', load.enabled !== false); // Default True
        updateInput('.load-name', load.name || '');
        updateInput('.load-nominal-power', load.nominal_power_kw);
        updateInput('.load-opt-start-value', (load.opt_value_start_eur !== undefined) ? load.opt_value_start_eur : 0.05);
        updateInput('.load-opt-saturation', (load.opt_saturation_kwh !== undefined) ? load.opt_saturation_kwh : 10.0);
        updateCheck('.load-opt-overshoot', load.opt_prevent_overshoot);

        // Mode Select & Visibility
        const modeSel = row.querySelector('.load-opt-mode');
        if (modeSel && document.activeElement !== modeSel) {
            modeSel.value = load.opt_mode || 'smart_dump';
            const customSettings = row.querySelector('.custom-curve-settings');
            if (customSettings) {
                // Ensure visibility matches mode
                // Note: user might be changing it right now via 'change' event which updates 'load'
                // But if this is a background update, we enforce it.
                // If user just changed it, load.opt_mode is updated, so this is safe.
                customSettings.style.display = modeSel.value === 'custom_curve' ? 'block' : 'none';
            }
        }

        // --- Buttons (Text Only) ---
        const switchBtn = row.querySelector('.load-switch-entity');
        if (switchBtn) {
            const currentId = load.switch_entity || '';
            const label = resolveEntityLabel(currentId) || 'Select switch...';
            if (switchBtn.textContent !== label) switchBtn.textContent = label;

            if (!currentId) switchBtn.classList.add('error');
            else switchBtn.classList.remove('error');
        }

        const powerBtn = row.querySelector('.load-power-entity');
        if (powerBtn) {
            const currentId = (load.power_entity && load.power_entity.entity_id) ? load.power_entity.entity_id : '';
            const label = resolveEntityLabel(currentId) || 'Select sensor...';
            if (powerBtn.textContent !== label) powerBtn.textContent = label;
        }

        // Remove Button (just needs to exist, delegation handles click)
        const removeBtn = row.querySelector('.remove-load-btn');
        if (removeBtn) removeBtn.setAttribute('data-index', index);
    });
}

function populateEntitySelect(select, catalog, currentVal, includeNone = false) {
    select.innerHTML = '';
    if (includeNone) {
        const opt = document.createElement('option');
        opt.value = '';
        opt.textContent = 'None';
        select.appendChild(opt);
    }

    const sorted = [...catalog].sort((a, b) => (a.name || a.entity_id).localeCompare(b.name || b.entity_id));

    sorted.forEach(ent => {
        const opt = document.createElement('option');
        opt.value = ent.entity_id;
        opt.textContent = `${ent.name || ent.entity_id} (${ent.entity_id})`;
        if (ent.entity_id === currentVal) opt.selected = true;
        select.appendChild(opt);
    });

    if (currentVal && !sorted.find(c => c.entity_id === currentVal) && currentVal !== '') {
        const opt = document.createElement('option');
        opt.value = currentVal;
        opt.textContent = `Missing: ${currentVal}`;
        opt.selected = true;
        select.insertBefore(opt, select.firstChild);
    }
}

let haEntitiesSaveTimer = null;
let haEntitiesPendingResave = false;

function scheduleSaveHAEntities() {
    if (haEntitiesSaveTimer) clearTimeout(haEntitiesSaveTimer);
    setHAEntitiesMessage('Saving changes...', 'pending');
    haEntitiesSaveTimer = setTimeout(saveHAEntitiesSettings, 1000);
}

async function saveHAEntitiesSettings() {
    if (!haEntitiesSettings) return;



    // If already saving, mark that we need to save again (queuing the latest state)
    if (haEntitiesBusy) {
        console.warn("Debug: haEntitiesBusy is TRUE. Returning and queuing.");
        haEntitiesPendingResave = true;
        return;
    }

    haEntitiesBusy = true;
    setEntityTriggersDisabled(true);
    try {
        console.log("Debug: About to call fetchJson('api/settings', PATCH)...");
        const payload = await fetchJson('api/settings', {
            method: 'PATCH',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ ha_entities: haEntitiesSettings }),
        });
        console.log("Debug: fetchJson returned successfully.");

        // Only apply updates if we don't have new pending local changes
        if (!haEntitiesPendingResave) {
            applySettingsPayload(payload);
        }

        setHAEntitiesMessage('Saved.', 'success', 2000);
        setHAEntitiesMessage('Saved.', 'success', 2000);
    } catch (err) {
        console.error("Save failed:", err);
        setHAEntitiesMessage("Save failed: " + (err instanceof Error ? err.message : String(err)), 'error');
    } finally {
        haEntitiesBusy = false;
        setEntityTriggersDisabled(false);

        // Process queued save
        if (haEntitiesPendingResave) {
            haEntitiesPendingResave = false;
            scheduleSaveHAEntities();
        }
    }
}


window.addEventListener('DOMContentLoaded', () => {
    bindClick(document.getElementById('refreshBtn'), refreshAll);
    bindClick(document.getElementById('trainBtn'), triggerTraining);
    bindClick(document.getElementById('cycleBtn'), triggerCycle);
    bindClick(document.getElementById('settingsBtn'), toggleSettings);
    initDeferrableLoads();
    document.querySelectorAll('.settings-nav button').forEach((btn) => {
        btn.addEventListener('click', () => {
            activateSettingsTab(btn.dataset.settingsTab);
        });
    });
    [
        [houseTrigger, 'house_consumption'],
        [pvTrigger, 'pv_power'],
        [batteryTrigger, 'battery_soc'],
    ].forEach(([trigger, target]) => {
        const el = typeof trigger === 'string' ? document.getElementById(trigger) : trigger;
        if (el) bindClick(el, () => openEntityModal(target));
    });
    Object.entries(batteryFieldConfigs).forEach(([fieldKey, config]) => {
        if (!config.input) {
            return;
        }
        const handler = () => handleBatteryNumericInput(fieldKey);
        config.input.addEventListener('change', handler);
        config.input.addEventListener('blur', handler);
    });
    if (exportLimitToggle) {
        exportLimitToggle.addEventListener('change', (event) => {
            updateExportLimitSetting(event.target.checked);
            scheduleControlSave();
        });
    }
    const exportEnabledToggle = document.getElementById('exportEnabledToggle');
    if (exportEnabledToggle) {
        exportEnabledToggle.addEventListener('change', (event) => {
            ensurePricingSettings();
            pricingSettings.export_enabled = event.target.checked;
            renderPricingForm();
            schedulePricingSave();
        });
    }
    const exportPowerLimitInput = document.getElementById('exportPowerLimit');
    if (exportPowerLimitInput) {
        const handler = (event) => {
            ensurePricingSettings();
            const raw = event.target.value.trim();
            if (!raw) {
                pricingSettings.export_limit_kw = null;
            } else {
                const val = Number(raw);
                pricingSettings.export_limit_kw = Number.isFinite(val) && val >= 0 ? val : null;
            }
            schedulePricingSave();
        };
        exportPowerLimitInput.addEventListener('change', handler);
        exportPowerLimitInput.addEventListener('blur', handler);
    }
    const autoTrainingToggle = document.getElementById('autoTrainingToggle');
    if (autoTrainingToggle) {
        autoTrainingToggle.addEventListener('change', (event) => {
            updateAutoTrainingSetting(event.target.checked);
        });
    }
    [entityModalBackdrop, entityModalClose, entityModalCancel].forEach((element) => {
        bindClick(element, closeEntityModal);
    });
    if (entitySearchInput) {
        entitySearchInput.addEventListener('input', (event) => {
            renderEntityList(event.target.value);
        });
    }
    [
        [importTariffModeSelect, 'import'],
        [exportTariffModeSelect, 'export'],
    ].forEach(([select, scope]) => {
        if (!select) return;
        select.addEventListener('change', (event) => {
            handleTariffModeChange(scope, event.target.value);
        });
    });
    bindTariffFields();
    window.addEventListener('keydown', (event) => {
        if (event.key === 'Escape' && entityModal && entityModal.classList.contains('open')) {
            closeEntityModal();
        }
    });

    renderPricingForm();
    refreshAll();
    renderPricingForm();
    refreshAll();
    loadSettingsData();
    loadControlSettings();
    updateViewMode();
    renderIntervention(null, 'Current intervention: Not available');
    activateSettingsTab('control');

    const controlSwitch = document.getElementById('controlSwitch');
    if (controlSwitch) {
        controlSwitch.addEventListener('click', async (e) => {
            e.preventDefault();
            const currentState = controlSwitch.getAttribute('aria-checked') === 'true';
            const newState = !currentState;

            // If enabling, check if driver is configured
            if (newState) {
                try {
                    const driverResp = await fetch('api/settings/inverter-driver');
                    const driverConfig = await driverResp.json();

                    if (!driverConfig || !driverConfig.driver_id) {
                        alert('Error: No inverter driver configured. Please configure a driver in Settings > Inverter Control before enabling automatic control.');
                        return;
                    }
                } catch (err) {
                    console.error('Failed to check driver configuration:', err);
                    alert('Error: Could not verify driver configuration. Please configure a driver in Settings > Inverter Control.');
                    return;
                }
            }

            try {
                const response = await fetch('api/control', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ active: newState })
                });
                if (response.ok) {
                    controlSwitch.setAttribute('aria-checked', newState.toString());
                    fetchStatus();
                } else {
                    console.error('Failed to toggle control');
                }
            } catch (err) {
                console.error('Error toggling control:', err);
            }
        });
    }

    setInterval(() => {
        // Don't poll if we are viewing settings (modal) OR if we are actively saving anything
        // This prevents connection pool exhaustion if saving takes a few seconds (e.g. waiting for HA)
        if (!showingSettings && !haEntitiesBusy && !batteryBusy && !pricingSaving) {
            refreshAll();
        }
    }, 1000); // 1s quick update interval per user request

    // Initial fetch to avoid blank screen
    refreshAll();
});
