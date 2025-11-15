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
let priceImputedFlags = [];
let showingSettings = false;
let statisticsCatalog = [];
let inverterSettings = null;
let inverterBusy = false;
let entityCatalog = [];
let powerEntityCatalog = [];
let batteryEntityCatalog = [];
let entityLookup = Object.create(null);
let entityModalTarget = null;
let lastEntityTrigger = null;
let lastTrainingRunning = false;
let lastCycleRunning = false;
let autoRefreshTimer = null;
let pendingForecastRefresh = null;
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
const inverterMessage = document.getElementById('inverterMessage');
const batteryMessage = document.getElementById('batteryMessage');
const batteryWearInput = document.getElementById('batteryWearInput');
const batteryWearHint = document.getElementById('batteryWearHint');
const pricingForm = document.getElementById('pricingForm');
const pricingMessage = document.getElementById('pricingMessage');
const savePricingBtn = document.getElementById('savePricingBtn');
const resetPricingBtn = document.getElementById('resetPricingBtn');
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
const entityModal = document.getElementById('entityModal');
const entityModalBackdrop = document.getElementById('entityModalBackdrop');
const entityModalClose = document.getElementById('entityModalClose');
const entityModalCancel = document.getElementById('entityModalCancel');
const entitySearchInput = document.getElementById('entitySearch');
const entityListContainer = document.getElementById('entityList');
const exportLimitToggle = document.getElementById('exportLimitToggle');
let batterySettings = null;
let batteryBusy = false;
let batteryMessageClearTimer = null;
let pricingSettings = null;
let pricingSaveTimer = null;
let pricingSaving = false;
let pricingPendingResave = false;
let pricingMessageClearTimer = null;
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
        loadSettingsData();
    } else {
        refreshStatus();
        refreshForecast();
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

    if (!intervention) {
        const text = fallbackText || 'Current intervention: Eco mode';
        el.classList.add('none');
        el.innerHTML = text;
        el.style.display = 'block';
        return;
    }

    const mode = intervention.mode || 'none';
    const power = Number(intervention.power_kw);
    const price = Number(intervention.price_eur_per_kwh);
    const soc = Number(intervention.soc);
    const reason = typeof intervention.reason === 'string' ? intervention.reason.trim() : '';

    let mainText = 'Current intervention: Eco mode';
    let modeClass = 'none';

    if (mode === 'charge') {
        if (Number.isFinite(power) && power > 0.05) {
            mainText = `Current intervention: Charge battery at ${numberFmt1.format(power)} kW`;
        } else {
            mainText = 'Current intervention: Charge battery';
        }
        modeClass = 'charge';
    } else if (mode === 'discharge') {
        if (Number.isFinite(power) && power > 0.05) {
            mainText = `Current intervention: Discharge battery at ${numberFmt1.format(power)} kW`;
        } else {
            mainText = 'Current intervention: Discharge battery';
        }
        modeClass = 'discharge';
    } else if (mode === 'limit_export') {
        if (Number.isFinite(power) && power > 0.05) {
            mainText = `Current intervention: Limit grid export to ${numberFmt1.format(power)} kW`;
        } else {
            mainText = 'Current intervention: Limit grid export';
        }
        modeClass = 'limit';
    } else {
        if (fallbackText) {
            mainText = fallbackText;
        }
        modeClass = 'none';
    }

    const detailParts = [];
    if (modeClass !== 'none' && Number.isFinite(price)) {
        detailParts.push(`${currencyFmt.format(price)} / kWh`);
    }
    if (modeClass !== 'none' && Number.isFinite(soc)) {
        detailParts.push(`SoC ${numberFmt2.format(soc * 100)} %`);
    }
    if (reason) {
        detailParts.push(reason);
    }

    el.classList.add(modeClass);
    el.style.display = 'block';
    if (detailParts.length) {
        el.innerHTML = `${mainText}<span class="detail">${detailParts.join(' • ')}</span>`;
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

function setInverterMessage(text, kind = 'info') {
    if (!inverterMessage) return;
    if (!text) {
        inverterMessage.style.display = 'none';
        inverterMessage.classList.remove('error');
        inverterMessage.textContent = '';
        return;
    }
    inverterMessage.style.display = 'block';
    if (kind === 'error') {
        inverterMessage.classList.add('error');
    } else {
        inverterMessage.classList.remove('error');
    }
    inverterMessage.textContent = text;
}

function setBatteryMessage(text, kind = 'info', autoHideMs = 0) {
    if (!batteryMessage) return;
    if (batteryMessageClearTimer) {
        clearTimeout(batteryMessageClearTimer);
        batteryMessageClearTimer = null;
    }
    if (!text) {
        batteryMessage.style.display = 'none';
        batteryMessage.textContent = '';
        batteryMessage.classList.remove('error');
        return;
    }
    batteryMessage.style.display = 'block';
    batteryMessage.textContent = text;
    if (kind === 'error') {
        batteryMessage.classList.add('error');
    } else {
        batteryMessage.classList.remove('error');
    }
    if (text && kind !== 'error' && autoHideMs > 0) {
        batteryMessageClearTimer = setTimeout(() => {
            batteryMessageClearTimer = null;
            setBatteryMessage('');
        }, autoHideMs);
    }
}

function renderInverterHints() {
    if (!inverterSettings) return;
    const catalogMap = Object.create(null);
    statisticsCatalog.forEach((item) => {
        if (item && item.statistic_id) {
            catalogMap[item.statistic_id] = item;
        }
    });
    const house = inverterSettings.house_consumption || {};
    const pv = inverterSettings.pv_power || {};
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
    };
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
    return settings;
}

function applySettingsPayload(payload) {
    if (Array.isArray(payload.statistics)) {
        statisticsCatalog = payload.statistics;
    }
    if (Array.isArray(payload.entities)) {
        refreshEntityCatalogs(payload.entities);
    }
    if (payload.settings) {
        if (payload.settings.inverter) {
            inverterSettings = payload.settings.inverter;
        }
        if (payload.settings.battery) {
            const incoming = payload.settings.battery;
            const wearRaw = Number(incoming.wear_cost_eur_per_kwh);
            const wearCost = Number.isFinite(wearRaw) && wearRaw >= 0 ? wearRaw : 0.1;
            batterySettings = {
                ...incoming,
                soc_sensor: incoming.soc_sensor || null,
                wear_cost_eur_per_kwh: wearCost,
            };
        } else if (!batterySettings) {
            batterySettings = { soc_sensor: null, wear_cost_eur_per_kwh: 0.1 };
        } else if (typeof batterySettings.wear_cost_eur_per_kwh !== 'number') {
            batterySettings.wear_cost_eur_per_kwh = 0.1;
        }
        if (payload.settings.pricing) {
            pricingSettings = normalizePricingSettings(payload.settings.pricing);
        }
    }
    renderInverterForm();
    renderBatteryForm();
    renderPricingForm();
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

function renderInverterForm() {
    if (!inverterSettings) return;
    updateEntityButton(
        houseTrigger,
        houseLabel,
        inverterSettings.house_consumption,
        'None selected',
        inverterBusy,
        powerEntityCatalog.length > 0,
    );
    updateEntityButton(
        pvTrigger,
        pvLabel,
        inverterSettings.pv_power,
        'None selected',
        inverterBusy,
        powerEntityCatalog.length > 0,
    );
    if (exportLimitToggle) {
        exportLimitToggle.checked = Boolean(inverterSettings.export_power_limited);
        if (inverterBusy) {
            exportLimitToggle.setAttribute('disabled', 'disabled');
        } else {
            exportLimitToggle.removeAttribute('disabled');
        }
    }
    renderInverterHints();
    const recorderStatuses = [
        inverterSettings.house_consumption && inverterSettings.house_consumption.recorder_status,
        inverterSettings.pv_power && inverterSettings.pv_power.recorder_status,
    ];
    if (recorderStatuses.every((value) => value === 'missing')) {
        setInverterMessage('No Home Assistant statistics were returned for the selected sensors. Verify long-term statistics are enabled.', 'error');
    } else if (!powerEntityCatalog.length) {
        setInverterMessage('No power sensors detected. Expose Home Assistant sensors with power readings to proceed.', 'error');
    }
}

function renderBatteryForm() {
    if (!batteryTrigger || !batterySettings) return;
    const selection = batterySettings.soc_sensor || null;
    updateEntityButton(
        batteryTrigger,
        batteryLabel,
        selection,
        'None selected',
        batteryBusy,
        batteryEntityCatalog.length > 0,
        true,
    );
    if (batteryHint) {
        if (!batteryEntityCatalog.length) {
            batteryHint.textContent = 'No Home Assistant sensors with battery state of charge were detected. Expose a sensor reporting % and retry.';
            batteryHint.classList.add('error');
            setBatteryMessage('No eligible battery sensors detected. Once a sensor publishes SoC in %, it will appear here.', 'error');
        } else if (selection && selection.entity_id) {
            batteryHint.textContent = 'Optimization will start from the latest value of this sensor.';
            batteryHint.classList.remove('error');
            setBatteryMessage('');
        } else {
            batteryHint.textContent = 'Choose a sensor reporting battery charge in percent (0-100) or fraction (0-1).';
            batteryHint.classList.remove('error');
            setBatteryMessage('');
        }
    }
    if (batteryWearInput) {
        const wearValue = Number(batterySettings.wear_cost_eur_per_kwh);
        batteryWearInput.value = Number.isFinite(wearValue) && wearValue >= 0 ? wearValue.toString() : '';
        if (batteryBusy) {
            batteryWearInput.setAttribute('disabled', 'disabled');
        } else {
            batteryWearInput.removeAttribute('disabled');
        }
    }
    if (batteryWearHint && !batteryWearHint.classList.contains('error')) {
        batteryWearHint.textContent = 'Applied as a throughput penalty in the optimization to limit excessive cycling.';
    }
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
    if (!pricingMessage) return;
    if (pricingMessageClearTimer) {
        clearTimeout(pricingMessageClearTimer);
        pricingMessageClearTimer = null;
    }
    if (!text) {
        pricingMessage.style.display = 'none';
        pricingMessage.textContent = '';
        pricingMessage.classList.remove('error');
        return;
    }
    pricingMessage.style.display = 'block';
    pricingMessage.textContent = text;
    pricingMessage.classList.remove('error');
    if (kind === 'error') {
        pricingMessage.classList.add('error');
    }
    if (text && kind !== 'error' && autoHideMs > 0) {
        pricingMessageClearTimer = setTimeout(() => {
            pricingMessageClearTimer = null;
            setPricingMessage('');
        }, autoHideMs);
    }
}

function setNumericInputValue(input, value) {
    if (!input) return;
    if (Number.isFinite(value)) {
        input.value = value.toString();
    } else if (value === null || value === undefined) {
        input.value = '';
    } else {
        const parsed = Number(value);
        input.value = Number.isFinite(parsed) ? parsed.toString() : '';
    }
}

function setTimeInputValue(input, value, fallback) {
    if (!input) return;
    input.value = sanitizeTimeValue(value, fallback);
}

function updateTariffVisibility(scope) {
    const cfg = pricingSettings && pricingSettings[scope] ? pricingSettings[scope] : defaultTariffConfig(scope);
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
    const cfg = pricingSettings && pricingSettings[scope] ? pricingSettings[scope] : defaultTariffConfig(scope);
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
    const importDefaults = defaultTariffConfig('import');
    const exportDefaults = defaultTariffConfig('export');
    const importConfig = normalizeTariffConfig(pricingSettings.import, 'import');
    const exportConfig = normalizeTariffConfig(pricingSettings.export, 'export');
    pricingSettings = {
        import: importConfig,
        export: exportConfig,
    };

    if (importTariffModeSelect) {
        importTariffModeSelect.value = importConfig.mode;
    }
    if (exportTariffModeSelect) {
        exportTariffModeSelect.value = exportConfig.mode;
    }

    setNumericInputValue(importConstantInput, importConfig.constant_eur_per_kwh);
    setNumericInputValue(importSpotOffsetInput, importConfig.spot_offset_eur_per_kwh);
    setNumericInputValue(importPeakRateInput, importConfig.dual_peak_eur_per_kwh);
    setNumericInputValue(importOffpeakRateInput, importConfig.dual_offpeak_eur_per_kwh);
    setTimeInputValue(importPeakStartInput, importConfig.dual_peak_start_local, importDefaults.dual_peak_start_local);
    setTimeInputValue(importPeakEndInput, importConfig.dual_peak_end_local, importDefaults.dual_peak_end_local);

    setNumericInputValue(exportConstantInput, exportConfig.constant_eur_per_kwh);
    setNumericInputValue(exportSpotOffsetInput, exportConfig.spot_offset_eur_per_kwh);
    setNumericInputValue(exportPeakRateInput, exportConfig.dual_peak_eur_per_kwh);
    setNumericInputValue(exportOffpeakRateInput, exportConfig.dual_offpeak_eur_per_kwh);
    setTimeInputValue(exportPeakStartInput, exportConfig.dual_peak_start_local, exportDefaults.dual_peak_start_local);
    setTimeInputValue(exportPeakEndInput, exportConfig.dual_peak_end_local, exportDefaults.dual_peak_end_local);

    updateTariffVisibility('import');
    updateTariffHint('import');
    updateTariffVisibility('export');
    updateTariffHint('export');
}

function handleTariffModeChange(scope, mode) {
    ensurePricingSettings();
    const normalized = TARIFF_MODES.includes(mode) ? mode : defaultTariffConfig(scope).mode;
    pricingSettings[scope].mode = normalized;
    renderPricingForm();
    schedulePricingSave();
}

function resetPricingToDefaults() {
    pricingSettings = defaultPricingSettings();
    renderPricingForm();
    setPricingMessage('Restored tariff defaults. Saving…');
    schedulePricingSave();
}

function handleSavePricingClick() {
    void savePricingSettings(true);
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
    if (savePricingBtn) {
        savePricingBtn.setAttribute('disabled', 'disabled');
    }
    if (resetPricingBtn) {
        resetPricingBtn.setAttribute('disabled', 'disabled');
    }
    if (manual) {
        setPricingMessage('Saving pricing…');
    } else {
        setPricingMessage('');
    }
    try {
        const payload = normalizePricingSettings(pricingSettings);
        pricingSettings = payload;
        const response = await fetchJson('/api/settings/pricing', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
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
        if (savePricingBtn) {
            savePricingBtn.removeAttribute('disabled');
        }
        if (resetPricingBtn) {
            resetPricingBtn.removeAttribute('disabled');
        }
        if (pricingPendingResave) {
            pricingPendingResave = false;
            schedulePricingSave();
        }
    }
}

function bindTariffNumericInput(input, scope, key) {
    if (!input) return;
    input.addEventListener('change', (event) => {
        ensurePricingSettings();
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
        schedulePricingSave();
    });
}

function bindTariffTimeInput(input, scope, key) {
    if (!input) return;
    input.addEventListener('change', (event) => {
        ensurePricingSettings();
        const fallback = defaultTariffConfig(scope)[key];
        const sanitized = sanitizeTimeValue(event.target.value, fallback);
        pricingSettings[scope][key] = sanitized;
        event.target.value = sanitized;
        schedulePricingSave();
    });
}

async function loadSettingsData() {
    try {
        const payload = await fetchJson('/api/settings');
        applySettingsPayload(payload);
        const recorderStatuses = [
            inverterSettings && inverterSettings.house_consumption && inverterSettings.house_consumption.recorder_status,
            inverterSettings && inverterSettings.pv_power && inverterSettings.pv_power.recorder_status,
        ];
        if (recorderStatuses.some((value) => value !== 'missing') && powerEntityCatalog.length) {
            setInverterMessage('');
        }
        if (batteryEntityCatalog.length) {
            setBatteryMessage('');
        }
    } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        setInverterMessage(message, 'error');
        setBatteryMessage(message, 'error');
    }
}

function setEntityTriggersDisabled(disabled) {
    [houseTrigger, pvTrigger].forEach((btn) => {
        if (!btn) return;
        if (disabled) {
            btn.setAttribute('disabled', 'disabled');
        } else {
            btn.removeAttribute('disabled');
        }
    });
    if (batteryTrigger) {
        if (disabled) {
            batteryTrigger.setAttribute('disabled', 'disabled');
        } else if (!batteryBusy && batteryEntityCatalog.length) {
            batteryTrigger.removeAttribute('disabled');
        }
    }
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
}

async function updateInverterSelection(key, entityId) {
    if (!inverterSettings || inverterBusy) return;
    const current = inverterSettings[key] ? inverterSettings[key].entity_id : null;
    if (!entityId || entityId === current) return;
    inverterBusy = true;
    setEntityTriggersDisabled(true);
    setInverterMessage('Saving selection…');
    try {
        const payload = await fetchJson('/api/settings/inverter', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ [key]: { entity_id: entityId } }),
        });
        applySettingsPayload(payload);
        setInverterMessage('Saved. Re-run training to refresh the models with the new inputs.', 'success');
    } catch (err) {
        setInverterMessage(err instanceof Error ? err.message : String(err), 'error');
    } finally {
        inverterBusy = false;
        setEntityTriggersDisabled(false);
        if (inverterSettings) {
            renderInverterForm();
        }
    }
}

async function updateExportLimitSetting(enabled) {
    if (!inverterSettings || inverterBusy) return;
    inverterBusy = true;
    setEntityTriggersDisabled(true);
    setInverterMessage('Saving selection…');
    try {
        const payload = await fetchJson('/api/settings/inverter', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ export_power_limited: Boolean(enabled) }),
        });
        applySettingsPayload(payload);
        setInverterMessage('Saved export limit preference. Re-run training to incorporate the change.', 'success');
    } catch (err) {
        setInverterMessage(err instanceof Error ? err.message : String(err), 'error');
        if (exportLimitToggle) {
            exportLimitToggle.checked = Boolean(inverterSettings.export_power_limited);
        }
    } finally {
        inverterBusy = false;
        setEntityTriggersDisabled(false);
        if (inverterSettings) {
            renderInverterForm();
        }
    }
}

async function updateBatterySelection(entityId) {
    if (!batterySettings || batteryBusy) return;
    const current = batterySettings.soc_sensor && batterySettings.soc_sensor.entity_id ? batterySettings.soc_sensor.entity_id : null;
    if (!entityId || entityId === current) return;
    batteryBusy = true;
    renderBatteryForm();
    setBatteryMessage('Saving selection…');
    try {
        const payload = await fetchJson('/api/settings/battery', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ soc_sensor: { entity_id: entityId } }),
        });
        applySettingsPayload(payload);
        setBatteryMessage('Saved. Future optimizations will start from this sensor reading.', 'success');
    } catch (err) {
        setBatteryMessage(err instanceof Error ? err.message : String(err), 'error');
    } finally {
        batteryBusy = false;
        renderBatteryForm();
    }
}

async function submitBatteryWearCost(costValue) {
    if (!batterySettings || batteryBusy) return;
    if (!Number.isFinite(costValue) || costValue < 0) {
        if (batteryWearHint) {
            batteryWearHint.classList.add('error');
            batteryWearHint.textContent = 'Enter a non-negative number to penalize battery throughput.';
        }
        setBatteryMessage('Battery wear cost must be a non-negative number.', 'error');
        if (batteryWearInput) {
            const current = Number(batterySettings.wear_cost_eur_per_kwh);
            batteryWearInput.value = Number.isFinite(current) && current >= 0 ? current.toString() : '';
        }
        return;
    }
    const current = Number(batterySettings.wear_cost_eur_per_kwh);
    if (Number.isFinite(current) && Math.abs(current - costValue) < 1e-6) {
        return;
    }
    batteryBusy = true;
    renderBatteryForm();
    let finalMessage = '';
    let finalKind = 'info';
    try {
        const payload = await fetchJson('/api/settings/battery', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ wear_cost_eur_per_kwh: costValue }),
        });
        applySettingsPayload(payload);
        if (batteryWearHint) {
            batteryWearHint.classList.remove('error');
            batteryWearHint.textContent = 'Applied as a throughput penalty in the optimization to limit excessive cycling.';
        }
    } catch (err) {
        finalMessage = err instanceof Error ? err.message : String(err);
        finalKind = 'error';
        if (batteryWearHint) {
            batteryWearHint.classList.add('error');
            batteryWearHint.textContent = 'Failed to save wear cost. ' + String(finalMessage);
        }
    } finally {
        batteryBusy = false;
        renderBatteryForm();
        if (finalMessage) {
            setBatteryMessage(finalMessage, finalKind);
        } else {
            setBatteryMessage('');
        }
    }
}

function onBatteryWearInputChange(event) {
    if (!batterySettings || !event || !event.target) return;
    const raw = event.target.value;
    if (typeof raw !== 'string' || !raw.trim()) {
        renderBatteryForm();
        return;
    }
    const parsed = Number.parseFloat(raw);
    if (!Number.isFinite(parsed) || parsed < 0) {
        if (batteryWearHint) {
            batteryWearHint.classList.add('error');
            batteryWearHint.textContent = 'Enter a non-negative number to penalize battery throughput.';
        }
        setBatteryMessage('Battery wear cost must be a non-negative number.', 'error');
        if (batteryWearInput) {
            const current = Number(batterySettings.wear_cost_eur_per_kwh);
            batteryWearInput.value = Number.isFinite(current) && current >= 0 ? current.toString() : '';
        }
        return;
    }
    submitBatteryWearCost(parsed);
}

function openEntityModal(target) {
    if (!entityModal) return;
    entityModalTarget = target;
    if (target === 'battery_soc') {
        if (!batteryEntityCatalog.length) {
            setBatteryMessage('No battery sensors detected. Expose a sensor with % unit to proceed.', 'error');
        } else {
            setBatteryMessage('');
        }
        lastEntityTrigger = batteryTrigger;
    } else {
        if (!powerEntityCatalog.length) {
            setInverterMessage('No power sensors detected. Expose Home Assistant sensors with power readings to proceed.', 'error');
            return;
        }
        lastEntityTrigger = target === 'pv_power' ? pvTrigger : houseTrigger;
    }
    const modalTitle = document.getElementById('entityModalTitle');
    if (modalTitle) {
        if (target === 'pv_power') {
            modalTitle.textContent = 'Select PV sensor';
        } else if (target === 'battery_soc') {
            modalTitle.textContent = 'Select battery SoC sensor';
        } else {
            modalTitle.textContent = 'Select house sensor';
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
    const catalog = entityModalTarget === 'battery_soc' ? batteryEntityCatalog : powerEntityCatalog;
    const items = catalog.filter((item) => matchesEntity(item, term));
    entityListContainer.textContent = '';
    if (!items.length) {
        const empty = document.createElement('div');
        empty.className = 'entity-empty';
        if (term) {
            empty.textContent = 'No matching sensors found.';
        } else if (entityModalTarget === 'battery_soc') {
            empty.textContent = 'No battery sensors available.';
        } else {
            empty.textContent = 'No power sensors available.';
        }
        entityListContainer.appendChild(empty);
        return;
    }
    let current = null;
    if (entityModalTarget === 'battery_soc') {
        current = batterySettings && batterySettings.soc_sensor ? batterySettings.soc_sensor.entity_id : null;
    } else if (entityModalTarget && inverterSettings && inverterSettings[entityModalTarget]) {
        current = inverterSettings[entityModalTarget].entity_id;
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
            if (targetKey === 'battery_soc') {
                updateBatterySelection(item.entity_id);
            } else if (targetKey) {
                updateInverterSelection(targetKey, item.entity_id);
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
    const dot = document.querySelector('#status .dot');
    const label = document.getElementById('statusLabel');
    const errorEl = document.getElementById('error');
    const trainingStateEl = document.getElementById('trainingState');
    const trainingErrorEl = document.getElementById('trainingError');
    const trainBtn = document.getElementById('trainBtn');
    const haError = payload.home_assistant_error || null;
    const cycleBtn = document.getElementById('cycleBtn');
    const cycleMessageEl = document.getElementById('cycleMessage');
    const summaryWindowEl = document.getElementById('summaryWindow');

    if (!haError && payload.snapshot_available) {
        dot.classList.add('ok');
        label.textContent = 'Service online';
    } else if (!haError) {
        dot.classList.remove('ok');
        label.textContent = 'Waiting for forecast…';
        if (summaryWindowEl) {
            summaryWindowEl.textContent = '';
        }
        renderIntervention(null, 'Current intervention: Waiting for forecast…');
    } else {
        dot.classList.remove('ok');
        label.textContent = 'Home Assistant unavailable';
        if (summaryWindowEl) {
            summaryWindowEl.textContent = '';
        }
        renderIntervention(null, 'Current intervention unavailable until Home Assistant connects');
    }

    const errorMessages = [];
    if (haError) {
        errorMessages.push(haError);
    }
    errorEl.textContent = errorMessages.join(' ');

    const training = payload.training || {};
    if (training.running) {
        trainingStateEl.textContent = training.started_at
            ? `Training in progress (started ${formatDateTime(training.started_at)})`
            : 'Training in progress…';
        trainingStateEl.classList.add('running');
        trainBtn.textContent = 'Training…';
        trainBtn.disabled = true;
    } else if (haError) {
        trainingStateEl.textContent = 'Training unavailable until Home Assistant connects';
        trainingStateEl.classList.remove('running');
        trainBtn.textContent = 'Trigger Training';
        trainBtn.disabled = true;
    } else {
        trainingStateEl.textContent = training.finished_at
            ? `Last trained ${formatDateTime(training.finished_at)}`
            : 'Training idle';
        trainingStateEl.classList.remove('running');
        trainBtn.textContent = 'Trigger Training';
        trainBtn.disabled = false;
    }

    if (training.error) {
        trainingErrorEl.style.display = 'block';
        trainingErrorEl.textContent = `Last training error: ${training.error}`;
    } else {
        trainingErrorEl.style.display = 'none';
        trainingErrorEl.textContent = '';
    }

    if (cycleMessageEl) {
        if (payload.cycle_running) {
            cycleMessageEl.style.display = 'block';
            cycleMessageEl.classList.remove('error');
            cycleMessageEl.textContent = 'Optimization cycle running…';
            cycleMessageEl.dataset.auto = 'running';
        } else if (cycleMessageEl.dataset.auto === 'running') {
            cycleMessageEl.style.display = 'none';
            cycleMessageEl.textContent = '';
            delete cycleMessageEl.dataset.auto;
        }
    }

    if (cycleBtn) {
        if (haError) {
            cycleBtn.disabled = true;
            cycleBtn.textContent = 'Recompute Plan';
        } else if (payload.cycle_running) {
            cycleBtn.disabled = true;
            cycleBtn.textContent = 'Recomputing…';
        } else {
            cycleBtn.disabled = false;
            cycleBtn.textContent = 'Recompute Plan';
        }
    }

    updateSummary(payload.summary);

    const trainingRunning = Boolean(training.running);
    const cycleRunningState = Boolean(payload.cycle_running);
    const trainingCompleted = lastTrainingRunning && !trainingRunning;
    const cycleCompleted = lastCycleRunning && !cycleRunningState;

    lastTrainingRunning = trainingRunning;
    lastCycleRunning = cycleRunningState;

    if (autoRefreshTimer) {
        clearTimeout(autoRefreshTimer);
        autoRefreshTimer = null;
    }
    if (!showingSettings && (trainingRunning || cycleRunningState)) {
        autoRefreshTimer = window.setTimeout(() => {
            autoRefreshTimer = null;
            refreshStatus();
        }, 15000);
    }

    const shouldRefreshForecast = !showingSettings && (trainingCompleted || cycleCompleted);
    if (shouldRefreshForecast) {
        if (pendingForecastRefresh) {
            clearTimeout(pendingForecastRefresh);
        }
        pendingForecastRefresh = window.setTimeout(() => {
            pendingForecastRefresh = null;
            refreshForecast();
        }, 1200);
    }
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
                legend: { display: true },
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
                x: { ticks: { maxRotation: 0, autoSkip: true } },
                y: {
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
    planChart.data.datasets[0].segment = {
        borderDash: (ctx) => (segmentImputed(ctx) ? [6, 4] : []),
    };
    planChart.data.datasets[1].segment = {
        borderDash: (ctx) => (segmentImputed(ctx) ? [6, 4] : []),
    };
    const imputedRadiusValues = labels.map((_, idx) => (priceImputedFlags[idx] ? 2 : 0));
    planChart.data.datasets[0].pointRadius = imputedRadiusValues;
    planChart.data.datasets[1].pointRadius = imputedRadiusValues;
    planChart.data.datasets[0].pointHoverRadius = imputedRadiusValues;
    planChart.data.datasets[1].pointHoverRadius = imputedRadiusValues;
    planChart.data.datasets[0].pointBackgroundColor = labels.map(() => '#f97316');
    planChart.data.datasets[1].pointBackgroundColor = labels.map(() => '#2563eb');
    planChart.data.datasets[2].pointRadius = 0;
    planChart.data.datasets[2].pointBackgroundColor = '#a855f7';
    planChart.update();

    gridChart.data.labels = labels;
    gridChart.data.datasets[0].data = sanitizedImport;
    gridChart.data.datasets[1].data = sanitizedExport;
    gridChart.update();

    updateSummary(payload.summary);
    forecastMessage.textContent = '';
    forecastMessage.classList.remove('error');

    const summaryWindowEl = document.getElementById('summaryWindow');
    if (summaryWindowEl) {
        const windowInfo = payload.summary_window || {};
        if (windowInfo.start && windowInfo.end) {
            const startText = formatDateTime(windowInfo.start);
            const endText = formatDateTime(windowInfo.end);
            summaryWindowEl.textContent = `Forecast horizon: ${startText} → ${endText}`;
        } else {
            summaryWindowEl.textContent = '';
        }
    }

    renderIntervention(payload.intervention || null);
}

async function refreshStatus() {
    try {
        const payload = await fetchJson('/api/status');
        applyStatus(payload);
    } catch (err) {
        const dot = document.querySelector('#status .dot');
        const label = document.getElementById('statusLabel');
        const errorEl = document.getElementById('error');
        dot.classList.remove('ok');
        label.textContent = 'Failed to reach API';
        errorEl.textContent = err instanceof Error ? err.message : String(err);
    }
}

async function refreshForecast() {
    const forecastMessage = document.getElementById('forecastMessage');
    forecastMessage.textContent = 'Loading latest forecast…';
    forecastMessage.classList.remove('error');
    try {
        const payload = await fetchJson('/api/forecast');
        applyForecast(payload);
    } catch (err) {
        forecastMessage.textContent = err instanceof Error ? err.message : String(err);
        forecastMessage.classList.add('error');
        renderIntervention(null, `Current intervention unavailable: ${err instanceof Error ? err.message : String(err)}`);
    }
}

async function triggerTraining() {
    const button = document.getElementById('trainBtn');
    button.disabled = true;
    button.textContent = 'Starting…';
    try {
        await fetchJson('/api/training', { method: 'POST' });
    } catch (err) {
        const errorEl = document.getElementById('trainingError');
        errorEl.style.display = 'block';
        errorEl.textContent = err instanceof Error ? `Failed to start training: ${err.message}` : String(err);
    }
    await refreshStatus();
    button.textContent = button.disabled ? 'Training…' : 'Trigger Training';
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
        await fetchJson('/api/cycle', { method: 'POST' });
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
            button.disabled = false;
            button.textContent = 'Recompute Plan';
        }
    }
    await refreshStatus();
    await refreshForecast();
}

window.addEventListener('DOMContentLoaded', () => {
    document.getElementById('refreshBtn').addEventListener('click', () => {
        refreshStatus();
        refreshForecast();
    });
    document.getElementById('trainBtn').addEventListener('click', triggerTraining);
    document.getElementById('cycleBtn').addEventListener('click', triggerCycle);
    document.getElementById('settingsBtn').addEventListener('click', toggleSettings);
    document.querySelectorAll('.settings-nav button').forEach((btn) => {
        btn.addEventListener('click', () => {
            activateSettingsTab(btn.dataset.settingsTab);
        });
    });
    if (houseTrigger) {
        houseTrigger.addEventListener('click', () => openEntityModal('house_consumption'));
    }
    if (pvTrigger) {
        pvTrigger.addEventListener('click', () => openEntityModal('pv_power'));
    }
    if (batteryTrigger) {
        batteryTrigger.addEventListener('click', () => openEntityModal('battery_soc'));
    }
    if (batteryWearInput) {
        batteryWearInput.addEventListener('change', onBatteryWearInputChange);
        batteryWearInput.addEventListener('blur', onBatteryWearInputChange);
    }
    if (exportLimitToggle) {
        exportLimitToggle.addEventListener('change', (event) => {
            updateExportLimitSetting(event.target.checked);
        });
    }
    if (entityModalBackdrop) {
        entityModalBackdrop.addEventListener('click', closeEntityModal);
    }
    if (entityModalClose) {
        entityModalClose.addEventListener('click', closeEntityModal);
    }
    if (entityModalCancel) {
        entityModalCancel.addEventListener('click', closeEntityModal);
    }
    if (entitySearchInput) {
        entitySearchInput.addEventListener('input', (event) => {
            renderEntityList(event.target.value);
        });
    }
    if (importTariffModeSelect) {
        importTariffModeSelect.addEventListener('change', (event) => {
            handleTariffModeChange('import', event.target.value);
        });
    }
    if (exportTariffModeSelect) {
        exportTariffModeSelect.addEventListener('change', (event) => {
            handleTariffModeChange('export', event.target.value);
        });
    }
    bindTariffNumericInput(importConstantInput, 'import', 'constant_eur_per_kwh');
    bindTariffNumericInput(importSpotOffsetInput, 'import', 'spot_offset_eur_per_kwh');
    bindTariffNumericInput(importPeakRateInput, 'import', 'dual_peak_eur_per_kwh');
    bindTariffNumericInput(importOffpeakRateInput, 'import', 'dual_offpeak_eur_per_kwh');
    bindTariffTimeInput(importPeakStartInput, 'import', 'dual_peak_start_local');
    bindTariffTimeInput(importPeakEndInput, 'import', 'dual_peak_end_local');
    bindTariffNumericInput(exportConstantInput, 'export', 'constant_eur_per_kwh');
    bindTariffNumericInput(exportSpotOffsetInput, 'export', 'spot_offset_eur_per_kwh');
    bindTariffNumericInput(exportPeakRateInput, 'export', 'dual_peak_eur_per_kwh');
    bindTariffNumericInput(exportOffpeakRateInput, 'export', 'dual_offpeak_eur_per_kwh');
    bindTariffTimeInput(exportPeakStartInput, 'export', 'dual_peak_start_local');
    bindTariffTimeInput(exportPeakEndInput, 'export', 'dual_peak_end_local');
    if (savePricingBtn) {
        savePricingBtn.addEventListener('click', handleSavePricingClick);
    }
    if (resetPricingBtn) {
        resetPricingBtn.addEventListener('click', resetPricingToDefaults);
    }
    window.addEventListener('keydown', (event) => {
        if (event.key === 'Escape' && entityModal && entityModal.classList.contains('open')) {
            closeEntityModal();
        }
    });

    renderPricingForm();
    refreshStatus();
    refreshForecast();
    loadSettingsData();
    updateViewMode();
    renderIntervention(null, 'Current intervention: Waiting for forecast…');
    activateSettingsTab('inverter');

    setInterval(() => {
        if (!showingSettings) {
            refreshStatus();
            refreshForecast();
        }
    }, 5 * 60 * 1000);
});
