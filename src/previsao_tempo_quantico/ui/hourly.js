const UF_TO_MACROREGION = {
  AC: "Norte",
  AL: "Nordeste",
  AP: "Norte",
  AM: "Norte",
  BA: "Nordeste",
  CE: "Nordeste",
  DF: "Centro-Oeste",
  ES: "Sudeste",
  GO: "Centro-Oeste",
  MA: "Nordeste",
  MG: "Sudeste",
  MS: "Centro-Oeste",
  MT: "Centro-Oeste",
  PA: "Norte",
  PB: "Nordeste",
  PE: "Nordeste",
  PI: "Nordeste",
  PR: "Sul",
  RJ: "Sudeste",
  RN: "Nordeste",
  RO: "Norte",
  RR: "Norte",
  RS: "Sul",
  SC: "Sul",
  SE: "Nordeste",
  SP: "Sudeste",
  TO: "Norte",
};

const MACROREGIONS = ["Norte", "Nordeste", "Centro-Oeste", "Sudeste", "Sul"];

const map = L.map("map", { zoomControl: true, preferCanvas: true }).setView([-14.2, -52.2], 4);

L.tileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", {
  attribution: "Tiles (c) Esri",
  maxZoom: 9,
}).addTo(map);

L.tileLayer("https://{s}.basemaps.cartocdn.com/light_only_labels/{z}/{x}/{y}{r}.png", {
  attribution: "(c) OpenStreetMap (c) CARTO",
  maxZoom: 9,
  opacity: 0.95,
}).addTo(map);

const state = {
  hour: 0,
  statesGeo: null,
  statePayload: null,
  cityPayload: null,
  stateLayer: null,
  cityLayer: L.layerGroup().addTo(map),
  lastUfZoom: "",
};

const controls = {
  uf: document.getElementById("uf"),
  cityLimit: document.getElementById("cityLimit"),
  cityQuery: document.getElementById("cityQuery"),
  metric: document.getElementById("metric"),
  hourRange: document.getElementById("hourRange"),
  hourValue: document.getElementById("hourValue"),
  refreshBtn: document.getElementById("refreshBtn"),
  statusText: document.getElementById("statusText"),
  sourceTime: document.getElementById("sourceTime"),
  cityCount: document.getElementById("cityCount"),
  regionCards: document.getElementById("regionCards"),
  cityRows: document.getElementById("cityRows"),
};

function fmt(value, decimals = 1) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
  return Number(value).toFixed(decimals);
}

function fmtDate(value) {
  if (!value) return "-";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString("pt-BR");
}

function normalizeText(value) {
  return (value || "")
    .toString()
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .toLowerCase()
    .trim();
}

function parseUfFromFeature(feature) {
  const props = feature?.properties || {};
  return (props.sigla || props.SIGLA || props.uf || props.UF || props.sigla_uf || "").toString().toUpperCase();
}

function parseStateName(feature) {
  const props = feature?.properties || {};
  return props.name || props.NOME || props.nome || props.NAME || "Estado";
}

function getHourlyAt(item, hour) {
  if (!item || !Array.isArray(item.hourly)) return null;
  return (
    item.hourly.find((row) => Number(row.hour_of_day) === Number(hour)) ||
    item.hourly.find((row) => Number(row.hour_offset) === Number(hour)) ||
    null
  );
}

function metricValue(row) {
  if (!row) return Number.NaN;
  if (controls.metric.value === "rain") return Number(row.rain_probability_pct);
  return Number(row.temperature_c);
}

function metricUnit() {
  return controls.metric.value === "rain" ? "%" : "C";
}

function calcColor(value, min, max) {
  if (!Number.isFinite(value)) return "#6f7f90";
  const span = max - min || 1;
  const ratio = Math.max(0, Math.min(1, (value - min) / span));
  if (ratio < 0.33) return "#13b489";
  if (ratio < 0.66) return "#f7bc3e";
  return "#f85d49";
}

function extent(values) {
  const valid = values.filter((v) => Number.isFinite(v));
  if (!valid.length) return { min: 0, max: 1 };
  return { min: Math.min(...valid), max: Math.max(...valid) };
}

function groupByMacroregion(cityItems) {
  const groups = {};
  for (const macro of MACROREGIONS) groups[macro] = [];
  for (const item of cityItems) {
    const macro = UF_TO_MACROREGION[(item.uf || "").toUpperCase()] || "Desconhecida";
    if (!groups[macro]) groups[macro] = [];
    groups[macro].push(item);
  }
  return groups;
}

function avg(values) {
  if (!values.length) return Number.NaN;
  let total = 0;
  for (const value of values) total += value;
  return total / values.length;
}

function buildStateValueMap() {
  const output = {};
  const items = state.statePayload?.items || [];
  for (const item of items) {
    const uf = (item.uf || "").toUpperCase();
    const row = getHourlyAt(item, state.hour);
    output[uf] = metricValue(row);
  }
  return output;
}

function cityMatchesFilter(item) {
  const term = normalizeText(controls.cityQuery.value);
  if (!term) return true;
  return normalizeText(item.region_name).includes(term);
}

function getFilteredCityItems() {
  const items = state.cityPayload?.items || [];
  return items.filter(cityMatchesFilter);
}

function pickFocusedCity(items) {
  const term = normalizeText(controls.cityQuery.value);
  if (!term || !items.length) return null;
  const exact = items.find((item) => normalizeText(item.region_name) === term);
  return exact || items[0];
}

function tryZoomToUf(uf) {
  if (!uf || !state.stateLayer) return;
  const targetUf = uf.toUpperCase();
  if (state.lastUfZoom === targetUf) return;
  state.lastUfZoom = targetUf;

  state.stateLayer.eachLayer((layer) => {
    const layerUf = parseUfFromFeature(layer.feature);
    if (layerUf !== targetUf) return;
    const bounds = layer.getBounds?.();
    if (bounds && bounds.isValid()) {
      map.fitBounds(bounds.pad(0.28));
    }
  });
}

function renderStatesLayer() {
  if (!state.statesGeo) return;
  const valuesMap = buildStateValueMap();
  const metricValues = Object.values(valuesMap);
  const { min, max } = extent(metricValues);

  if (state.stateLayer) {
    state.stateLayer.remove();
  }

  state.stateLayer = L.geoJSON(state.statesGeo, {
    style: (feature) => {
      const uf = parseUfFromFeature(feature);
      const value = valuesMap[uf];
      return {
        color: "#dce8f4",
        weight: 1.2,
        fillOpacity: 0.66,
        fillColor: calcColor(value, min, max),
      };
    },
    onEachFeature: (feature, layer) => {
      const uf = parseUfFromFeature(feature);
      const value = valuesMap[uf];
      const stateName = parseStateName(feature);
      layer.bindTooltip(`${stateName} (${uf})<br/>${fmt(value, 1)} ${metricUnit()}`, { sticky: true });
    },
  }).addTo(map);

  const uf = controls.uf.value.trim().toUpperCase();
  if (uf.length === 2) {
    tryZoomToUf(uf);
  } else {
    state.lastUfZoom = "";
  }
}

function renderCityMarkersAndTable() {
  state.cityLayer.clearLayers();
  controls.cityRows.innerHTML = "";

  const items = getFilteredCityItems();
  const focused = pickFocusedCity(items);
  const rows = [];
  for (const item of items) {
    const hourRow = getHourlyAt(item, state.hour);
    if (!hourRow) continue;
    rows.push({
      item,
      hourRow,
      value: metricValue(hourRow),
    });
  }

  const metricValues = rows.map((row) => row.value);
  const { min, max } = extent(metricValues);

  for (const row of rows) {
    const lat = Number(row.item.lat);
    const lon = Number(row.item.lon);
    if (!Number.isFinite(lat) || !Number.isFinite(lon)) continue;
    const color = calcColor(row.value, min, max);
    L.circleMarker([lat, lon], {
      radius: 4.2,
      color: "#ffffff",
      weight: 0.8,
      fillColor: color,
      fillOpacity: 0.92,
    })
      .bindPopup(
        `<strong>${row.item.region_name}</strong> (${row.item.uf})<br/>` +
          `${String(state.hour).padStart(2, "0")}:00: ${fmt(row.hourRow.temperature_c, 1)} C<br/>` +
          `Chuva: ${fmt(row.hourRow.rain_probability_pct, 1)}% | ${fmt(row.hourRow.rain_mm_h, 2)} mm/h<br/>` +
          `Condicao: ${row.hourRow.condition || "-"}`
      )
      .addTo(state.cityLayer);
  }

  if (focused) {
    const lat = Number(focused.lat);
    const lon = Number(focused.lon);
    if (Number.isFinite(lat) && Number.isFinite(lon)) {
      map.setView([lat, lon], 8);
    }
  }

  rows.sort((a, b) => {
    const va = Number.isFinite(a.value) ? a.value : -Infinity;
    const vb = Number.isFinite(b.value) ? b.value : -Infinity;
    return vb - va;
  });

  for (const row of rows.slice(0, 300)) {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${row.item.region_name || "-"}</td>
      <td>${row.item.uf || "-"}</td>
      <td>${fmt(row.hourRow.temperature_c, 1)}</td>
      <td>${fmt(row.hourRow.rain_probability_pct, 1)}</td>
      <td>${row.hourRow.condition || "-"}</td>
    `;
    controls.cityRows.appendChild(tr);
  }

  const total = Number(state.cityPayload?.count || 0);
  controls.cityCount.textContent = `${rows.length}/${total}`;
}

function renderRegionCards() {
  controls.regionCards.innerHTML = "";
  const items = getFilteredCityItems();
  const grouped = groupByMacroregion(items);

  for (const macro of MACROREGIONS) {
    const group = grouped[macro] || [];
    const temps = [];
    const probs = [];
    for (const item of group) {
      const hourRow = getHourlyAt(item, state.hour);
      if (!hourRow) continue;
      const temp = Number(hourRow.temperature_c);
      const prob = Number(hourRow.rain_probability_pct);
      if (Number.isFinite(temp)) temps.push(temp);
      if (Number.isFinite(prob)) probs.push(prob);
    }

    const card = document.createElement("article");
    card.className = "region-card";
    card.innerHTML = `
      <h3>${macro}</h3>
      <div class="mini-grid">
        <span>Cidades: ${group.length}</span>
        <span>Hora: ${String(state.hour).padStart(2, "0")}:00</span>
        <span>T media: ${fmt(avg(temps), 1)} C</span>
        <span>P chuva: ${fmt(avg(probs), 1)}%</span>
      </div>
    `;
    controls.regionCards.appendChild(card);
  }
}

function renderAll() {
  if (!state.statePayload || !state.cityPayload) return;
  renderStatesLayer();
  renderCityMarkersAndTable();
  renderRegionCards();
}

async function fetchJson(url) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Falha ${response.status} em ${url}`);
  }
  return response.json();
}

async function fetchStatesGeo() {
  if (state.statesGeo) return state.statesGeo;
  state.statesGeo = await fetchJson("/api/geo/states");
  return state.statesGeo;
}

async function fetchForecast(scope, uf, municipalityLimit) {
  const params = new URLSearchParams({
    scope,
    horizons: "6,12,18,24",
  });
  if (scope === "municipio") {
    params.set("municipality_limit", String(municipalityLimit));
    if (uf) params.set("uf", uf);
  }
  return fetchJson(`/api/forecast/next24h?${params.toString()}`);
}

function syncHourLabel() {
  controls.hourValue.textContent = `${String(state.hour).padStart(2, "0")}:00`;
}

async function refreshDashboard() {
  controls.statusText.textContent = "Atualizando previsao hora a hora...";
  const uf = controls.uf.value.trim().toUpperCase();
  const cityLimit = Number(controls.cityLimit.value) || 1200;

  try {
    const [statesGeo, statePayload, cityPayload] = await Promise.all([
      fetchStatesGeo(),
      fetchForecast("state", "", cityLimit),
      fetchForecast("municipio", uf, cityLimit),
    ]);

    state.statesGeo = statesGeo;
    state.statePayload = statePayload;
    state.cityPayload = cityPayload;
    controls.sourceTime.textContent = fmtDate(cityPayload.source_time || statePayload.source_time);

    renderAll();
    controls.statusText.textContent = `Previsao carregada para ${String(state.hour).padStart(2, "0")}:00.`;
  } catch (error) {
    controls.statusText.textContent = `Erro: ${error.message}`;
  }
}

controls.hourRange.addEventListener("input", () => {
  state.hour = Number(controls.hourRange.value) || 1;
  syncHourLabel();
  renderAll();
});

controls.metric.addEventListener("change", renderAll);
controls.cityQuery.addEventListener("input", renderAll);
controls.refreshBtn.addEventListener("click", refreshDashboard);

controls.uf.addEventListener("keydown", (event) => {
  if (event.key === "Enter") refreshDashboard();
});

controls.cityLimit.addEventListener("keydown", (event) => {
  if (event.key === "Enter") refreshDashboard();
});

controls.cityQuery.addEventListener("keydown", (event) => {
  if (event.key === "Enter") renderAll();
});

const urlParams = new URLSearchParams(window.location.search);
const ufParam = (urlParams.get("uf") || "").trim().toUpperCase();
const metricParam = normalizeText(urlParams.get("metric") || "");
const hourParam = Number(urlParams.get("hour") || "");
const limitParam = Number(urlParams.get("limit") || "");
const cityParam = (urlParams.get("city") || "").trim();

if (ufParam) controls.uf.value = ufParam;
if (metricParam === "rain" || metricParam === "temperature") controls.metric.value = metricParam;
if (Number.isFinite(hourParam) && hourParam >= 0 && hourParam <= 23) {
  state.hour = hourParam;
  controls.hourRange.value = String(hourParam);
}
if (Number.isFinite(limitParam) && limitParam >= 50 && limitParam <= 5570) {
  controls.cityLimit.value = String(Math.floor(limitParam));
}
if (cityParam) controls.cityQuery.value = cityParam;

syncHourLabel();
refreshDashboard();
