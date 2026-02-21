const map = L.map("map", { zoomControl: true }).setView([-14.2, -52.2], 4);

L.tileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", {
  attribution: "Tiles © Esri",
  maxZoom: 8,
}).addTo(map);

L.tileLayer("https://{s}.basemaps.cartocdn.com/light_only_labels/{z}/{x}/{y}{r}.png", {
  attribution: "© OpenStreetMap © CARTO",
  maxZoom: 8,
  opacity: 0.95,
}).addTo(map);

const state = {
  selectedHorizon: 6,
  statePayload: null,
  cityPayload: null,
  statesGeo: null,
  borderLayer: null,
  cityLayer: L.layerGroup().addTo(map),
};

const controls = {
  uf: document.getElementById("uf"),
  cityLimit: document.getElementById("cityLimit"),
  cityQuery: document.getElementById("cityQuery"),
  snapDate: document.getElementById("snapDate"),
  metric: document.getElementById("metric"),
  refreshBtn: document.getElementById("refreshBtn"),
  sourceTime: document.getElementById("sourceTime"),
  cityCount: document.getElementById("cityCount"),
  statusText: document.getElementById("statusText"),
  cityRows: document.getElementById("cityRows"),
  cityFocus: document.getElementById("cityFocus"),
  snapRows: document.getElementById("snapRows"),
  regionCards: document.getElementById("regionCards"),
  horizonButtons: Array.from(document.querySelectorAll(".h-btn")),
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

function isoToInputDate(isoValue) {
  if (!isoValue) return "";
  const date = new Date(isoValue);
  if (Number.isNaN(date.getTime())) return "";
  const shifted = new Date(date.getTime() - date.getTimezoneOffset() * 60000);
  return shifted.toISOString().slice(0, 10);
}

function toStateValueMap(payload, metric, horizon) {
  const output = {};
  const key = String(horizon);
  const byState = payload?.state_horizon_values || {};
  for (const [uf, horizons] of Object.entries(byState)) {
    const item = horizons[key] || {};
    output[uf] = metric === "rain" ? Number(item.rain_probability_pct) : Number(item.temperature_c);
  }
  return output;
}

function calcColor(value, min, max) {
  if (!Number.isFinite(value)) return "#5c6d82";
  const span = max - min || 1;
  const ratio = Math.max(0, Math.min(1, (value - min) / span));
  if (ratio < 0.33) return "#00b383";
  if (ratio < 0.66) return "#ffbf3c";
  return "#ff5b45";
}

function parseUfFromFeature(feature) {
  const props = feature?.properties || {};
  return (props.sigla || props.SIGLA || props.uf || props.UF || props.sigla_uf || "").toString().toUpperCase();
}

function parseNameFromFeature(feature) {
  const props = feature?.properties || {};
  return props.name || props.NOME || props.nome || props.NAME || "Estado";
}

function valueExtent(valuesMap) {
  const values = Object.values(valuesMap).filter((v) => Number.isFinite(v));
  if (!values.length) return { min: 0, max: 1 };
  return { min: Math.min(...values), max: Math.max(...values) };
}

function cityForecastAt(item, horizon) {
  return (item.forecasts || []).find((f) => Number(f.horizon_hours) === Number(horizon));
}

function getFilteredCityItems() {
  const items = state.cityPayload?.items || [];
  const term = normalizeText(controls.cityQuery.value);
  if (!term) return items;
  return items.filter((item) => normalizeText(item.region_name).includes(term));
}

function pickFocusedCity(filteredItems) {
  const term = normalizeText(controls.cityQuery.value);
  if (!term || !filteredItems.length) return null;
  const exact = filteredItems.find((item) => normalizeText(item.region_name) === term);
  return exact || filteredItems[0];
}

function renderStatesChoropleth() {
  if (!state.statesGeo) return;
  const metric = controls.metric.value;
  const valuesMap = toStateValueMap(state.statePayload, metric, state.selectedHorizon);
  const extent = valueExtent(valuesMap);

  if (state.borderLayer) state.borderLayer.remove();

  state.borderLayer = L.geoJSON(state.statesGeo, {
    style: (feature) => {
      const uf = parseUfFromFeature(feature);
      const value = valuesMap[uf];
      return {
        color: "#d5e5f6",
        weight: 1.1,
        fillOpacity: 0.68,
        fillColor: calcColor(value, extent.min, extent.max),
      };
    },
    onEachFeature: (feature, layer) => {
      const uf = parseUfFromFeature(feature);
      const stateName = parseNameFromFeature(feature);
      const value = valuesMap[uf];
      const unit = metric === "rain" ? "%" : "C";
      layer.bindTooltip(`${stateName} (${uf})<br/>${fmt(value, 1)} ${unit}`, { sticky: true });
    },
  }).addTo(map);
}

function renderCityMarkers(items, focusedCity) {
  state.cityLayer.clearLayers();
  if (!items.length) return;

  const metric = controls.metric.value;
  const values = items
    .map((item) => cityForecastAt(item, state.selectedHorizon))
    .filter(Boolean)
    .map((fc) => (metric === "rain" ? Number(fc.rain_probability_pct) : Number(fc.temperature_c)))
    .filter((v) => Number.isFinite(v));
  const extent = values.length ? { min: Math.min(...values), max: Math.max(...values) } : { min: 0, max: 1 };

  for (const item of items) {
    const fc = cityForecastAt(item, state.selectedHorizon);
    if (!fc) continue;
    const lat = Number(item.lat);
    const lon = Number(item.lon);
    const value = metric === "rain" ? Number(fc.rain_probability_pct) : Number(fc.temperature_c);
    if (!Number.isFinite(lat) || !Number.isFinite(lon) || !Number.isFinite(value)) continue;

    const isFocused = focusedCity && item.region_id === focusedCity.region_id;
    const color = calcColor(value, extent.min, extent.max);
    L.circleMarker([lat, lon], {
      radius: isFocused ? 7 : 4,
      color: isFocused ? "#ffffff" : color,
      fillColor: color,
      fillOpacity: 0.92,
      weight: isFocused ? 1.8 : 0.8,
    })
      .bindPopup(
        `<strong>${item.region_name}</strong> (${item.uf})<br/>` +
          `+${state.selectedHorizon}h: ${fmt(fc.temperature_c, 1)} C | ${fmt(fc.rain_probability_pct, 1)}%<br/>` +
          `Chuva: ${fmt(fc.rain_mm_h, 2)} mm/h | Condicao: ${fc.condition || "-"}`
      )
      .addTo(state.cityLayer);
  }

  if (focusedCity && normalizeText(controls.cityQuery.value)) {
    const lat = Number(focusedCity.lat);
    const lon = Number(focusedCity.lon);
    if (Number.isFinite(lat) && Number.isFinite(lon)) {
      map.setView([lat, lon], 8);
    }
  }
}

function renderRegionCards() {
  controls.regionCards.innerHTML = "";
  const summaries = state.statePayload?.summaries || [];
  for (const summary of summaries) {
    const card = document.createElement("article");
    card.className = "region-card";

    const rows = summary.horizons || [];
    const h6 = rows.find((r) => Number(r.horizon_hours) === 6) || {};
    const h12 = rows.find((r) => Number(r.horizon_hours) === 12) || {};
    const h18 = rows.find((r) => Number(r.horizon_hours) === 18) || {};
    const h24 = rows.find((r) => Number(r.horizon_hours) === 24) || {};

    card.innerHTML = `
      <h3>${summary.macroregion}</h3>
      <div class="mini-grid">
        <span>T+6 ${fmt(h6.temperature_c_mean, 1)} C</span>
        <span>P+6 ${fmt(h6.rain_probability_pct_mean, 1)}%</span>
        <span>T+12 ${fmt(h12.temperature_c_mean, 1)} C</span>
        <span>P+12 ${fmt(h12.rain_probability_pct_mean, 1)}%</span>
        <span>T+18 ${fmt(h18.temperature_c_mean, 1)} C</span>
        <span>P+18 ${fmt(h18.rain_probability_pct_mean, 1)}%</span>
        <span>T+24 ${fmt(h24.temperature_c_mean, 1)} C</span>
        <span>P+24 ${fmt(h24.rain_probability_pct_mean, 1)}%</span>
      </div>
    `;
    controls.regionCards.appendChild(card);
  }
}

function renderCityFocus(focusedCity) {
  if (!focusedCity) {
    const term = normalizeText(controls.cityQuery.value);
    controls.cityFocus.textContent = term
      ? "Cidade nao encontrada no conjunto atual. Ajuste UF ou limite de cidades."
      : "Digite uma cidade para ver a previsao detalhada.";
    controls.snapRows.innerHTML = "";
    return;
  }

  const h6 = cityForecastAt(focusedCity, 6) || {};
  const h24 = cityForecastAt(focusedCity, 24) || {};
  controls.cityFocus.innerHTML = `
    <h3>${focusedCity.region_name} (${focusedCity.uf})</h3>
    <div class="focus-grid">
      <span>T+6 ${fmt(h6.temperature_c, 1)} C</span>
      <span>P+6 ${fmt(h6.rain_probability_pct, 1)}%</span>
      <span>T+24 ${fmt(h24.temperature_c, 1)} C</span>
      <span>P+24 ${fmt(h24.rain_probability_pct, 1)}%</span>
    </div>
    <p class="focus-cond">Condicao +${state.selectedHorizon}h: ${
      (cityForecastAt(focusedCity, state.selectedHorizon) || {}).condition || "-"
    }</p>
  `;

  const selectedDate = controls.snapDate.value;
  const hourlyRows = (focusedCity.hourly || []).filter((row) => {
    if (!selectedDate) return true;
    return String(row.timestamp || "").slice(0, 10) === selectedDate;
  });

  controls.snapRows.innerHTML = "";
  if (!hourlyRows.length) {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td colspan="5">Sem horas para a data selecionada.</td>`;
    controls.snapRows.appendChild(tr);
    return;
  }

  for (const row of hourlyRows) {
    const d = new Date(row.timestamp);
    const hh = Number.isNaN(d.getTime())
      ? `+${row.hour_offset}h`
      : d.toLocaleTimeString("pt-BR", { hour: "2-digit", minute: "2-digit" });
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${hh}</td>
      <td>${fmt(row.temperature_c, 1)}</td>
      <td>${fmt(row.rain_probability_pct, 1)}</td>
      <td>${fmt(row.rain_mm_h, 2)}</td>
      <td>${row.condition || "-"}</td>
    `;
    controls.snapRows.appendChild(tr);
  }
}

function renderCityTable(items) {
  controls.cityRows.innerHTML = "";
  const sorted = items.slice().sort((a, b) => {
    const fa = cityForecastAt(a, state.selectedHorizon);
    const fb = cityForecastAt(b, state.selectedHorizon);
    const va = controls.metric.value === "rain" ? Number(fa?.rain_probability_pct) : Number(fa?.temperature_c);
    const vb = controls.metric.value === "rain" ? Number(fb?.rain_probability_pct) : Number(fb?.temperature_c);
    return (Number.isFinite(vb) ? vb : -Infinity) - (Number.isFinite(va) ? va : -Infinity);
  });

  for (const item of sorted.slice(0, 300)) {
    const h6 = cityForecastAt(item, 6) || {};
    const h12 = cityForecastAt(item, 12) || {};
    const h18 = cityForecastAt(item, 18) || {};
    const h24 = cityForecastAt(item, 24) || {};

    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${item.region_name || "-"}</td>
      <td>${item.uf || "-"}</td>
      <td>${fmt(h6.temperature_c, 1)}</td>
      <td>${fmt(h6.rain_probability_pct, 1)}</td>
      <td>${fmt(h12.temperature_c, 1)}</td>
      <td>${fmt(h12.rain_probability_pct, 1)}</td>
      <td>${fmt(h18.temperature_c, 1)}</td>
      <td>${fmt(h18.rain_probability_pct, 1)}</td>
      <td>${fmt(h24.temperature_c, 1)}</td>
      <td>${fmt(h24.rain_probability_pct, 1)}</td>
    `;
    controls.cityRows.appendChild(tr);
  }
}

function renderAll() {
  renderStatesChoropleth();
  renderRegionCards();
  const filteredCities = getFilteredCityItems();
  const focusedCity = pickFocusedCity(filteredCities);
  renderCityMarkers(filteredCities, focusedCity);
  renderCityTable(filteredCities);
  renderCityFocus(focusedCity);
  const total = Number(state.cityPayload?.count || 0);
  controls.cityCount.textContent = `${filteredCities.length}/${total}`;
}

async function fetchStatesGeo() {
  const res = await fetch("/api/geo/states");
  if (!res.ok) throw new Error(`Falha ao carregar malha de estados (${res.status})`);
  return res.json();
}

async function fetchForecast(scope, uf, cityLimit) {
  const params = new URLSearchParams({ scope, horizons: "6,12,18,24" });
  if (uf) params.set("uf", uf);
  if (scope === "municipio") params.set("municipality_limit", String(cityLimit));
  const res = await fetch(`/api/forecast/next24h?${params.toString()}`);
  if (!res.ok) throw new Error(`Falha no endpoint next24h (${res.status})`);
  return res.json();
}

async function refreshDashboard() {
  controls.statusText.textContent = "Atualizando previsao...";
  const uf = controls.uf.value.trim().toUpperCase();
  const cityLimit = Number(controls.cityLimit.value) || 1200;

  try {
    const [statesGeo, statePayload, cityPayload] = await Promise.all([
      state.statesGeo ? Promise.resolve(state.statesGeo) : fetchStatesGeo(),
      fetchForecast("state", "", cityLimit),
      fetchForecast("municipio", uf, cityLimit),
    ]);

    state.statesGeo = statesGeo;
    state.statePayload = statePayload;
    state.cityPayload = cityPayload;

    controls.sourceTime.textContent = fmtDate(statePayload.source_time);
    if (!controls.snapDate.value) {
      controls.snapDate.value = isoToInputDate(statePayload.source_time);
    }

    renderAll();
    controls.statusText.textContent = `Previsao carregada para +${state.selectedHorizon}h`;
  } catch (error) {
    controls.statusText.textContent = `Erro: ${error.message}`;
  }
}

for (const btn of controls.horizonButtons) {
  btn.addEventListener("click", () => {
    for (const b of controls.horizonButtons) b.classList.remove("active");
    btn.classList.add("active");
    state.selectedHorizon = Number(btn.dataset.h || "6");
    renderAll();
  });
}

controls.metric.addEventListener("change", renderAll);
controls.cityQuery.addEventListener("input", renderAll);
controls.snapDate.addEventListener("change", renderAll);
controls.refreshBtn.addEventListener("click", refreshDashboard);
controls.uf.addEventListener("keydown", (event) => {
  if (event.key === "Enter") refreshDashboard();
});
controls.cityQuery.addEventListener("keydown", (event) => {
  if (event.key === "Enter") renderAll();
});

const urlParams = new URLSearchParams(window.location.search);
if (urlParams.get("city")) controls.cityQuery.value = urlParams.get("city");
if (urlParams.get("uf")) controls.uf.value = urlParams.get("uf").toUpperCase();
if (urlParams.get("date")) controls.snapDate.value = urlParams.get("date");
if (urlParams.get("metric")) controls.metric.value = urlParams.get("metric");

refreshDashboard();
