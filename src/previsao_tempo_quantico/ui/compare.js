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
  meta: null,
  payload: null,
  statesGeo: null,
  statesLayer: null,
  focusLayer: L.layerGroup().addTo(map),
};

const controls = {
  scope: document.getElementById("scopeSelect"),
  uf: document.getElementById("ufInput"),
  municipalityLimit: document.getElementById("municipalityLimit"),
  region: document.getElementById("regionSelect"),
  compareDate: document.getElementById("compareDate"),
  refreshFiltersBtn: document.getElementById("refreshFiltersBtn"),
  compareBtn: document.getElementById("compareBtn"),
  statusText: document.getElementById("statusText"),
  sourceTime: document.getElementById("sourceTime"),
  observedTime: document.getElementById("observedTime"),
  coverageTop: document.getElementById("coverageTop"),
  tempRmse: document.getElementById("tempRmse"),
  tempMae: document.getElementById("tempMae"),
  rainRmse: document.getElementById("rainRmse"),
  rainMae: document.getElementById("rainMae"),
  regionMeta: document.getElementById("regionMeta"),
  compareRows: document.getElementById("compareRows"),
};

function fmt(value, decimals = 2) {
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

function baseStateStyle() {
  return {
    color: "#d7e5f3",
    weight: 1.1,
    fillOpacity: 0.2,
    fillColor: "#4c7fab",
  };
}

function resetStateStyles() {
  if (!state.statesLayer) return;
  state.statesLayer.eachLayer((layer) => {
    layer.setStyle(baseStateStyle());
  });
}

function focusState(uf) {
  if (!state.statesLayer || !uf) return;
  const targetUf = uf.toUpperCase();
  let targetBounds = null;

  state.statesLayer.eachLayer((layer) => {
    const layerUf = parseUfFromFeature(layer.feature);
    const isTarget = layerUf === targetUf;
    layer.setStyle(
      isTarget
        ? { color: "#ffffff", weight: 1.9, fillOpacity: 0.62, fillColor: "#29c49a" }
        : baseStateStyle()
    );
    if (isTarget) {
      targetBounds = layer.getBounds?.();
    }
  });

  if (targetBounds && targetBounds.isValid()) {
    map.fitBounds(targetBounds.pad(0.35));
  }
}

function focusPoint(lat, lon, label) {
  if (!Number.isFinite(lat) || !Number.isFinite(lon)) return;
  const marker = L.circleMarker([lat, lon], {
    radius: 8,
    color: "#ffffff",
    weight: 1.4,
    fillColor: "#2cd0b4",
    fillOpacity: 0.96,
  })
    .bindPopup(label || "Regiao selecionada")
    .addTo(state.focusLayer);
  marker.openPopup();
  map.setView([lat, lon], 8);
}

async function fetchJson(url) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Falha ${response.status} em ${url}`);
  }
  return response.json();
}

async function ensureStatesLayer() {
  if (state.statesLayer) return;
  state.statesGeo = await fetchJson("/api/geo/states");
  state.statesLayer = L.geoJSON(state.statesGeo, {
    style: () => baseStateStyle(),
  }).addTo(map);
}

function syncScopeFields() {
  const isMunicipio = controls.scope.value === "municipio";
  controls.uf.disabled = !isMunicipio;
  controls.municipalityLimit.disabled = !isMunicipio;
  if (!isMunicipio) {
    controls.uf.value = "";
  }
}

function renderRegionOptions(regions, preferredValue = "") {
  controls.region.innerHTML = "";
  const sorted = (regions || []).slice().sort((a, b) => {
    const an = String(a.region_name || "");
    const bn = String(b.region_name || "");
    return an.localeCompare(bn, "pt-BR");
  });

  for (const item of sorted) {
    const option = document.createElement("option");
    option.value = String(item.region_id || "");
    option.textContent = `${item.region_name || "-"} (${item.uf || "-"})`;
    controls.region.appendChild(option);
  }

  if (!sorted.length) return;
  if (preferredValue) {
    const idx = sorted.findIndex((item) => String(item.region_id || "") === String(preferredValue));
    if (idx >= 0) {
      controls.region.selectedIndex = idx;
      return;
    }
  }
  controls.region.selectedIndex = 0;
}

function updateDateBounds(meta) {
  controls.compareDate.min = meta.min_date || "";
  controls.compareDate.max = meta.max_selectable_date || "";
  if (!controls.compareDate.value) {
    controls.compareDate.value = meta.recommended_date || meta.max_selectable_date || "";
  }

  if (controls.compareDate.min && controls.compareDate.value < controls.compareDate.min) {
    controls.compareDate.value = meta.recommended_date || controls.compareDate.min;
  }
  if (controls.compareDate.max && controls.compareDate.value > controls.compareDate.max) {
    controls.compareDate.value = controls.compareDate.max;
  }
}

async function loadMeta(preserveRegion = true) {
  controls.statusText.textContent = "Carregando filtros de comparacao...";
  const regionBefore = preserveRegion ? controls.region.value : "";
  const scope = controls.scope.value;
  const uf = controls.uf.value.trim().toUpperCase();
  const municipalityLimit = Number(controls.municipalityLimit.value) || 1200;

  const params = new URLSearchParams({ scope });
  if (scope === "municipio") {
    if (uf) params.set("uf", uf);
    params.set("municipality_limit", String(municipalityLimit));
  }

  try {
    const meta = await fetchJson(`/api/forecast/compare24h/meta?${params.toString()}`);
    state.meta = meta;

    if (meta.error) {
      controls.statusText.textContent = meta.error;
      renderRegionOptions([], "");
      return;
    }

    updateDateBounds(meta);
    renderRegionOptions(meta.regions || [], regionBefore);
    controls.statusText.textContent =
      `Filtros carregados: ${meta.count || 0} regioes | Base observada ate ${meta.max_observed_date || "-"}.`;
  } catch (error) {
    controls.statusText.textContent = `Erro ao carregar filtros: ${error.message}`;
  }
}

function errorClass(value) {
  const num = Math.abs(Number(value));
  if (!Number.isFinite(num)) return "";
  if (num <= 1.0) return "err-low";
  if (num <= 3.0) return "err-mid";
  return "err-high";
}

function renderRows(rows) {
  controls.compareRows.innerHTML = "";
  if (!rows || !rows.length) {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td colspan="9">Sem dados para comparacao.</td>`;
    controls.compareRows.appendChild(tr);
    return;
  }

  for (const row of rows) {
    const tr = document.createElement("tr");
    const tempErrClass = errorClass(row.error_temperature_c);
    const probErrClass = errorClass(row.error_rain_probability_pct);

    tr.innerHTML = `
      <td>${row.hour_label || "-"}</td>
      <td>${fmt(row.pred_temperature_c, 1)}</td>
      <td>${fmt(row.obs_temperature_c, 1)}</td>
      <td class="${tempErrClass}">${fmt(row.error_temperature_c, 2)}</td>
      <td>${fmt(row.pred_rain_probability_pct, 1)}</td>
      <td>${fmt(row.obs_rain_probability_pct, 1)}</td>
      <td class="${probErrClass}">${fmt(row.error_rain_probability_pct, 2)}</td>
      <td>${row.pred_condition || "-"}</td>
      <td>${row.obs_condition || "-"}</td>
    `;
    controls.compareRows.appendChild(tr);
  }
}

function renderSummary(payload) {
  const metrics = payload.metrics || {};
  const region = payload.region || {};
  controls.sourceTime.textContent = fmtDate(payload.source_time);
  controls.observedTime.textContent = fmtDate(payload.observed_source_time || payload.data_source_time);
  controls.coverageTop.textContent = `${fmt(metrics.coverage_pct, 1)}%`;

  controls.tempRmse.textContent = fmt(metrics.temperature_rmse, 3);
  controls.tempMae.textContent = fmt(metrics.temperature_mae, 3);
  controls.rainRmse.textContent = fmt(metrics.rain_probability_rmse, 3);
  controls.rainMae.textContent = fmt(metrics.rain_probability_mae, 3);

  controls.regionMeta.innerHTML = `
    <strong>${region.region_name || "-"}</strong> (${region.uf || "-"})<br/>
    Data: ${payload.date || "-"}<br/>
    Lat/Lon: ${fmt(region.lat, 4)}, ${fmt(region.lon, 4)}<br/>
    Amostras validas: ${metrics.samples || 0}/24
  `;
}

function renderFocus(payload) {
  state.focusLayer.clearLayers();
  resetStateStyles();

  const region = payload.region || {};
  const scope = payload.scope;
  if (scope === "state") {
    focusState(String(region.uf || ""));
    return;
  }
  focusPoint(Number(region.lat), Number(region.lon), `${region.region_name || "-"} (${region.uf || "-"})`);
}

async function runComparison() {
  const date = controls.compareDate.value.trim();
  const scope = controls.scope.value;
  const region = controls.region.value;
  const uf = controls.uf.value.trim().toUpperCase();
  const municipalityLimit = Number(controls.municipalityLimit.value) || 1200;

  if (!date) {
    controls.statusText.textContent = "Selecione uma data.";
    return;
  }
  if (!region) {
    controls.statusText.textContent = "Selecione uma regiao.";
    return;
  }

  controls.statusText.textContent = "Processando comparacao prevista x real...";
  const params = new URLSearchParams({
    date,
    region,
    scope,
    municipality_limit: String(municipalityLimit),
  });
  if (scope === "municipio" && uf) {
    params.set("uf", uf);
  }

  try {
    const payload = await fetchJson(`/api/forecast/compare24h?${params.toString()}`);
    if (payload.error) {
      controls.statusText.textContent = payload.error;
      renderRows([]);
      return;
    }

    state.payload = payload;
    renderSummary(payload);
    renderRows(payload.rows || []);
    renderFocus(payload);
    controls.statusText.textContent = `Comparacao carregada: ${payload.date} | ${payload.region?.region_name || "-"}.`;
  } catch (error) {
    controls.statusText.textContent = `Erro na comparacao: ${error.message}`;
  }
}

controls.scope.addEventListener("change", async () => {
  syncScopeFields();
  await loadMeta(false);
});

controls.refreshFiltersBtn.addEventListener("click", () => loadMeta(true));
controls.compareBtn.addEventListener("click", runComparison);

controls.uf.addEventListener("keydown", async (event) => {
  if (event.key === "Enter" && controls.scope.value === "municipio") {
    await loadMeta(false);
  }
});

const urlParams = new URLSearchParams(window.location.search);
const scopeParam = (urlParams.get("scope") || "").toLowerCase();
const ufParam = (urlParams.get("uf") || "").toUpperCase();
const dateParam = urlParams.get("date") || "";
const regionParam = urlParams.get("region") || "";

if (scopeParam === "state" || scopeParam === "municipio") controls.scope.value = scopeParam;
if (ufParam) controls.uf.value = ufParam;
if (dateParam) controls.compareDate.value = dateParam;

async function bootstrap() {
  await ensureStatesLayer();
  syncScopeFields();
  await loadMeta(true);

  if (regionParam) {
    const idx = Array.from(controls.region.options).findIndex((opt) => String(opt.value) === String(regionParam));
    if (idx >= 0) controls.region.selectedIndex = idx;
  }

  if (controls.region.value && controls.compareDate.value) {
    await runComparison();
  } else {
    controls.statusText.textContent = "Selecione filtros e execute a comparacao.";
  }
}

bootstrap();
