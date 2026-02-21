const map = L.map("map", { zoomControl: true, preferCanvas: true }).setView([-14.2, -52.2], 4);

L.tileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", {
  attribution: "Tiles (c) Esri",
  maxZoom: 10,
}).addTo(map);

L.tileLayer("https://{s}.basemaps.cartocdn.com/light_only_labels/{z}/{x}/{y}{r}.png", {
  attribution: "(c) OpenStreetMap (c) CARTO",
  maxZoom: 10,
  opacity: 0.95,
}).addTo(map);

const state = {
  statesGeo: null,
  statesLayer: null,
  markerLayer: L.layerGroup().addTo(map),
  cities: [],
  selectedPayload: null,
};

const controls = {
  uf: document.getElementById("uf"),
  loadCitiesBtn: document.getElementById("loadCitiesBtn"),
  citySelect: document.getElementById("citySelect"),
  snapDate: document.getElementById("snapDate"),
  searchBtn: document.getElementById("searchBtn"),
  statusText: document.getElementById("statusText"),
  cityMeta: document.getElementById("cityMeta"),
  dateMeta: document.getElementById("dateMeta"),
  summaryBox: document.getElementById("summaryBox"),
  snapRows: document.getElementById("snapRows"),
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

function todayISODate() {
  const now = new Date();
  const shifted = new Date(now.getTime() - now.getTimezoneOffset() * 60000);
  return shifted.toISOString().slice(0, 10);
}

function parseUfFromFeature(feature) {
  const props = feature?.properties || {};
  return (props.sigla || props.SIGLA || props.uf || props.UF || props.sigla_uf || "").toString().toUpperCase();
}

function parseStateName(feature) {
  const props = feature?.properties || {};
  return props.name || props.NOME || props.nome || props.NAME || "Estado";
}

async function fetchJson(url) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Falha ${response.status} em ${url}`);
  }
  return response.json();
}

async function ensureStatesLayer() {
  if (state.statesGeo) return;
  state.statesGeo = await fetchJson("/api/geo/states");
  state.statesLayer = L.geoJSON(state.statesGeo, {
    style: () => ({
      color: "#dce8f4",
      weight: 1.1,
      fillOpacity: 0.12,
      fillColor: "#4b84b7",
    }),
    onEachFeature: (feature, layer) => {
      const uf = parseUfFromFeature(feature);
      layer.bindTooltip(`${parseStateName(feature)} (${uf})`, { sticky: true });
    },
  }).addTo(map);
}

function selectedCityName() {
  const option = controls.citySelect.selectedOptions?.[0];
  return option ? option.value : "";
}

function populateCitySelect(preferredCity = "") {
  controls.citySelect.innerHTML = "";
  for (const city of state.cities) {
    const option = document.createElement("option");
    option.value = city.region_name;
    option.textContent = `${city.region_name} (${city.uf})`;
    controls.citySelect.appendChild(option);
  }

  if (!state.cities.length) return;

  const preferred = normalizeText(preferredCity);
  if (preferred) {
    const index = state.cities.findIndex((item) => normalizeText(item.region_name) === preferred);
    if (index >= 0) {
      controls.citySelect.selectedIndex = index;
      return;
    }
  }

  controls.citySelect.selectedIndex = 0;
}

async function loadCities(preferredCity = "") {
  const uf = controls.uf.value.trim().toUpperCase();
  const params = new URLSearchParams({
    scope: "municipio",
    limit: "5570",
  });
  if (uf) params.set("uf", uf);

  controls.statusText.textContent = "Carregando cidades...";

  try {
    const payload = await fetchJson(`/api/regions?${params.toString()}`);
    state.cities = (payload.items || []).slice().sort((a, b) => {
      const aName = String(a.region_name || "");
      const bName = String(b.region_name || "");
      return aName.localeCompare(bName, "pt-BR");
    });
    populateCitySelect(preferredCity);

    if (state.cities.length === 0) {
      controls.statusText.textContent = "Nenhuma cidade encontrada para a UF informada.";
    } else {
      controls.statusText.textContent = `${state.cities.length} cidades carregadas.`;
    }
  } catch (error) {
    controls.statusText.textContent = `Erro ao carregar cidades: ${error.message}`;
  }
}

function renderMap(payload) {
  state.markerLayer.clearLayers();
  const lat = Number(payload.lat);
  const lon = Number(payload.lon);
  if (!Number.isFinite(lat) || !Number.isFinite(lon)) return;

  const marker = L.circleMarker([lat, lon], {
    radius: 8,
    color: "#ffffff",
    weight: 1.4,
    fillColor: "#2cd0b4",
    fillOpacity: 0.96,
  })
    .bindPopup(
      `<strong>${payload.city || "-"}</strong> (${payload.uf || "-"})<br/>` +
        `Lat/Lon: ${fmt(lat, 4)}, ${fmt(lon, 4)}<br/>` +
        `Data: ${payload.date || "-"}`
    )
    .addTo(state.markerLayer);

  marker.openPopup();
  map.setView([lat, lon], 8);
}

function renderSummary(payload) {
  const forecasts = payload.forecasts || [];
  const h6 = forecasts.find((item) => Number(item.horizon_hours) === 6) || {};
  const h24 = forecasts.find((item) => Number(item.horizon_hours) === 24) || {};
  const availableDates = payload.available_dates || [];

  controls.summaryBox.innerHTML = `
    <div><strong>${payload.city || "-"}</strong> (${payload.uf || "-"})</div>
    <div>T+6: ${fmt(h6.temperature_c, 1)} C | P+6: ${fmt(h6.rain_probability_pct, 1)}%</div>
    <div>T+24: ${fmt(h24.temperature_c, 1)} C | P+24: ${fmt(h24.rain_probability_pct, 1)}%</div>
    <div>Base previsao: ${fmtDate(payload.source_time)}</div>
    <div>Base dados: ${fmtDate(payload.data_source_time)}</div>
    <div>Datas disponiveis: ${availableDates.join(", ") || "-"}</div>
  `;
}

function renderRows(payload) {
  controls.snapRows.innerHTML = "";
  const rows = payload.hourly || [];

  if (!rows.length) {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td colspan="5">Sem registros para cidade/data selecionadas.</td>`;
    controls.snapRows.appendChild(tr);
    return;
  }

  for (const row of rows) {
    const hourOfDay = Number(row.hour_of_day);
    const hourLabel = Number.isFinite(hourOfDay)
      ? `${String(hourOfDay).padStart(2, "0")}:00`
      : `${String(Number(row.hour_offset) || 0).padStart(2, "0")}:00`;
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${hourLabel}</td>
      <td>${fmt(row.temperature_c, 1)}</td>
      <td>${fmt(row.rain_probability_pct, 1)}</td>
      <td>${fmt(row.rain_mm_h, 2)}</td>
      <td>${row.condition || "-"}</td>
    `;
    controls.snapRows.appendChild(tr);
  }
}

async function searchSnap() {
  const city = selectedCityName();
  const uf = controls.uf.value.trim().toUpperCase();
  const date = controls.snapDate.value.trim();

  if (!city) {
    controls.statusText.textContent = "Selecione uma cidade.";
    return;
  }
  if (!date) {
    controls.statusText.textContent = "Selecione uma data.";
    return;
  }

  controls.statusText.textContent = "Buscando snap da cidade/data...";

  const params = new URLSearchParams({
    city,
    date,
    municipality_limit: "5570",
  });
  if (uf) params.set("uf", uf);

  try {
    const payload = await fetchJson(`/api/forecast/snap?${params.toString()}`);
    if (payload.error) {
      controls.statusText.textContent = payload.error;
      renderRows({ hourly: [] });
      controls.summaryBox.textContent = "Sem dados para o snap.";
      return;
    }

    state.selectedPayload = payload;
    if (!controls.snapDate.value) {
      controls.snapDate.value = payload.date || date;
    }
    controls.cityMeta.textContent = `${payload.city || "-"} (${payload.uf || "-"})`;
    controls.dateMeta.textContent = payload.date || date;

    renderSummary(payload);
    renderRows(payload);
    renderMap(payload);
    controls.statusText.textContent = `Snap carregado: ${payload.city || "-"} em ${payload.date || date}.`;
  } catch (error) {
    controls.statusText.textContent = `Erro ao buscar snap: ${error.message}`;
  }
}

controls.loadCitiesBtn.addEventListener("click", () => loadCities(selectedCityName()));
controls.searchBtn.addEventListener("click", searchSnap);

controls.uf.addEventListener("keydown", (event) => {
  if (event.key === "Enter") loadCities(selectedCityName());
});

controls.citySelect.addEventListener("change", () => {
  if (controls.snapDate.value) searchSnap();
});

controls.snapDate.addEventListener("change", () => {
  if (selectedCityName()) searchSnap();
});

async function bootstrap() {
  const urlParams = new URLSearchParams(window.location.search);
  const ufParam = (urlParams.get("uf") || "").trim().toUpperCase();
  const cityParam = (urlParams.get("city") || "").trim();
  const dateParam = (urlParams.get("date") || "").trim();

  if (ufParam) controls.uf.value = ufParam;
  controls.snapDate.value = dateParam || todayISODate();

  await ensureStatesLayer();
  await loadCities(cityParam);

  if (selectedCityName()) {
    await searchSnap();
  } else {
    controls.statusText.textContent = "Carregue as cidades e selecione uma cidade/data.";
  }
}

bootstrap();
