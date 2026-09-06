# Handoff — Strava Global Heatmap → растр МКАД + матч с drive-графом

**Created:** 2026-09-05  
**Purpose:** Повторить/продолжить съём heatmap и сшивку с OSM-графом внутри МКАД.  
**Артефакты:** `.scratch/strava-heatmap-msk/`  
**Этот файл:** `.scratch/handoffs/20260905-strava-heatmap-mkad/HANDOFF.md`

---

## Статус

| Шаг | Состояние |
|-----|-----------|
| Растр z12 (без логина) | готово, грубый |
| Растр z13 @2x (с сессией) | готово |
| Растр z14 @2x (с сессией) | готово — основной |
| Heat на `mkad_drive.graphml` | готово (z13 и z14) |

Главный тиф:  
`.scratch/strava-heatmap-msk/strava_msk_mkad_z14_auth_clip.tif` (~2.4 м/пикс, EPSG:3857)

---

## Важно про авторизацию

1. **Нужно быть залогиненным в Strava** в браузере (подписка / `globalHeatmapAllowed` — иначе высокие зумы не отдают).
2. Высокозумовые тайлы (`content-a.strava.com/identified/...`, z≥13) требуют **живой сессии браузера** (CloudFront / session cookies).
3. **Секреты в репозиторий и в этот handoff не класть** — ни CloudFront Policy/Signature, ни `_strava4_session`, ни Key-Pair-Id.
4. С диска Brave cookies надёжно не читались (HttpOnly + jar в памяти). Рабочий путь: **fetch тайлов из уже открытой вкладки Strava** через AppleScript → `execute javascript` (XHR `withCredentials`).
5. Активная вкладка Brave должна оставаться на `strava.com` на всё время съёмки. Если уйти на другой сайт — CORS/NetworkError, куча ERR.

Без логина публично открывается только низкий зум (`heatmap-external-*.strava.com/tiles/...`, примерно z≤12).

---

## Что это за данные

- Синие «треки» на Global Heatmap — **не отдельные GPS-линии**, а **растровая плотность**.
- Подложка `.mvt` / `.hfz` на `tiles.strava.com` — Mapbox/рельеф, не heat.
- Heat для карты Maps UI:  
  `https://content-a.strava.com/identified/globalheat/{sport}/{color}/{z}/{x}/{y}@2x.png?v=20&missing=empty`  
  Пример: `sport=all`, `color=mobileblue`, `@2x` → 1024×1024 на тайл.

Старый CDN `heatmap-external-*/tiles-auth/...` для этого UI не понадобился (InvalidKey без query-подписи); `content-a` + сессия вкладки — да.

---

## Bbox (МКАД + буфер)

```
west=37.28  east=37.93
south=55.49 north=55.99
```

| z | тайлов | ~м/пикс (@2x) |
|---|--------|----------------|
| 12 | 88 | ~21 (без auth, 256px) |
| 13 | 336 | ~4.8 |
| 14 | 1302 | ~2.4 |

z14 через Brave: ~15–20 мин при стабильной вкладке; при уходе со Strava — докачка pending.

---

## Как снимали растр (рабочий рецепт)

1. Открыть в Brave:  
   `https://www.strava.com/maps/global-heatmap?sport=All&gColor=mobileblue#14/55.75/37.62`  
   Убедиться, что heatmap виден (авторизован).
2. Батчами (6–8 тайлов) из JS на странице: sync XHR на `content-a...@2x.png` с `withCredentials`, ответ → base64 → Python пишет PNG в `tiles_z14_auth/`.
3. Периодически проверять `location.href` содержит `strava.com`; иначе вернуть URL вкладки на heatmap и ждать загрузку.
4. Мозаика: на каждый PNG — лёгкий georef VRT (EPSG:3857), `gdal.BuildVRT` → `gdal.Translate` / `Warp` clip по bbox.  
   Полный CreateCopy каждого тайла в GeoTIFF — медленно и рвётся; VRT быстрее.
5. Несколько PNG с другим color interpretation (Palette vs Gray) VRT может пропустить — единицы тайлов, на МКАД некритично.

---

## Матч с графом

Готовый drive-граф (МКАД + 20 м buffer):

`~/.agents/skills/vibe-map/vibemaps/examples/mkad_router/data/mkad_drive.graphml`  
(сборка: `build_graph.py` + `mkad_poly.gpkg`)

Сэмпл интенсивности растра вдоль ребра (~10–15 м), поля: `heat_mean`, `heat_p90`, `heat_max`, `heat_n`, `length_m`.

Env: conda `geo` (`osmnx`, `rasterio`, `geopandas`).

Ограничение: **drive**-сеть — тропы без автодорог могут не попасть. Walk/bike при необходимости: `build_graph_walk.py` (файл walk раньше не был собран).

---

## Артефакты

Каталог: `.scratch/strava-heatmap-msk/`

| Файл | Содержание |
|------|------------|
| `strava_msk_mkad_z14_auth_clip.tif` | основной растр z14 @2x |
| `strava_msk_mkad_z14_auth.tif` | полный мозаик тайлов |
| `preview_msk_z14.png` | превью |
| `tiles_z14_auth/` | сырые PNG |
| `mkad_drive_strava_heat_z14.gpkg` | все рёбра + heat |
| `mkad_drive_strava_heat_z14_top25.gpkg` | верхние 25% по `heat_mean` |
| `mkad_drive_strava_heat.gpkg` | то же для z13 |
| `mkad_strava_match_preview.html` | превью матча (z13 top25) |
| `download_z14.log` | лог съёмки |

---

## Если повторять / чинить

1. Залогиниться в Strava в Brave, открыть global-heatmap, не переключать активную вкладку.
2. Докачать missing из `tiles_z14_auth/` (или другой z) тем же JS-батом.
3. Пересобрать GeoTIFF через VRT-мозаику.
4. Пересчитать heat на graphml → gpkg.
5. Не коммитить cookies, session env, экспорты Cookie-Editor.

---

## Чего не делать

- Не рассчитывать на парсинг «векторных треков» из heatmap — их нет, только растр / агрегаты.
- Не писать CloudFront / session значения в git, чат, handoff.
- Не полагаться на чтение `Cookies` SQLite Brave для Strava session — в этой сессии jar не отдавал нужные записи с диска.
