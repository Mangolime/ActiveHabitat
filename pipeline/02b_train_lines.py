"""Download suburban/MCD train routes and attach line ids to stations."""

import os
import re
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from shapely.geometry import LineString, MultiLineString, Point

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_extent import load_bbox_and_polygon

os.makedirs("data/processed", exist_ok=True)

OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]
HEADERS = {"User-Agent": "ActiveHabitat/1.0"}
SNAP_M = 250

# Пригородное движение отличается от дальнего тегом service, а не сетью:
# у ЦППК и МТППК стоит regional, у МЦД — commuter, у ФПК — long_distance.
SUBURBAN_SERVICE = {"commuter", "regional"}

# Диаметры проверяем первыми и возвращаем только их: в названии МЦД-1
# («Белорусско-Савёловский диаметр») сидят сразу два радиальных направления,
# и без этого одна линия считалась бы за три.
MCD_RULES = [
    (re.compile(r"мцд[-\s:]*1|d\s*1\b|диаметр\s*1", re.I), "МЦД D1"),
    (re.compile(r"мцд[-\s:]*2|d\s*2\b|диаметр\s*2", re.I), "МЦД D2"),
    (re.compile(r"мцд[-\s:]*3|d\s*3\b|диаметр\s*3", re.I), "МЦД D3"),
    (re.compile(r"мцд[-\s:]*4|d\s*4\b|диаметр\s*4", re.I), "МЦД D4"),
    (re.compile(r"мцд[-\s:]*5|d\s*5\b|диаметр\s*5", re.I), "МЦД D5"),
    (re.compile(r"мцк|центральн\w+ кольц", re.I), "МЦК"),
]

# Маршруты ЦППК названы по конечным станциям («Апрелевка => Балашиха»), поэтому
# кроме вокзалов узнаём направление по опорным станциям хода.
DIRECTION_RULES = [
    (re.compile(r"ярославск|мытищ|пушкино|фрязин|щёлково|монино|болшево"
                r"|сергиев|александров", re.I), "Ярославское"),
    (re.compile(r"казанск|раменск|люберц|куровск|шатура|черусти|голутвин"
                r"|коломна|виноградово", re.I), "Казанское"),
    (re.compile(r"горьковск|владимир|петушки|фрязево|купавна|балашиха"
                r"|электрогорск|храпуново|крутое|ногинск|железнодорожн"
                r"|нижегородск", re.I), "Горьковское"),
    (re.compile(r"курск|серпухов|чехов|подольск|царицыно|львовск|столбовая"
                r"|шарапова|щербинк|люблино", re.I), "Курское"),
    (re.compile(r"павелецк|домодедово|ступино|кашира|барыбино|бирюлёво"
                r"|расторгуево", re.I), "Павелецкое"),
    (re.compile(r"киевс?к|апрелевк|лесной городок|крёкшино|солнечная"
                r"|наро-фоминск|малояросл|селятино", re.I), "Киевское"),
    (re.compile(r"белорусск|одинцово|голицыно|кубинка|можайск|бородино"
                r"|дорохово|звенигород|усово", re.I), "Белорусское"),
    (re.compile(r"рижск|нахабино|дедовск|новоиерусалим|волоколамск"
                r"|шаховская|стрешнево", re.I), "Рижское"),
    (re.compile(r"савёл|савел|бутырск|лобня|икша|дмитров|дубна|талдом"
                r"|вербилки|большая волга", re.I), "Савёловское"),
    (re.compile(r"ленинград|октябрьск|крюково|зеленоград|клин|солнечногорск"
                r"|поварово|алабушево|подсолнечная", re.I), "Ленинградское"),
]


def directions(blob: str) -> set:
    blob = str(blob or "")
    hit = {name for pat, name in MCD_RULES if pat.search(blob)}
    if hit:
        return hit
    return {name for pat, name in DIRECTION_RULES if pat.search(blob)}


def fetch_json(query):
    last_err = None
    for endpoint in OVERPASS_ENDPOINTS:
        try:
            print(f"  -> {endpoint}")
            resp = requests.post(
                endpoint, data={"data": query}, headers=HEADERS, timeout=300
            )
            if resp.status_code == 200:
                return resp.json()
            last_err = f"HTTP {resp.status_code}"
        except Exception as exc:
            last_err = exc
    raise RuntimeError(f"Overpass failed: {last_err}")


def route_blob(tags) -> str:
    """Всё, по чему можно узнать направление.

    Раньше сюда шёл только ref, а у пригородных маршрутов ЦППК он выглядит как
    «=> Фрязино» — направление из такого не вытащить. Название при этом читается
    прекрасно: «Пригородный электропоезд: Ярославский вокзал => Фрязино».
    """
    tags = tags or {}
    return " ".join(str(tags.get(k) or "") for k in ("name", "ref", "network"))


def is_suburban(tags) -> bool:
    return str((tags or {}).get("service") or "").strip().lower() in SUBURBAN_SERVICE


def relation_geom(el):
    lines = []
    for member in el.get("members") or []:
        geom = member.get("geometry") or []
        if member.get("type") == "way" and len(geom) >= 2:
            coords = [(p["lon"], p["lat"]) for p in geom]
            lines.append(LineString(coords))
    if not lines:
        return None
    return MultiLineString(lines) if len(lines) > 1 else lines[0]


print("0. Контур выгрузки: гексы + 10 км...")
_, BBOX = load_bbox_and_polygon()

print("1. Станции railway=station|halt (не метро)...")
# Регулярка заякорена: без этого в выборку попадали узлы
# railway=train_station_entrance — в них тоже есть подстрока «station».
stations_raw = fetch_json(
    f"""
[out:json][timeout:300];
node["railway"~"^(station|halt)$"]["station"!="subway"]{BBOX};
out body;
"""
)
station_rows = []
for el in stations_raw.get("elements", []):
    if el.get("type") != "node" or "lat" not in el:
        continue
    tags = el.get("tags") or {}
    station_rows.append(
        {
            "osm_id": el["id"],
            "name": tags.get("name") or "",
            "network": tags.get("network") or "",
            "geometry": Point(el["lon"], el["lat"]),
        }
    )
stations = gpd.GeoDataFrame(station_rows, crs="EPSG:4326")
print(f"   станций: {len(stations)}")

print("2. Геометрия маршрутов route=train|light_rail...")
routes_raw = fetch_json(
    f"""
[out:json][timeout:300];
(
  relation["route"="train"]{BBOX};
  relation["route"="light_rail"]{BBOX};
);
out geom;
"""
)
route_rows = []
n_all = 0
n_named = 0
for el in routes_raw.get("elements", []):
    if el.get("type") != "relation":
        continue
    n_all += 1
    tags = el.get("tags") or {}
    if not is_suburban(tags):
        continue
    geom = relation_geom(el)
    if geom is None:
        continue
    dirs = directions(route_blob(tags))
    n_named += bool(dirs)
    route_rows.append({"dirs": ";".join(sorted(dirs)), "geometry": geom})
print(
    f"   relations всего: {n_all}, пригородных с геометрией: {len(route_rows)},"
    f" из них с узнанным направлением: {n_named}"
)

if not route_rows:
    raise RuntimeError("Не удалось собрать геометрию пригородных маршрутов.")

routes = gpd.GeoDataFrame(route_rows, crs="EPSG:4326").to_crs(32637)
stations_m = stations.to_crs(32637)
buf = routes.copy()
buf["geometry"] = routes.geometry.buffer(SNAP_M)
joined = gpd.sjoin(
    stations_m[["osm_id", "geometry"]],
    buf[["dirs", "geometry"]],
    how="left",
    predicate="intersects",
)
hit = joined.dropna(subset=["dirs"])
# сама привязка к пассажирскому маршруту и делает станцию электричкой: у грузовой
# «Рублёво» маршрутов нет, поэтому она больше не годится в «ближайшую электричку»
on_route = set(hit["osm_id"])
dir_map = hit.groupby("osm_id")["dirs"].agg(
    lambda s: ";".join(sorted({d for row in s for d in str(row).split(";") if d}))
)

# МЦК в OSM не оформлено маршрутным отношением — ни train, ни light_rail, ни
# subway. Опознаём его по тегу network самой станции.
own = stations["network"].fillna("").map(lambda s: ";".join(sorted(directions(s))))
merged = stations["osm_id"].map(dir_map).fillna("")
stations["lines"] = [
    ";".join(sorted({d for d in f"{a};{b}".split(";") if d}))
    for a, b in zip(merged, own)
]
stations["is_suburban"] = stations["osm_id"].isin(on_route) | (own.str.len() > 0)
# направление удалось назвать не у всех маршрутов, но если поезд тут
# останавливается, веток заведомо не ноль
named = stations["lines"].map(lambda s: 0 if not s else s.count(";") + 1)
stations["n_lines"] = np.where(
    stations["is_suburban"], np.maximum(named, 1), named
).astype(int)

n_sub = int(stations["is_suburban"].sum())
print(f"   пригородных станций (snap {SNAP_M} м): {n_sub} из {len(stations)}")
print(f"   из них с названным направлением: {int((named > 0).sum())}")

out = "data/processed/suburban_trains.gpkg"
if os.path.exists(out):
    os.remove(out)
stations.to_file(out, layer="stations", driver="GPKG")
print(f"3. Сохранено {out}")
if n_sub:
    print(stations.loc[named > 0, "lines"].value_counts().head(20).to_string())
