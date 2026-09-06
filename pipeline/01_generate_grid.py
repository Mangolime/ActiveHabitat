import os
import requests
import geopandas as gpd
import h3
from shapely.geometry import Polygon

os.makedirs("data/raw", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

print("1. Загрузка полигонов районов Москвы...")
url = "https://raw.githubusercontent.com/codeforamerica/click_that_hood/master/public/data/moscow.geojson"
raw_path = "data/raw/moscow_districts_raw.geojson"

# Кешируем скачивание сырого файла, чтобы не дергать сеть каждый раз
if not os.path.exists(raw_path):
    with open(raw_path, "wb") as f:
        f.write(requests.get(url).content)

gdf_districts = gpd.read_file(raw_path)

# Отсекаем Новую Москву и Зеленоград
exclude = ["Троицк", "Щербинка", "поселение", "Зеленоград", "Крюково", "Силино", "Старое Крюково", "Матушкино", "Савелки"]
gdf_clean = gdf_districts[~gdf_districts["name"].str.contains("|".join(exclude), case=False, na=False)].copy()

# 📍 НОВОЕ: Сохраняем очищенные границы как отдельный слой для фронтенда и графа
print("2. Сохранение границ районов (target_districts.geojson)...")
gdf_clean.to_file("data/processed/target_districts.geojson", driver="GeoJSON")

print("3. Генерация сетки H3 Res 8...")
study_area = gdf_clean.union_all() if hasattr(gdf_clean, "union_all") else gdf_clean.unary_union

def poly_to_h3(geom, res):
    polys = [geom] if geom.geom_type == 'Polygon' else list(geom.geoms)
    cells = set()
    for poly in polys:
        ext = [(p[1], p[0]) for p in poly.exterior.coords]
        ints = [[(p[1], p[0]) for p in i.coords] for i in poly.interiors]
        if hasattr(h3, 'LatLngPoly'):
            cells.update(h3.polygon_to_cells(h3.LatLngPoly(ext, *ints), res))
        else:
            import shapely.geometry
            cells.update(h3.polyfill(shapely.geometry.mapping(poly), res, geo_json_conformant=True))
    return cells

hex_ids = poly_to_h3(study_area, 8)
hex_polygons = [Polygon([(p[1], p[0]) for p in (h3.cell_to_boundary(hid) if hasattr(h3, 'cell_to_boundary') else h3.h3_to_geo_boundary(hid))]) for hid in hex_ids]

gdf_h3 = gpd.GeoDataFrame({"h3_index": list(hex_ids), "geometry": hex_polygons}, crs="EPSG:4326")
joined = gpd.sjoin(gdf_h3.assign(geometry=gdf_h3.centroid), gdf_clean[["name", "geometry"]], how="left", predicate="within")
gdf_h3["district_name"] = joined["name"]

gdf_h3.to_parquet("data/processed/moscow_grid_h3.parquet")
print(f"Сетка сохранена: {len(gdf_h3)} гексагонов.")