import sys
import os
import subprocess
import importlib

def running_in_notebook():
    try:
        from IPython import get_ipython  # noqa
        ip = get_ipython()
        if ip is None:
            return False
        return "IPKernelApp" in ip.config
    except Exception:
        return False

IN_NOTEBOOK = running_in_notebook()

# Auto-install dependencies

REQUIRED_PACKAGES = [
    ("matplotlib", "matplotlib"),
    ("geopandas", "geopandas"),
    ("shapely", "shapely"),
    ("contextily", "contextily"),
    ("osmnx", "osmnx"),
    ("networkx", "networkx"),
    ("scikit-learn", "sklearn"),
]


def ensure_packages_installed():
    missing_pip = []
    for pip_name, import_name in REQUIRED_PACKAGES:
        try:
            importlib.import_module(import_name)
        except ImportError:
            missing_pip.append(pip_name)

    if not missing_pip:
        return

    print("Installing missing packages:", missing_pip)
    subprocess.check_call([sys.executable, "-m", "pip", "install", *missing_pip])

ensure_packages_installed()

import csv
import itertools

import matplotlib.pyplot as plt
import geopandas as gpd
import contextily as ctx

import osmnx as ox
import networkx as nx

# Config: file path works in script 

try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 
except NameError:
    BASE_DIR = os.getcwd()

STATIONS_FILE = os.path.join(BASE_DIR, "metro.csv")

ox.settings.log_console = False
ox.settings.use_cache = True

WALK_SPEED_KMH = 4.8
BIKE_SPEED_KMH = 15.0

# Load stations
def load_stations():
    if not os.path.exists(STATIONS_FILE):
        raise FileNotFoundError(
            f"metro.csv not found at:\n{STATIONS_FILE}\n\n"
            "Put metro.csv in the same folder as this script (or current working directory in notebooks)."
        )

    stations = {}
    with open(STATIONS_FILE, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            name = row["nom_arret"].strip()
            lat, lon = row["coordonné gps"].replace(" ", "").split(",")
            stations[name] = {
                "name": name,
                "lat": float(lat),
                "lon": float(lon),
                "line": row.get("ligne", "").strip(),
            }
    return stations

def load_walking_graph():
    return ox.graph_from_place("Paris, France", network_type="walk")

# Time
def meters_to_minutes(distance_m, speed_kmh):
    speed_m_per_s = (speed_kmh * 1000.0) / 3600.0
    minutes = (distance_m / speed_m_per_s) / 60.0
    return round(minutes)

# Routing
def walking_distance(G, s1, s2):
    orig = ox.nearest_nodes(G, s1["lon"], s1["lat"])
    dest = ox.nearest_nodes(G, s2["lon"], s2["lat"])
    path = nx.shortest_path(G, orig, dest, weight="length")

    dist = 0.0
    for u, v in zip(path[:-1], path[1:]):
        ed = G.get_edge_data(u, v)
        dist += min(data.get("length", 0.0) for data in ed.values())

    return path, dist

def tsp_walking(stations, G):
    best_order = None
    best_distance = float("inf")

    cache = {}
    def dist(a, b):
        key = (a["name"], b["name"])
        if key in cache:
            return cache[key]
        _, d = walking_distance(G, a, b)
        cache[key] = d
        return d

    for perm in itertools.permutations(stations):
        total = 0.0
        for i in range(len(perm) - 1):
            total += dist(perm[i], perm[i + 1])
            if total >= best_distance:
                break
        if total < best_distance:
            best_distance = total
            best_order = perm

    return best_order, best_distance

# Plot route

def plot_walking_route(G, route, output_file="route.png"):
    lines = []

    for i in range(len(route) - 1):
        path, _ = walking_distance(G, route[i], route[i + 1])

        gdf_path = ox.routing.route_to_gdf(G, path)
        lines.extend(list(gdf_path["geometry"]))

    gdf = gpd.GeoDataFrame(geometry=lines, crs="EPSG:4326").to_crs(epsg=3857)

    fig, ax = plt.subplots(figsize=(10, 10))
    gdf.plot(ax=ax, linewidth=3, color="red", zorder=2)
    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)
    ax.set_axis_off()

    plt.title("Optimal Walking Route (real streets)")
    plt.savefig(output_file, dpi=200, bbox_inches="tight")
    plt.show()

def run_text_mode():
    print("Running in text mode (notebook/colab).")
    print("Make sure metro.csv is in:", BASE_DIR)
    print("Type station names separated by commas.\n")

    stations_dict = load_stations()
    G = load_walking_graph()

    user_input = input("Stations: ").strip()
    names = [s.strip() for s in user_input.split(",") if s.strip()]

    if len(names) < 2:
        print("Please enter at least two stations.")
        return

    unknown = [n for n in names if n not in stations_dict]
    if unknown:
        print("Unknown stations:\n" + "\n".join(unknown))
        return

    stations = [stations_dict[n] for n in names]
    route, dist_m = tsp_walking(stations, G)

    walk_time = meters_to_minutes(dist_m, WALK_SPEED_KMH)
    bike_time = meters_to_minutes(dist_m, BIKE_SPEED_KMH)

    print("\nOPTIMAL ORDER:")
    for i, s in enumerate(route, 1):
        print(f"{i}. {s['name']} (Line {s['line']})")

    print(f"\nTotal distance: {dist_m/1000:.2f} km")
    print(f"Walking time (@ {WALK_SPEED_KMH} km/h): {walk_time} min")
    print(f"Biking  time (@ {BIKE_SPEED_KMH} km/h): {bike_time} min")

    plot_walking_route(G, route)

# Tkinter

def run_gui_mode():
    import tkinter as tk
    from tkinter import messagebox, scrolledtext

    class MetroApp:
        def __init__(self, root): 
            self.root = root
            self.root.title("Metro Walking Route Optimizer")
            self.root.geometry("600x600")

            try:
                self.stations = load_stations()
            except Exception as e:
                messagebox.showerror("Error", str(e))
                raise

            self.G = load_walking_graph()

            tk.Label(root, text="Metro Walking Route Optimizer", font=("Arial", 16, "bold")).pack(pady=10)

            tk.Label(root, text="Enter stations (comma-separated):").pack()
            self.entry = tk.Entry(root, width=60)
            self.entry.pack(pady=5)

            tk.Button(root, text="Compute walking route", command=self.compute).pack(pady=10)

            self.output = scrolledtext.ScrolledText(root, width=70, height=20)
            self.output.pack(pady=10)

        def compute(self):
            self.output.delete("1.0", tk.END)

            names = [s.strip() for s in self.entry.get().split(",") if s.strip()]
            if len(names) < 2:
                messagebox.showerror("Error", "Please enter at least two stations.")
                return

            unknown = [n for n in names if n not in self.stations]
            if unknown:
                messagebox.showerror("Unknown stations", "\n".join(unknown))
                return

            stations = [self.stations[n] for n in names]

            self.output.insert(tk.END, "Computing optimal walking route...\n")
            self.root.update()

            route, dist_m = tsp_walking(stations, self.G)

            walk_time = meters_to_minutes(dist_m, WALK_SPEED_KMH)
            bike_time = meters_to_minutes(dist_m, BIKE_SPEED_KMH)

            self.output.insert(tk.END, "\nOPTIMAL ORDER:\n\n")
            for i, s in enumerate(route, 1):
                self.output.insert(tk.END, f"{i}. {s['name']} (Line {s['line']})\n")

            self.output.insert(tk.END, f"\nTotal distance: {dist_m/1000:.2f} km\n")
            self.output.insert(tk.END, "\n--- TIME ESTIMATES (rounded) ---\n")
            self.output.insert(tk.END, f"Walking (@ {WALK_SPEED_KMH:.1f} km/h): {walk_time} min\n")
            self.output.insert(tk.END, f"Biking  (@ {BIKE_SPEED_KMH:.1f} km/h): {bike_time} min\n")

            plot_walking_route(self.G, route)

    root = tk.Tk()
    app = MetroApp(root)
    root.mainloop()

if __name__ == "__main__":
    if IN_NOTEBOOK:
        run_text_mode()
    else:
        run_gui_mode()
