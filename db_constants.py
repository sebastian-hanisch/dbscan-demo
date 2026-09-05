"""Defaults, Regler-Grenzen, Sicherheitsgrenzen und Presets für die DBSCAN-Demo."""

DEFAULT_N_POINTS = 150
DEFAULT_K = 3
DEFAULT_SPREAD = 0.25
DEFAULT_DENSITY_IMBALANCE = 0.0
DEFAULT_OUTLIER_FRACTION = 0.0
DEFAULT_SEED = 7
DEFAULT_SHAPE = "blobs"
DEFAULT_EPS = 0.6
DEFAULT_MIN_SAMPLES = 5

N_POINTS_MIN, N_POINTS_MAX = 30, 400
K_MIN, K_MAX = 2, 6
SPREAD_MIN, SPREAD_MAX = 0.05, 0.9
DENSITY_IMBALANCE_MIN, DENSITY_IMBALANCE_MAX = 0.0, 1.0
OUTLIER_FRACTION_MIN, OUTLIER_FRACTION_MAX = 0.0, 0.3
EPS_MIN, EPS_MAX = 0.05, 3.0
MIN_SAMPLES_MIN, MIN_SAMPLES_MAX = 2, 15

SHAPES = ("blobs", "moons")
SHAPE_LABELS = {"blobs": "Gruppen (Blobs)", "moons": "Zwei Halbmonde"}

# Hard safety limit - DBSCAN selbst braucht keine Iterationsgrenze (terminiert
# nach spaetestens n besuchten Punkten), dies begrenzt nur die Animationsschritte.
MAX_STEPS_RENDERED = 400

# k-Distanz-/eps-Sweep-Sektion: Anzahl abgetasteter eps-Werte ueber die live
# aus den Daten abgeleitete Spanne.
EPS_SWEEP_STEPS = 40

PRESETS = {
    "Einfaches Beispiel (gleichmäßige Dichte)": {
        "n_points": 90, "k": 3, "spread": 0.15, "density_imbalance": 0.0,
        "outlier_fraction": 0.0, "shape": "blobs", "eps": 0.6, "min_samples": 5, "seed": 1,
    },
    "Nicht-konvexe Formen (zwei Halbmonde)": {
        "n_points": 150, "k": 3, "spread": 0.08, "density_imbalance": 0.0,
        "outlier_fraction": 0.0, "shape": "moons", "eps": 0.65, "min_samples": 5, "seed": 2,
    },
    "Schwerer Fall (sehr unterschiedliche Dichte)": {
        "n_points": 150, "k": 2, "spread": 0.12, "density_imbalance": 0.9,
        "outlier_fraction": 0.0, "shape": "blobs", "eps": 0.35, "min_samples": 5, "seed": 3,
    },
    "Viele Ausreißer (Noise-Erkennung im Fokus)": {
        "n_points": 160, "k": 3, "spread": 0.15, "density_imbalance": 0.0,
        "outlier_fraction": 0.2, "shape": "blobs", "eps": 0.5, "min_samples": 5, "seed": 4,
    },
}
