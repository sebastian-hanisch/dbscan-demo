# DBSCAN für Sammel-Routen ohne feste Anzahl – Streamlit-Demo

Drittes Stück der "Konzepte"-Reihe für die Website "Sebastian Hanisch – Operations
Research und Machine Learning", zweites Stück der **Clustering-Linie** nach
[kmeans-demo](../kmeans-demo): anders als die Fall-Demos im Portfolio (ein
Anwendungsfall, mehrere Verfahren im Vergleich) zeigt diese Demo **ein** Verfahren –
DBSCAN – und lässt stattdessen die **Parameterempfindlichkeit** wachsen. Vehikel-Problem:
Lieferadressen zu Sammel-Routen gruppieren, ohne die Anzahl der Routen vorher
festzulegen – dichte Nachbarschaften bilden automatisch eine Route, vereinzelte Adressen
werden explizit als "nicht routentauglich" (Noise) markiert statt zwangsweise
zugeordnet.

Die Konzepte-Reihe ist kein linearer Pfad, sondern mehrere unabhängige Linien: diese
Demo baut nicht auf [branch-bound-demo](../branch-bound-demo) (Exakte-Suche-Linie) auf,
sondern auf kmeans-demo. Die Kante ist konkret: DBSCAN behebt zwei Schwächen von
k-Means – k muss nicht vorab feststehen, und Cluster müssen nicht konvex/kugelförmig
sein. Geplante Fortsetzung dieser Linie: **HDBSCAN**, motiviert durch DBSCANs eigene
Schwäche bei stark unterschiedlicher Dichte – genau die Beobachtung, die diese Demo
selbst live nachweist (siehe unten).

## Warum diese Demo anders aufgebaut ist

Bei k-Means war die Schwierigkeitsachse der Zufall der Startpunkte. DBSCAN ist
deterministisch – hier wächst stattdessen, wie empfindlich das Ergebnis von den beiden
Parametern **eps** (Suchradius) und **min_samples** abhängt:

- **Einfaches Beispiel**: gleichmäßig dichte Gruppen – ein eps aus einer weiten Spanne
  funktioniert.
- **Nicht-konvexe Formen**: das klassische Zwei-Halbmonde-Beispiel – zeigt konkret, warum
  k-Means hier grundsätzlich scheitern würde, DBSCAN aber nicht.
- **Schwerer Fall**: eine dichte, enge Gruppe plus eine lockere, diffuse Gruppe (gleiche
  Punktzahl, nur unterschiedliche Streuung) – bei jedem eps, das die dichte Gruppe schon
  vollständig erfasst (0% Noise), bleibt die diffuse Gruppe noch großteils Noise. Live in
  der "📐"-Sektion nachgewiesen: der Noise-Anteil je wahrer Gruppe über eine eps-Spanne,
  nicht nur behauptet.
- **Viele Ausreißer**: zeigt den Kernvorteil gegenüber k-Means – Ausreißer werden
  explizit als Noise erkannt statt einem Zentrum zugewiesen.

## Visualisierung

Ein Animationsschritt ist ein besuchter Punkt (in der tatsächlichen Ausführungsreihenfolge
von `db_algorithm.run`) – ausgefüllte Punkte sind Core-Punkte, umrandete Punkte
Rand-Punkte (Border), graue Kreuze Noise. Direkt darunter, wie in kmeans-demo eingeführt:
eine Kleinmultiples-Reihe mit demselben Szenario bei 4 verschiedenen eps-Werten (Vielfache
des aktuell eingestellten Werts), macht die Parameterempfindlichkeit sofort sichtbar.

Die front-and-center "📐 Wie wählt man eps überhaupt?"-Sektion zeigt einen live berechneten
**k-Distanz-Plot** (Ester et al., 1996 – Standard-Heuristik zur eps-Wahl) sowie ein
**eps-Sweep-Diagramm**, das den Noise-Anteil insgesamt und je wahrer Gruppe über eine aus
den Daten abgeleitete eps-Spanne zeigt.

## Sicherheitsgrenzen

Keine eigene Iterationsgrenze nötig – DBSCAN terminiert nach spätestens `2n` Ereignissen
(jeder Punkt wird höchstens zweimal klassifiziert: einmal als vorläufiges Noise, einmal
bei späterer Aufnahme als Rand-Punkt). `N_POINTS_MAX` (400) hält die naive $O(n^2)$-Suche
schnell genug für eine flüssige Animation.

## Verifikation

- **Handgerechnete Kleinstinstanz**: vier Punkte auf einer Linie plus ein Ausreißer –
  Core-/Rand-/Noise-Klassifizierung von Hand nachgerechnet, unabhängig von der
  Besuchsreihenfolge (siehe `tests/test_algorithm.py`).
- **Kreuzvergleich mit scikit-learn**: dieselbe Partition (bis auf Cluster-ID-Umbenennung),
  dieselben Noise-Punkte, dieselbe Menge an Core-Punkten wie `sklearn.cluster.DBSCAN`,
  über mehrere Szenarien (Blobs, Halbmonde) und Parameterkombinationen (scikit-learn ist
  ausschließlich ein Test-Dependency, kein Laufzeit-Dependency der App).
- **Struktur-Invarianten**: jeder Punkt genau eine Rolle, Cluster-IDs lückenlos von 0 bis
  `n_clusters - 1`.
- **Kern-Behauptung der Demo direkt getestet**: bei stark unterschiedlicher Dichte gibt es
  ein eps, bei dem die dichte Gruppe 0% Noise hat, während die diffuse Gruppe noch
  deutlich über 30% Noise hat (`test_dense_and_sparse_group_need_different_eps_to_avoid_noise`).

## Dateistruktur

| Datei | Inhalt |
|---|---|
| `app.py` | Streamlit-Hauptablauf: Presets, Einstellungen, Besuchs-Animation, Kleinmultiples, eps-Werkzeuge, Formulierungs-Expander |
| `db_constants.py` | Defaults, Regler-Grenzen, Sicherheitsgrenzen, `PRESETS` |
| `db_presets.py` | `SettingSpec`/`SETTING_SPECS`, Permalink-Logik, Presets, Zufalls-Seed-Button |
| `db_scenario.py` | Zufällige Punktwolken: Gauß-Gruppen mit Dichte-Ungleichgewicht oder Zwei-Halbmonde, plus optionale verstreute Ausreißer |
| `db_algorithm.py` | DBSCAN from scratch mit vollständigem Besuchsprotokoll |
| `db_evaluation.py` | Live-Kennzahlen pro Schritt, k-Distanz-Werte, eps-Sweep (gesamt + je wahrer Gruppe) |
| `db_visualization.py` | Punktwolken-, k-Distanz- und eps-Sweep-Diagramm (Plotly) |
| `tests/` | Handinstanz, sklearn-Kreuzvergleich, Struktur-Invarianten, Szenario-Reproduzierbarkeit, eps-Sweep-Verhalten |

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

---

Teil des [Operations-Research-Demo-Portfolios](https://sebastianhanisch.net/demos.html) von
[Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning.
Interesse an einer maßgeschneiderten Lösung? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html).
