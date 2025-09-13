# ClearCut – Lokale Bild‑Hintergrundentfernung

Eine Desktop-App mit moderner PyQt6‑GUI, die Bilder lädt und den Hintergrund komplett lokal mit dem U²‑Net‑Modell entfernt. Ergebnis wird als PNG mit Transparenz exportiert.

## Features
- Lokale Inferenz, keine externen APIs
- U²‑Net (kleines `u2netp` Gewicht) für schnelle Ergebnisse
- PyQt6 GUI mit zweigeteilter Vorschau (links Original, rechts Cutout)
- JPG/PNG Input, PNG mit Transparenz als Output
- Lokale Gewichte: Lege `u2netp.pth` oder `u2net.pth` in `u2net/models/` (kein Download erforderlich)
- Helles/Dunkles Theme via OS (PyQt6 Styles) – kann leicht erweitert werden

## Screenshots
![Screenshot 1](docs/screenshot-1.png)
![Screenshot 2](docs/screenshot-2.png)

> Lege eigene Screenshots in `docs/` ab, die Platzhalter-Dateien fehlen bewusst.

## Installation

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
python app.py
```

Hinweise:
- Lege die Gewichte in `u2net/models/` ab: `u2netp.pth` (klein, ~4.7 MB) oder `u2net.pth` (größer, qualitativ besser). Es erfolgt kein automatischer Download.
- Für beste Performance wird automatisch CUDA genutzt, falls verfügbar.

## Nutzung
1. `Bild öffnen` wählen (JPG oder PNG)
2. `Hintergrund entfernen` starten
3. `Als PNG speichern` auswählen, um mit Alphakanal zu exportieren

## Paketierung
Einfaches Packaging via `pyproject.toml`:

```bash
python -m build
```

Das erzeugt ein Wheel in `dist/` (setuptools wird verwendet).

## Projektstruktur
```
app.py                 # GUI‑Einstiegspunkt
u2net/                 # Modellcode & Utilities
  ├─ __init__.py
  ├─ model.py          # U²‑Net Architektur (kompakte Variante für Inferenz)
  ├─ infer.py          # Pre-/Postprocessing und Inferenz
  └─ downloader.py     # Automatischer Download der Gewichte
models/                # Gewichtsdateien (wird automatisch angelegt)
requirements.txt       # Python Abhängigkeiten
pyproject.toml         # Packaging Konfiguration
README.md              # Dieses Dokument
```

## Lizenz
- Der Modellcode basiert auf dem Open-Source Projekt [U‑2‑Net](https://github.com/xuebinqin/U-2-Net) (MIT-Lizenz). Bitte beachte deren Lizenzbestimmungen.
- Dieser Code steht unter der MIT-Lizenz.
