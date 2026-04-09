# 2D Tolerance Stack-Up Simulator

A Python-based foundation for a mechanical engineering simulation tool focused on 2D tolerance stack-up analysis.

## Project Goals

- Provide a clean, domain-driven architecture for tolerance analysis workflows.
- Keep business logic out of the UI layer.
- Enable modular, extensible components for future simulation and reporting features.

## Current Structure

- `app/main.py` — Streamlit entrypoint + simulation workflow orchestration.
- `app/ui/geometry_canvas.py` — interactive 2D drawing/manipulation layer (drawable canvas).
- `app/ui/feature_mapper.py` — maps UI interactions to domain `Feature` objects.
- `app/ui/types.py` — UI-only dataclasses used to isolate view models from domain entities.
- `core/` — unchanged domain entities, tolerance models, simulation engine.
- `analysis/` — unchanged analytics for mean/failure computations.
- `infra/plotting/` — plotting adapters for simulation visualization.
- `tests/` — unit and integration tests.

## Requirements

- Python 3.11

## Setup

1. Create and activate a virtual environment:

   ```bash
   python3.11 -m venv .venv
   source .venv/bin/activate
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:

   ```bash
   streamlit run app/main.py
   ```

## UI/Domain Separation

- The **UI layer** is responsible for geometry drawing, user interaction, and feature-configuration forms.
- The **mapper layer** converts UI specs into immutable domain `Feature` objects.
- The **domain and simulation layers** (`core/`, `analysis/`) remain unchanged and UI-agnostic.
