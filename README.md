# 2D Tolerance Stack-Up Simulator

A Python-based foundation for a mechanical engineering simulation tool focused on 2D tolerance stack-up analysis.

## Project Goals

- Provide a clean, domain-driven architecture for tolerance analysis workflows.
- Keep business logic out of the UI layer.
- Enable modular, extensible components for future simulation and reporting features.

## Initial Structure

- `app/` — application entrypoints and UI-facing orchestration.
- `core/` — domain entities, value objects, and business rules.
- `analysis/` — simulation and analysis use cases/services.
- `infra/` — adapters for external systems and persistence.
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

3. Run the placeholder app:

   ```bash
   python -m app.main
   ```

## Notes

This repository is intentionally initialized with minimal scaffolding. Add domain models and analysis services first, then connect UI adapters (e.g., Streamlit) without embedding business logic in the interface.
