# Virtual Try‑On — UI‑Only Streamlit Starter

This is a front-end scaffold you can run **without any model/backend**.
It lets you upload a person image and a clothing image, tweak placement,
and generate a **mock composite** preview so you can iterate on the UI.

## Quickstart

```bash
pip install -r requirements.txt
streamlit run app.py
```

- Sidebar controls adjust the cloth placement/scale/opacity.
- The banner shows **Backend: Not connected** – by design.
- Later, replace the mock `simple_overlay(...)` with real API calls
  (Nova Canvas, OmniTry, TryOnDiffusion, etc.).
