# Cell‑Tracker

**Cell‑Tracker** is a Python toolbox for tracking red‑blood‑cell (RBC) nuclei across time‑lapse microscopy sequences.  It combines a classic constant‑velocity **Kalman filter** with a **large‑deformation diffeomorphic metric mapping (LDDMM)** module that can be driven by either MSE or Sinkhorn (unbalanced optimal‑transport) loss.

---

## Features
- **Modular configuration** – select a scenario (`baseline`, `model_a`, `model_b`, `baseline_ukf`, `model_b_ukf`) via the `--scenario` flag.  The configuration lives in `utils/configs.py`.
- **Kalman‑only baseline** – fast, linear motion model for quick prototyping.
- **LDDMM‑enhanced division handling** – robust matching of parent‑daughter cell pairs using a diffeomorphic deformation field.
- **Sinkhorn OT loss** – optional unbalanced optimal‑transport loss for more accurate division matching without handcrafted feature engineering.
- **Active‑pixel extraction** – sparse‑pixel strategy that avoids OOM errors while preserving full‑resolution accuracy.
- **Export to MOTChallenge format** – `run_tracking.py` writes `tracks.csv` for downstream evaluation.
- **Evaluation suite** – `evaluate.py` parses ground‑truth XML and computes MOTA, ID switches, etc.

---

## Quick start
```bash
# 1. Install dependencies
pip install -r requirements.txt   # torch, geomloss, motmetrics, etc.

# 2. Run segmentation to obtain detections (once)
python run_segmentation_stardist.py   # produces detections.json

# 3. Run a tracking scenario
python run_tracking.py --scenario baseline      # Kalman only
python run_tracking.py --scenario model_a       # Kalman + LDDMM (MSE)
python run_tracking.py --scenario model_b       # Kalman + LDDMM (Sinkhorn)

# 4. Evaluate the results
python evaluate.py
```

---

## Configuration file (`utils/configs.py`)
Key hyper‑parameters are grouped under `BaseTrackerHyperParams` and specialised per scenario.  Important fields:
- `kalman_filter_type` – `"kalman"` (default) or `"ukf"` (unscented Kalman filter).
- `lddmm_loss_type` – `"mse"` or `"sinkhorn"`.
- `lddmm_iterations`, `lddmm_sigma`, `lddmm_step_size` – control the gradient‑descent optimisation of the velocity field.
- `lddmm_distance_threshold` – radius (pixels) for candidate daughter‑cell search.

---

## Running a custom experiment
1. Add a new class inheriting from `BaseTrackerHyperParams`.
2. Register it in the `_SCENARIOS` dictionary at the bottom of `configs.py`.
3. Call `python run_tracking.py --scenario <your_name>`.

---

## License
This project is released under the MIT License.
