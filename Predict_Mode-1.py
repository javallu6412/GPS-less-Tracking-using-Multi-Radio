# ── MODEL ARTIFACTS (same for both modes) ─────────────────────────────────────
MODEL_DIR        = "."                        # Folder where training saved artifacts
MODEL_PATH       = f"{MODEL_DIR}/final_model.pkl"
SCALER_PATH      = f"{MODEL_DIR}/final_scaler.pkl"
FEAT_COLS_PATH   = f"{MODEL_DIR}/feature_columns.csv"
CONFIG_PATH      = f"{MODEL_DIR}/model_config.json"

# ── MODE 1: CSV testing ────────────────────────────────────────────────────────
TEST_CSV         = "./DataSet_2/filtered_test_radio_data.csv"      # Raw packets CSV with ground truth

# ── MODE 2: Live serial ────────────────────────────────────────────────────────
SERIAL_PORT      = 'COM5'
BAUD_RATE        = 115200
COLLECTION_TIME  = 15                         # seconds to collect packets

# ── RADIO CONSTANTS ───────────────────────────────────────────────────────────
VALID_ANCHORS        = ['A1', 'A2', 'A3', 'A4', 'A5']
RSSI_FLOOR           = -120.0
SNR_FLOOR            = -20.0
EXPECTED_LORA_COUNT  = 30



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import joblib, json, warnings
from itertools import combinations

warnings.filterwarnings('ignore')


# ── Haversine distance ─────────────────────────────────────────────────────────
def haversine_m(la1, lo1, la2, lo2):
    R = 6_371_000.0
    a = (np.sin(np.radians((la2 - la1) / 2)) ** 2 +
         np.cos(np.radians(la1)) * np.cos(np.radians(la2)) *
         np.sin(np.radians((lo2 - lo1) / 2)) ** 2)
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


# ── IQR outlier filter ─────────────────────────────────────────────────────────
def iqr_filter(series, k=1.5):
    if len(series) < 4:
        return series
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    return series[(series >= q1 - k * iqr) & (series <= q3 + k * iqr)]


# ── Feature extraction: raw packets → feature vector ──────────────────────────
def extract_features(raw_df, feat_cols,
                     rssi_floor=RSSI_FLOOR, snr_floor=SNR_FLOOR,
                     expected_lora=EXPECTED_LORA_COUNT,
                     valid_anchors=None):
    """
    Converts a raw packet DataFrame (one location) into a feature vector
    matching the exact column order expected by the model.

    Parameters
    ----------
    raw_df : DataFrame with columns [anchor, protocol, rssi, snr, seq]
    feat_cols : list of feature names (from feature_columns.csv)

    Returns
    -------
    np.ndarray of shape (1, n_features)
    """
    if valid_anchors is None:
        valid_anchors = VALID_ANCHORS

    MISSING_FILL = {
        'BLE':  ['mean', 'std', 'min', 'max', 'count'],
        'LORA': ['mean', 'std', 'min', 'max', 'count', 'rate', 'snr_mean', 'snr_std'],
        'WIFI': ['mean', 'std', 'min', 'max', 'count'],
    }

    # Clean: valid anchors, RSSI range
    df = raw_df[raw_df['anchor'].isin(valid_anchors)].copy()
    df = df[(df['rssi'] >= -130) & (df['rssi'] <= -30)]

    row = {}

    # Base per-anchor-per-protocol stats
    for anchor in valid_anchors:
        adf = df[df['anchor'] == anchor]
        for proto in ['BLE', 'LORA', 'WIFI']:
            pdf  = adf[adf['protocol'] == proto]
            pfx  = f'{anchor}_{proto}'

            if len(pdf) == 0:
                for stat in MISSING_FILL[proto]:
                    if stat in ('count', 'rate', 'std', 'snr_std'):
                        row[f'{pfx}_{stat}'] = 0.0
                    elif stat == 'snr_mean':
                        row[f'{pfx}_{stat}'] = snr_floor
                    else:
                        row[f'{pfx}_{stat}'] = rssi_floor
            else:
                r = iqr_filter(pdf['rssi'])
                n = len(r)
                row[f'{pfx}_mean']  = r.mean()          if n > 0 else rssi_floor
                row[f'{pfx}_std']   = r.std()           if n > 1 else 0.0
                row[f'{pfx}_min']   = r.min()           if n > 0 else rssi_floor
                row[f'{pfx}_max']   = r.max()           if n > 0 else rssi_floor
                row[f'{pfx}_count'] = n

                if proto == 'LORA':
                    row[f'{pfx}_rate'] = n / expected_lora
                    s = iqr_filter(pdf['snr'])
                    row[f'{pfx}_snr_mean'] = s.mean() if len(s) > 0 else snr_floor
                    row[f'{pfx}_snr_std']  = s.std()  if len(s) > 1 else 0.0

    # DIFF features (LoRa)
    for a1, a2 in combinations(valid_anchors, 2):
        key = f'DIFF_LORA_{a1}_{a2}'
        if key in feat_cols:
            row[key] = row.get(f'{a1}_LORA_mean', rssi_floor) - row.get(f'{a2}_LORA_mean', rssi_floor)

    # DIFF features (BLE)
    for a1, a2 in [('A1', 'A3'), ('A1', 'A5'), ('A3', 'A5')]:
        key = f'DIFF_BLE_{a1}_{a2}'
        if key in feat_cols:
            row[key] = row.get(f'{a1}_BLE_mean', rssi_floor) - row.get(f'{a2}_BLE_mean', rssi_floor)

    # Return in correct column order
    vec = np.array([row.get(c, rssi_floor) for c in feat_cols], dtype=float)
    return vec.reshape(1, -1)


# ── Load model artifacts ───────────────────────────────────────────────────────
model     = joblib.load(MODEL_PATH)
scaler    = joblib.load(SCALER_PATH)
feat_cols = pd.read_csv(FEAT_COLS_PATH)['feature'].tolist()

with open(CONFIG_PATH) as f:
    config = json.load(f)

print("Model loaded successfully.")
print(f"  KNN k           : {config['knn_k']}")
print(f"  Training points : {config['n_training_points']}")
print(f"  CV MAE          : {config['cv_mae_m']} m")
print(f"  CV Median       : {config['cv_median_m']} m")
print(f"  CV P90          : {config['cv_p90_m']} m")
print(f"  Features        : {len(feat_cols)}")



# ── Load and inspect test CSV ──────────────────────────────────────────────────
test_raw = pd.read_csv(TEST_CSV)

# Parse ground truth coordinates
test_raw[['gt_lat', 'gt_lon']] = (
    test_raw['gps_coords'].str.strip().str.split(',', expand=True).astype(float)
)

locations = test_raw[['gps_coords', 'gt_lat', 'gt_lon']].drop_duplicates().reset_index(drop=True)

print(f"Test CSV loaded: {len(test_raw)} total packets")
print(f"Unique locations to predict: {len(locations)}")
print("\n\n")
print("Location list:")
for i, row in locations.iterrows():
    n_pkts = (test_raw['gps_coords'] == row['gps_coords']).sum()
    print(f"  [{i+1:2d}] {row['gps_coords']}  ({n_pkts} packets)")
    
    
    
    
    
# ── Run prediction for each location ──────────────────────────────────────────
results = []

print("\n\n")
print("-" * 75)
print(f"{'#':>3}  {'Ground Truth':^28}  {'Predicted':^28}  {'Error':>9}")
print(f"{'':>3}  {'lat':>12}  {'lon':>13}  {'lat':>12}  {'lon':>13}  {'(m)':>9}")
print("-" * 75)

for i, loc in locations.iterrows():
    # Extract packets for this location
    loc_packets = test_raw[test_raw['gps_coords'] == loc['gps_coords']].copy()

    # Build feature vector
    X_vec = extract_features(loc_packets, feat_cols)
    X_scaled = scaler.transform(X_vec)

    # Predict
    pred = model.predict(X_scaled)[0]
    pred_lat, pred_lon = pred[0], pred[1]

    # Error
    error_m = haversine_m(loc['gt_lat'], loc['gt_lon'], pred_lat, pred_lon)

    results.append({
        'location':   loc['gps_coords'],
        'gt_lat':     loc['gt_lat'],
        'gt_lon':     loc['gt_lon'],
        'pred_lat':   pred_lat,
        'pred_lon':   pred_lon,
        'error_m':    error_m,
        'n_packets':  len(loc_packets),
    })

    print(f"{i+1:3d}  {loc['gt_lat']:12.6f}  {loc['gt_lon']:13.6f}  "
          f"{pred_lat:12.6f}  {pred_lon:13.6f}  {error_m:9.2f}m")

results_df = pd.DataFrame(results)

print("\n\n")
print(f"{'=' * 50}")
print(f"  SUMMARY  ({len(results_df)} locations)")
print(f"{'=' * 50}")
print(f"  MAE          : {results_df['error_m'].mean():.2f} m")
print(f"  Median error : {results_df['error_m'].median():.2f} m")
print(f"  Std dev      : {results_df['error_m'].std():.2f} m")
print(f"  Min error    : {results_df['error_m'].min():.2f} m")
print(f"  Max error    : {results_df['error_m'].max():.2f} m")
print(f"  Within 10m   : {(results_df['error_m'] <= 10).sum()} / {len(results_df)} "
      f"({(results_df['error_m'] <= 10).mean()*100:.1f}%)")
print(f"  Within 15m   : {(results_df['error_m'] <= 15).sum()} / {len(results_df)} "
      f"({(results_df['error_m'] <= 15).mean()*100:.1f}%)")
print(f"{'=' * 50}")