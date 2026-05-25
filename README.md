<h1>A Real-Time GPS-less Hybrid Localization System for IoT Battery Operated Tracking Applications</h1>

<p>
  <img src="https://img.shields.io/badge/Python-3.9+-blue">
  <img src="https://img.shields.io/badge/LoRa-LPWAN-red">
  <img src="https://img.shields.io/badge/BLE-Bluetooth_Low_Energy-green">
  <img src="https://img.shields.io/badge/Wi--Fi-RSSI-yellow">
  <img src="https://img.shields.io/badge/WKNN-Localization-blue">
  <img src="https://img.shields.io/badge/Machine_Learning-Fingerprinting-red">
  <img src="https://img.shields.io/badge/IoT-GPS--less_Tracking-green">
  <img src="https://img.shields.io/badge/License-MIT-yellow">
</p>

> ### Electronics & Computer Engineering — VIT Chennai, April 2026  
> **Adithya Ajikumar • Joseph Alex Valluvassery • S Saran**

---

## The Problem

Most existing localization systems rely heavily on GPS for positioning and tracking. While GPS works effectively in open outdoor environments, it struggles in indoor areas, dense urban regions, and signal-obstructed environments where satellite signals become weak or unavailable. In addition, GPS modules consume significant power, making them unsuitable for long-term battery-operated IoT devices.

Modern IoT applications such as smart campuses, logistics, asset tracking, and environmental monitoring require a localization system that is:
- Low power
- Cost efficient
- Scalable
- Reliable in GPS-denied environments

Traditional RSSI-based localization methods also face several challenges including:
- Signal noise and multipath interference
- Environmental variability
- Weak spatial discrimination
- Poor extrapolation outside anchor coverage

This project addresses a different problem than traditional GPS-based tracking systems:

> Not *“How can we improve GPS accuracy?”*  
> But *“Can we achieve reliable real-time localization without GPS using low-power wireless technologies and machine learning?”*

---

## What This Project Does

This project is a real-time GPS-less hybrid localization system that:

1. Collects LoRa, Wi-Fi, and BLE signals from multiple anchor nodes
2. Captures RSSI and SNR data using a mobile IoT receiver
3. Creates unique multi-radio fingerprints for each location
4. Extracts statistical and spatial signal features using feature engineering
5. Uses machine learning models to map signal patterns to real-world coordinates
6. Applies Weighted K-Nearest Neighbors (WKNN) for localization prediction
7. Performs RSSI spread and bounding-box analysis for reliability evaluation
8. Improves localization accuracy using Iterative Leave-One-Out (LOO) pruning
9. Achieves real-time GPS-free positioning in indoor and outdoor environments
10. Provides an energy-efficient alternative to traditional GPS-based tracking systems

---

## How It Works

```text
Anchor Nodes (LoRa + Wi-Fi + BLE)
        ↓
Signal Transmission & RSSI Collection
        ↓
Mobile Receiver Data Acquisition
        ↓
Fingerprint Creation
        ↓
Feature Engineering
        ↓
RSSI + SNR + Spatial Features
        ↓
Machine Learning Model (WKNN)
        ↓
Real-Time GPS-less Localization
        ↓
Location Prediction & Error Analysis
```

The system operates using a multi-radio fingerprinting architecture with multiple anchor nodes deployed at fixed coordinates.

- **Anchor Transmission** — LoRa, BLE, and Wi-Fi anchors continuously broadcast wireless signals
- **Data Collection** — A mobile receiver captures RSSI and SNR values at predefined grid locations
- **Fingerprint Generation** — Multiple signal packets are aggregated into unique spatial fingerprints
- **Feature Engineering** — Statistical, spatial, and physics-based features are extracted from raw signals
- **Localization Engine** — Weighted K-Nearest Neighbors (WKNN) predicts device coordinates using fingerprint similarity
- **Data Refinement** — RSSI spread analysis and iterative LOO pruning improve prediction reliability
- **Performance Evaluation** — Localization accuracy is measured using Haversine distance and cross-validation

The architecture enables scalable, low-power, and GPS-free localization suitable for smart campuses, IoT asset tracking, and mixed indoor-outdoor environments.

---

## Architecture

![GPS-less Localization Architecture](assets/architecture.png)

---

## Models Used

| Component | Model | Purpose |
|---|---|---|
| Multi-Radio Communication | LoRa + BLE + Wi-Fi | Hybrid wireless signal transmission |
| Feature Engineering | RSSI + SNR + Spatial Features | Signal pattern extraction |
| Localization Model | Weighted K-Nearest Neighbors (WKNN) | Real-time coordinate prediction |
| Ensemble Models | Random Forest, ExtraTrees, KNN | Performance comparison and optimization |
| Physics-Based Estimation | LoRa Trilateration | Coarse location estimation |
| Evaluation Metric | Haversine Distance | Localization error calculation |

The optimized WKNN model achieved a mean localization error of **9.82 meters**, significantly outperforming traditional single-radio localization methods.

The system combines statistical, spatial, and physics-based features to improve localization robustness in both indoor and outdoor GPS-denied environments.

---

## Technologies Used

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Wireless Communication | LoRa, BLE, Wi-Fi |
| Machine Learning | Weighted K-Nearest Neighbors (WKNN) |
| Ensemble Models | Random Forest, ExtraTrees, KNN |
| Feature Engineering | RSSI, SNR, Spatial Features |
| Localization Technique | Multi-Radio Fingerprinting |
| Distance Estimation | LoRa Trilateration |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib |
| Evaluation Metrics | Haversine Distance, Cross-Validation |
| Hardware Platforms | LilyGo T-LoRa C6, Heltec Wi-Fi LoRa 32 |
| Communication Protocols | LoRaWAN, BLE, IEEE 802.11 (Wi-Fi) |
| Development Environment | VS Code, Jupyter Notebook |

---

## Features

- Real-time GPS-less localization using hybrid wireless communication
- Multi-radio fingerprinting with LoRa, BLE, and Wi-Fi
- Low-power IoT tracking architecture for battery-operated devices
- RSSI and SNR based signal analysis
- Statistical, spatial, and physics-based feature engineering
- Weighted K-Nearest Neighbors (WKNN) localization engine
- RSSI spread analysis for prediction confidence estimation
- Iterative Leave-One-Out (LOO) pruning for data refinement
- Bounding-box analysis for interpolation vs extrapolation evaluation
- Real-time coordinate prediction with optimized localization accuracy
- Cross-validation and Haversine distance based performance evaluation
- Scalable architecture suitable for indoor and outdoor environments

---

## Results

The proposed hybrid localization system successfully achieved reliable GPS-free positioning using multi-radio fingerprinting and machine learning.

```text
LOCALIZATION SUMMARY

Total grid locations collected: 442
Final usable locations:       401–417
Total packets captured:       32,561

Best Model:                   Weighted KNN (WKNN)
Mean Localization Error:      9.82 meters
Baseline Error:               ~22.8 meters
Error Reduction Achieved:     57%

Evaluation Metrics:
- Mean Absolute Error (MAE)
- Median Error
- CDF Error Analysis
- Cross-Validation Performance
- Zone-wise Accuracy Analysis
```

--- 

## Usage

```bash
# Step 1 — Collect Multi-Radio Signal Data
python collect_data.py

# Step 2 — Perform Feature Engineering
python feature_engineering.py

# Step 3 — Train the Localization Model
python train_model.py \
    --k_neighbors 3 \
    --enable_loo_pruning true \
    --cross_validation true \
    --rssi_threshold 10

# Step 4 — Run Real-Time Localization
python localization.py \
    --dataset data/fingerprints.csv \
    --model wknn \
    --anchors anchors_config.json \
    --output predictions.csv
```

---

## Conclusion

The proposed GPS-less hybrid localization system demonstrates how multi-radio communication and machine learning can be combined to achieve reliable real-time positioning in IoT environments without relying on GPS.

By integrating LoRa, BLE, and Wi-Fi fingerprinting with advanced feature engineering and Weighted K-Nearest Neighbors (WKNN), the system effectively maps wireless signal patterns to physical coordinates with improved accuracy and robustness.

The architecture enables continuous signal acquisition, fingerprint generation, localization prediction, and data refinement, making the solution scalable and suitable for real-world deployment in mixed indoor-outdoor environments.

Through RSSI spread analysis, bounding-box evaluation, and iterative Leave-One-Out (LOO) pruning, the project significantly reduced localization ambiguity and achieved a mean localization error of **9.82 meters**.

This project highlights the potential of low-power hybrid wireless systems for smart campuses, asset tracking, logistics, and next-generation IoT localization applications in GPS-denied environments.

---

## Future Scope

- Expand anchor deployment for wider localization coverage
- Increase fingerprint dataset density for improved spatial accuracy
- Implement 3D localization using multi-level anchor placement
- Enhance localization robustness in dense indoor environments
- Integrate deep learning models for advanced signal pattern recognition
- Develop real-time localization dashboards and monitoring systems
- Improve RSSI filtering and adaptive signal calibration techniques
- Deploy the system on edge and embedded IoT devices
- Explore hybrid localization using additional wireless technologies
- Implement dynamic confidence estimation for unreliable predictions
- Optimize scalability for smart campus and large-scale asset tracking applications
- Reduce localization error further using advanced ensemble learning techniques

---

## Authors

- **Adithya Ajikumar**
- **Joseph Alex Valluvassery**
- **S Saran**

**School of Electronics Engineering**  
Vellore Institute of Technology, Chennai  
April 2026
