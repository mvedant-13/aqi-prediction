# Delhi-NCR AQI Prediction

A machine learning project that predicts Air Quality Index (AQI) for Delhi-NCR
using weather and temporal features, without relying on pollutant sensor readings.

---

## Problem Statement

The CPCB formula for AQI calculation requires live pollutant sensor readings
(PM2.5, PM10, NO₂, SO₂, CO, O₃) that are often unavailable, delayed, or
inconsistent across stations. This project builds an ML regression model that
predicts AQI purely from **meteorological and temporal features** — inputs that
are far more widely available and reliably measured. At inference time, a user
provides temperature, humidity, wind speed, and time/location context, and the
model outputs the predicted AQI.

---

## Dataset

- **Records:** 201,664 hourly readings (2020–2025)
- **Cities:** Delhi, Noida, Gurugram, Faridabad, Ghaziabad
- **Stations:** 23 monitoring stations
- **Raw features:** Pollutants (PM2.5, PM10, NO₂, SO₂, CO, O₃) + Weather
  (Temperature, Humidity, Wind Speed, Visibility)
- **Target:** AQI (range: 25–500)

---

## Key EDA Findings

- Mean AQI **265.83** — Delhi-NCR averages the Poor category year-round
- **Severe** is the most common category (29.8% of readings)
- **Winter (avg. 458)** is the worst season; **Monsoon (avg. 83)** is the best
- **November** averages AQI 500 — the CPCB maximum cap
- **Temperature (−0.73)** and **Wind Speed (−0.54)** are the strongest
  legitimate predictors — cold, still air traps pollutants
- PM2.5 and PM10 are 0.99 correlated — they rise and fall together

---

## Preprocessing

**Columns dropped (12 total):**

| Column(s) | Reason |
|---|---|
| `datetime`, `date` | Redundant — year/month/day/hour already present |
| `latitude`, `longitude` | Redundant — station encodes location |
| `aqi_category` | Derived directly from AQI — target leakage |
| `pm25`, `pm10` | Primary CPCB AQI sub-index inputs — target leakage |
| `no2`, `so2`, `co`, `o3` | Remaining CPCB sub-index inputs — target leakage |
| `visibility` | Synthetically derived from pollutant levels in this dataset — acts as a proxy for PM, target leakage |

**Note on leakage:** India's CPCB AQI is computed as the maximum sub-index
across all six pollutants (PM2.5, PM10, NO₂, SO₂, CO, O₃). Including any of
these as features gives the model a direct back-channel to the target, producing
artificially inflated accuracy (R² → 0.9999). Removing them reduces R² to a
realistic ~0.97 for XGBoost, which reflects genuine predictive power from
weather and seasonal patterns.

**Final feature set (12 features):**
`temperature`, `humidity`, `wind_speed`, `hour`, `day_of_week`, `is_weekend`,
`month`, `year`, `season`, `city`, `station`

**Other steps:**
- Categorical encoding — ordinal for `season`, `day_of_week`; label encoding
  for `city`, `station`
- StandardScaler applied (fit on train only, transform applied to both)
- Random 80/20 train/test split — no temporal features present, so a random
  split gives a more representative test set across all seasons and years

---

## Known Limitations

**AQI ceiling at 500:** The dataset contains 45,314 records (22.48%) where AQI
is capped at 500 by the CPCB formula. Different levels of extreme pollution are
indistinguishably mapped to the same target value, which reduces model accuracy
at the upper end of the scale. This is an inherent limitation of the data source.

**Synthetic dataset:** This dataset was generated rather than collected from live
CPCB feeds. The visibility column was found to be derived from pollutant levels
(correlation −0.86 with AQI) rather than being an independent meteorological
observation, which is why it was excluded.

---

## Model Results

All models trained on 12 weather and temporal features. No pollutant data used.

| Model | RMSE | MAE | R² |
|---|---|---|---|
| Linear Regression | 53.58 | 42.65 | 0.9024 |
| Random Forest | 35.75 | 26.67 | 0.9566 |
| **XGBoost** | **29.36** | **23.19** | **0.9707** |

**Best model: XGBoost** — average prediction error of ~23 AQI points, which is
within one AQI category boundary. Linear Regression achieving R² = 0.902 from
weather alone confirms how strong Delhi's seasonal pollution signal is.

---

## Tech Stack

Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, XGBoost, Flask