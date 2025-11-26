# CS281A Final Project

## Abstract
**Event-Timed Modeling to Forecast 31-day mortality rate using EHR Time Series Data**

In healthcare settings, electronic health record (EHR) time series are both irregular and contain a large amount of missing values. While the data often includes auxiliary modalities like clinical notes, the primary challenge lies in correctly modeling the time-based structure of high-frequency measurements such as vital signs and lab results. Traditionally, missing observations first need to be imputed, after which irregular temporal patterns would be forced onto a fixed grid rather than modeled. Recent work[1] on multimodal time series provides strong baselines and dataset characterization, but methods that integrate imputation with irregular-aware modeling for healthcare forecasting remain limited. In this project, we will focus on the MIMIC-IV dataset[2], an ICU dataset collected at a hospital in Boston. This dataset contains rich, irregular time series data, including key patient measurements (like vital signs, lab results, and medication administration). Our task is to predict a clinically relevant outcome, such as in-hospital mortality or hospital length of stay. We will compare three approaches: (1) a baseline that regularizes the series onto a fixed daily grid with filled gaps, (2) an irregular-aware model that integrates the timestamp information, and (3) an irregular-explainable model that also tries to model missingness and time gaps as useful information. Our goal is to demonstrate that irregular-aware and explainable architectures significantly outperform grid-based baselines for timely risk stratification in healthcare.

*Reference:*
[1] Chang, Ching, Jeehyun Hwang, Yidan Shi, Haixin Wang, Wen-Chih Peng, Tien-Fu Chen, and Wei Wang. "Time-IMM: A Dataset and Benchmark for Irregular Multimodal Multivariate Time Series." arXiv preprint arXiv:2506.10412 (2025).
[2] Johnson, Alistair EW, et al. "MIMIC-IV, a freely accessible electronic health record dataset." Scientific data 10.1 (2023): 1.
[3] Yang, Xinyu, Yu Sun, and Xinyang Chen. "Frequency-aware generative models for multivariate time series imputation." Advances in Neural Information Processing Systems 37 (2024): 52595-52623.



## ICU Cohort Construction (features+labels)
- created a clean ICU cohort table (icu_cohort_small) by joining:
  - mimiciv_icu.icustays
  - mimiciv_hosp.admissions
  - mimiciv_hosp.patients
- Inclusion & extracted fields:
  - Adults (anchor_age >= 18)
  - Valid ICU intime and outtime
  - stay_id, subject_id, hadm_id
  - ICU LOS (hours)
  - Demographics (gender, anchor_age)
  - Outcome label: in-hospital mortality (hospital_expire_flag)

**This yields `data/icu_cohort_small.csv`: one row per ICU stay with demographics and outcomes.** 

Big Query used:

```
CREATE OR REPLACE TABLE `physionet-479306.my_dataset.icu_cohort_small` AS
SELECT
  icu.stay_id,
  icu.hadm_id,
  icu.subject_id,
  icu.intime,
  icu.outtime,
  DATETIME_DIFF(icu.outtime, icu.intime, HOUR) AS icu_los_hours,
  p.gender,
  p.anchor_age,   -- age at anchor_year (good proxy for age)
  a.hospital_expire_flag AS died_in_hosp
FROM `physionet-data.mimiciv_3_1_icu.icustays` AS icu
JOIN `physionet-data.mimiciv_3_1_hosp.admissions` AS a
  USING (subject_id, hadm_id)
JOIN `physionet-data.mimiciv_3_1_hosp.patients` AS p
  USING (subject_id)
WHERE p.anchor_age >= 18         -- adults only
  AND icu.intime IS NOT NULL
  AND icu.outtime IS NOT NULL;
```

## Irregular Vital-Sign Time Series Extraction

We built an irregular time-series table (vitals_ts_small) from:
- mimiciv_icu.chartevents
- imiciv_icu.d_items
- icu_cohort_small

Vital signs included (using d_items lookup):

- Heart rate
- Respiratory rate
- Mean arterial blood pressure
- SpO₂

Key processing choices:

- Extracted measurements only within the first 48 hours of each ICU stay
- Preserved true irregular timestamps, no resampling or imputation

Saved data in long format:

- stay_id
- charttime
- variable
- value

**This yields `data/vitals_ts_small.csv`** 

Big query used:
```
SELECT DISTINCT itemid, label
FROM `physionet-data.mimiciv_3_1_icu.d_items`
WHERE LOWER(label) IN (
  'heart rate',
  'respiratory rate',
  'o2 saturation pulseoxymetry',
  'mean blood pressure'
)
ORDER BY label, itemid;

CREATE OR REPLACE TABLE `physionet-479306.my_dataset.vitals_ts_small` AS
WITH vitals_items AS (
  SELECT
    itemid,
    LOWER(label) AS label
  FROM `physionet-data.mimiciv_3_1_icu.d_items`
  WHERE LOWER(label) IN (
    'heart rate',
    'respiratory rate',
    'o2 saturation pulseoxymetry',
    'mean blood pressure'
  )
)
SELECT
  ce.stay_id,
  ce.charttime,
  vi.label        AS variable,   -- e.g., "heart rate"
  ce.valuenum     AS value
FROM `physionet-data.mimiciv_3_1_icu.chartevents` AS ce
JOIN vitals_items AS vi
  USING (itemid)
JOIN `physionet-479306.my_dataset.icu_cohort_small` AS cohort
  USING (stay_id)
WHERE ce.charttime BETWEEN cohort.intime
                       AND DATETIME_ADD(cohort.intime, INTERVAL 48 HOUR)
  AND ce.valuenum IS NOT NULL;

```