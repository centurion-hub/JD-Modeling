# JD Modeling & Customer Choice Analysis (Demo Version)

This repository is a **demo version** of a larger JD.com analytics project.  
The full project contains extensive analyses and modeling tasks around **customer choice prediction, product structure optimization, promotion strategy evaluation, inventory allocation efficiency**, and more.  

⚠️ **Important Note**  
Due to data confidentiality, **only demo datasets and partial scripts** are provided here.  
The global model and certain sensitive pipelines are intentionally removed.  
This demo repository demonstrates the methodology and reproducibility without exposing business data.

---

## 📖 Project Overview

The full project covered seven research questions:

1. **Customer Choice Prediction**  
   - Built click–purchase training sets with SKU attributes, user profiles, and channel data.  
   - Constructed binary classification models (LightGBM) for predicting purchases.  
   - Identified key predictors such as product attributes, gender, brand, and purchase power.

2. **Feature Importance & Predictive Power**  
   - Extracted feature importances globally and by subgroups.  
   - Found attributes and brand variables strongly influence purchasing.

3. **Heterogeneity Analysis**  
   - Compared choice mechanisms across channels (WeChat, App, PC).  
   - Analyzed city-level differences and brand loyalty effects.

4. **Product Structure Optimization**  
   - Defined "similar product groups" by brand and attributes.  
   - Found that too many similar SKUs **reduce sales** due to intra-category competition.

5. **Target Customer Segmentation**  
   - Focused on female customers in Tier-1 cities.  
   - Identified sensitivity to brand and quality, suggesting positioning with premium products.

6. **Pricing & Promotion Strategies**  
   - Built regression models with log(sales).  
   - Bundling promotions showed the strongest positive effect, followed by quantity discounts and coupons.  
   - Direct discounts and gifts were less effective.

7. **Fulfillment & Inventory Optimization**  
   - Merged order, delivery, inventory, and network datasets.  
   - Found that non-local fulfillment is not always linked to delays, but mismatched inventory is.  
   - Recommended increasing local stock days for popular items and improving multi-level allocation.

---

## 📂 Repository Structure (Demo)

```
JD-Modeling-TrainingOnly/
│
├── data/
│   └── demo_training_dataset.csv   # anonymized demo dataset
│
├── src/
│   ├── model_by_region_trainingonly.py   # model by city_level
│   └── model_by_channel_trainingonly.py  # model by channel
│
├── results/                        # output folder (auto-created)
├── requirements.txt
└── README.md
```

---

## 🚀 Usage

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run region-level model:
   ```bash
   python src/model_by_region_trainingonly.py
   ```
   - Prints AUC per region.  
   - Saves results into `results/region_auc_trainingonly.csv` and feature importance plot.

3. Run channel-level model:
   ```bash
   python src/model_by_channel_trainingonly.py
   ```
   - Prints AUC per channel.  
   - Saves feature importance plot for one sample channel.

---

## 📊 Demo Dataset

- File: `data/demo_training_dataset.csv`  
- Aligned with the schema of the real JD training dataset but anonymized and simulated.  
- Contains user IDs, SKU IDs, purchase labels, channel info, product attributes, brand IDs, demographics, and region indicators.

This allows others to **reproduce the pipeline** without access to sensitive business data.

---

## 🔒 Why only two scripts?

The **global model script** and raw data cannot be published due to confidentiality.  
To ensure reproducibility, this repository includes:
- **Region-based modeling** (`city_level`)
- **Channel-based modeling** (`channel`)

These two scripts showcase the **core methodology** (preprocessing, LightGBM modeling, AUC evaluation, feature importance analysis) while protecting sensitive data.

---

## 🧩 Skills Demonstrated

Through this project, I gained:
- Proficiency in **pandas** for large-scale data cleaning and merging.  
- Mastery of **LightGBM** for classification/regression.  
- Ability to build **heterogeneous models** for different user groups.  
- Experience with **matplotlib** for visualization and exploratory analysis.  
- Skills in **feature importance interpretation** for business insights.  

---

## 📬 Contact

For collaboration or more details (academic/industry use), please contact:  
📧 hongzhen080@gmail.com
