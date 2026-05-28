# 🔌 EcoPlug — AI Smart Energy & Biodiversity Dashboard
### Python Flask + Pandas + Matplotlib + Scikit-learn

---

## 🚀 How to Run

### Step 1 — Install Python Libraries
```bash
pip install -r requirements.txt
```

### Step 2 — Run the App
```bash
python app.py
```

### Step 3 — Open in Browser
```
http://localhost:5000
```

---

## 📁 Project Structure
```
ecoplug/
│
├── app.py                  ← Main Flask app
├── requirements.txt
├── README.md
│
├── data/
│   ├── energydata_complete.csv   ← Home appliance energy (UCI)
│   ├── owid-co2-data.csv         ← Global / India CO₂ (OWID)
│   ├── Species.csv               ← Endangered species
│   └── long_data_.csv            ← India state electricity usage
│
└── templates/
    └── index.html
```

---

## 📊 Datasets Used

| # | File | Used for |
|---|------|----------|
| 1 | `energydata_complete.csv` | Appliance watt estimates, energy charts |
| 2 | `owid-co2-data.csv` | India CO₂ trends, emission factor, ML forecast |
| 3 | `Species.csv` | Species risk & planet health score |
| 4 | `long_data_.csv` | State list, usage benchmarks, bill estimates |

---

## 🛠️ Tech Stack

- **Backend:** Python 3.x + Flask
- **Data:** Pandas
- **Charts:** Matplotlib
- **ML:** Scikit-learn Linear Regression (India CO₂ forecast)
- **Frontend:** HTML + CSS + JavaScript
