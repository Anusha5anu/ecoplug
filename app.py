from flask import Flask, render_template, request, jsonify
import pandas as pd
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import io, base64, os, re, math
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

app = Flask(__name__)

BASE = os.path.dirname(__file__)
DATA = os.path.join(BASE, 'data')

# ── Load your 4 datasets ──────────────────────────────────
df_energy = pd.read_csv(os.path.join(DATA, 'energydata_complete.csv'))
df_co2_raw = pd.read_csv(os.path.join(DATA, 'owid-co2-data.csv'), low_memory=False)
df_species_raw = pd.read_csv(os.path.join(DATA, 'Species.csv'))
df_usage = pd.read_csv(os.path.join(DATA, 'long_data_.csv'))

# ── Dataset 1: UCI-style appliance watts from energydata_complete ──
def _wh_to_watts(series):
    s = pd.to_numeric(series.astype(str).str.strip(), errors='coerce')
    return float(s.mean() * 6)  # 10-min Wh → average watts


def build_appliances_table():
    app_w = _wh_to_watts(df_energy['Appliances'])
    light_w = _wh_to_watts(df_energy['lights'])
    shares = {
        'Air Conditioner': 1500,
        'Refrigerator': 150,
        'Washing Machine': 500,
        'Television': 100,
        'Ceiling Fan': 75,
    }
    total = sum(shares.values())
    rows = []
    for name, share in shares.items():
        watts = max(1, round(app_w * share / total))
        rows.append({'appliance': name, 'watts': watts, 'source': 'energydata_complete'})
    rows.append({'appliance': 'Lights (10 LED)', 'watts': max(1, round(light_w)), 'source': 'energydata_complete'})
    return pd.DataFrame(rows)


df_appliances = build_appliances_table()

# ── Dataset 2: OWID CO₂ (India) ───────────────────────────
df_india = (
    df_co2_raw[df_co2_raw['country'] == 'India']
    .dropna(subset=['year', 'co2_per_capita'])
    .sort_values('year')
)
df_india_recent = df_india[df_india['year'] >= 2018].copy()

# Model 1: long-history linear forecast (year → CO₂ per capita)
MODEL1_START_YEAR = 1900
ML_START_YEAR = 2000  # Models 2 & 3 (need complete feature columns)

india_factor = float(df_india['co2_per_unit_energy'].dropna().iloc[-1])


def _india_model1_frame():
    """Full history shown on Model 1 chart."""
    return df_india[df_india['year'] >= MODEL1_START_YEAR].copy()


def _india_model1_train_frame():
    """Modern era used to train Model 1 forecast (avoids 1900s flattening the line)."""
    return df_india[df_india['year'] >= ML_START_YEAR].copy()

# ── Dataset 3: Species (endangered list) ──────────────────
def _population_risk(pop):
    """Base extinction risk from estimated population (log scale → varied %)."""
    s = str(pop).lower()
    if 'unknown' in s:
        return 46
    if 'few' in s or '<' in s:
        return 86
    nums = [int(x) for x in re.findall(r'\d+', str(pop).replace(',', ''))]
    if nums:
        n = max(1, min(nums))
        return int(min(93, max(30, 98 - 14 * math.log10(n))))
    return 50


def build_species_table():
    rows = []
    for _, row in df_species_raw.iterrows():
        name = row['Common Name']
        if pd.isna(name) or str(name).strip() in ('', '-'):
            name = row['Species Name']
        risk = _population_risk(row['Estimated Population'])
        rows.append({
            'species': name,
            'species_name': row['Species Name'],
            'type': row['Type'],
            'location': row['Location(s)'],
            'population': row['Estimated Population'],
            'threat': row['Threats'],
            'risk_pct': risk,
            'co2_threshold_tonnes': round(3.5 - risk / 40, 1),
        })
    return pd.DataFrame(rows)


df_species = build_species_table()

# ── Dataset 4: State electricity usage (long_data_) ───────
usage_by_state = df_usage.groupby('States')['Usage'].median().to_dict()
usage_national_median = float(df_usage['Usage'].median())


def calc_state_bill(units, state_name):
    """Estimate bill from state median usage in long_data_.csv."""
    state_median = usage_by_state.get(state_name, usage_national_median)
    rate = 4.0 + state_median / 75.0
    fixed = 50.0
    return round(fixed + units * rate)


# Home-electricity footprint is much smaller than national per-capita CO₂ (~2.1 t/yr).
HOME_CO2_BENCHMARK_KG_MONTH = 75.0
HOUSEHOLD_THRESHOLD_SCALE = 0.38

# Eco points: earned only when user checks AI tips (not on every Calculate click)
TIP_POINTS = 2
HIGH_USAGE_DAILY_KWH = 11.0
HIGH_USAGE_MONTHLY_KWH = 360.0
LOW_USAGE_DAILY_KWH = 6.0
LOW_USAGE_BONUS_PTS = 5
MEDIUM_USAGE_AUTO_PTS = 3


def calc_planet_health(monthly_co2_kg):
    annual_tonnes = (monthly_co2_kg * 12) / 1000
    scaled_thresholds = df_species['co2_threshold_tonnes'] * HOUSEHOLD_THRESHOLD_SCALE
    threatened = int((scaled_thresholds < annual_tonnes).sum())
    total = len(df_species)
    species_score = max(10, round(100 - (threatened / total) * 100))
    usage_ratio = monthly_co2_kg / HOME_CO2_BENCHMARK_KG_MONTH
    usage_score = max(10, round(100 - min(85, usage_ratio * 45)))
    return max(10, min(100, round((species_score + usage_score) / 2)))


# ── Charts ────────────────────────────────────────────────
def make_chart(labels, values, title, colors=None):
    fig, ax = plt.subplots(figsize=(8, 3.5))
    fig.patch.set_facecolor('#0e1a10')
    ax.set_facecolor('#080f0a')
    if not colors:
        colors = ['#00c45a', '#4dd9ff', '#f5c518', '#00ff7f', '#cc88ff', '#ff7744']
    bars = ax.bar(labels, values, color=colors[:len(labels)], edgecolor='none', width=0.5)
    ax.set_title(title, color='#e8f5e9', fontsize=11, pad=10)
    ax.tick_params(colors='#5a7a5e', labelsize=9)
    ax.spines['bottom'].set_color('#1e3a22')
    ax.spines['left'].set_color('#1e3a22')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, color='#1e3a22', linewidth=0.5)
    for bar, val in zip(bars, values):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f'{val:.1f}', ha='center', va='bottom', color='#e8f5e9', fontsize=8)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=120, bbox_inches='tight', facecolor='#0e1a10')
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def make_co2_trend_chart():
    india = df_india_recent
    fig, ax = plt.subplots(figsize=(8, 3))
    fig.patch.set_facecolor('#0e1a10')
    ax.set_facecolor('#080f0a')
    ax.plot(india['year'], india['co2_per_capita'], color='#00ff7f', linewidth=2.5, marker='o', markersize=5)
    ax.fill_between(india['year'], india['co2_per_capita'], alpha=0.15, color='#00ff7f')
    ax.set_title('India CO₂ Per Capita (tonnes) — OWID Dataset', color='#e8f5e9', fontsize=10, pad=8)
    ax.tick_params(colors='#5a7a5e', labelsize=8)
    for spine in ax.spines.values():
        spine.set_color('#1e3a22')
    ax.yaxis.grid(True, color='#1e3a22', linewidth=0.5)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=120, bbox_inches='tight', facecolor='#0e1a10')
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def compute_species_risks(annual_co2_tonnes):
    """Risk % for all species at a given annual CO₂ footprint (tonnes)."""
    # Small lift for every species so low footprints still differentiate
    footprint_lift = min(20, round((annual_co2_tonnes / 2.1) * 22))
    rows = []
    for _, row in df_species.iterrows():
        boosted = int(row['risk_pct']) + footprint_lift
        threshold = row['co2_threshold_tonnes'] * HOUSEHOLD_THRESHOLD_SCALE
        if annual_co2_tonnes > threshold:
            boosted += round((annual_co2_tonnes - threshold) * 12)
        boosted = min(99, boosted)
        rows.append({
            'species': row['species'],
            'type': row['type'],
            'risk_pct': boosted,
            'co2_threshold_tonnes': row['co2_threshold_tonnes'],
        })
    return pd.DataFrame(rows)


def top_species_by_risk(annual_co2_tonnes, n=12, for_chart=False):
    ranked = (
        compute_species_risks(annual_co2_tonnes)
        .sort_values(['risk_pct', 'species'], ascending=[False, True])
        .head(n)
    )
    if for_chart:
        return ranked.sort_values('risk_pct', ascending=True)
    return ranked


def make_species_chart(annual_co2_tonnes):
    top = top_species_by_risk(annual_co2_tonnes, n=12, for_chart=True)
    short = [s[:22] + '…' if len(s) > 22 else s for s in top['species']]
    risks = top['risk_pct'].tolist()
    colors = ['#ff4d4d' if r >= 70 else '#f5c518' if r >= 40 else '#00c45a' for r in risks]
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor('#0e1a10')
    ax.set_facecolor('#080f0a')
    bars = ax.barh(short, risks, color=colors, edgecolor='none', height=0.6)
    ax.set_title(
        f'Top 12 Species by Risk — your footprint {annual_co2_tonnes:.2f} t CO₂/yr',
        color='#e8f5e9', fontsize=10, pad=8,
    )
    ax.tick_params(colors='#5a7a5e', labelsize=8)
    ax.set_xlim(0, 105)
    for spine in ax.spines.values():
        spine.set_color('#1e3a22')
    ax.xaxis.grid(True, color='#1e3a22', linewidth=0.5)
    for bar, val in zip(bars, risks):
        ax.text(val + 1, bar.get_y() + bar.get_height() / 2, f'{val}%', va='center', color='#e8f5e9', fontsize=8)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=120, bbox_inches='tight', facecolor='#0e1a10')
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def make_combined_forecast_chart():
    india_hist = _india_model1_frame()
    india_train = _india_model1_train_frame()
    hist_years = india_hist['year'].values
    hist_co2 = india_hist['co2_per_capita'].values

    train_years = india_train['year'].values
    train_co2 = india_train['co2_per_capita'].values
    train_fit = co2_model.predict(train_years.reshape(-1, 1))

    last_year = int(df_india['year'].max())
    future_years = np.arange(last_year + 1, 2036)
    future_co2 = co2_model.predict(future_years.reshape(-1, 1))

    y_pred_train = co2_model.predict(train_years.reshape(-1, 1))
    rmse = np.sqrt(mean_squared_error(train_co2, y_pred_train))

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor('#0e1a10')
    ax.set_facecolor('#080f0a')

    marker_size = 3 if len(hist_years) > 50 else 6
    ax.plot(hist_years, hist_co2, 'o-', color='#00ff7f', linewidth=2,
            markersize=marker_size, label=f'Historical ({int(hist_years.min())}-{int(hist_years.max())})')
    ax.plot(train_years, train_fit, '-', color='#4dd9ff', linewidth=1.8, alpha=0.85,
            label=f'Model fit (trained {ML_START_YEAR}-{last_year})')
    forecast_years = np.arange(train_years[-1], 2036)
    forecast_co2 = co2_model.predict(forecast_years.reshape(-1, 1))
    ax.plot(forecast_years, forecast_co2, 's--', color='#f5c518', linewidth=2,
            markersize=5, label=f'Predictions (trained {ML_START_YEAR}+)')

    future_only = future_years
    future_only_co2 = future_co2
    upper_bound = [c + rmse for c in future_only_co2]
    lower_bound = [c - rmse for c in future_only_co2]
    ax.fill_between(future_only, lower_bound, upper_bound,
                    alpha=0.2, color='#f5c518', label=f'±{rmse:.2f}t (1σ)')

    pred_2030 = float(co2_model.predict([[2030]])[0])
    ax.plot(2030, pred_2030, 'o', color='#ff4d4d', markersize=12,
            markeredgecolor='white', markeredgewidth=2, label=f'2030: {pred_2030:.2f}t')

    ax.set_title(
        f'India CO₂ Forecast — Linear Regression\n'
        f'History {MODEL1_START_YEAR}–{last_year} · Forecast trained on {ML_START_YEAR}–{last_year}',
        color='#e8f5e9', fontsize=11, pad=10,
    )
    ax.set_xlabel('Year', color='#5a7a5e', fontsize=10)
    ax.set_ylabel('CO₂ per capita (tonnes)', color='#5a7a5e', fontsize=10)
    ax.tick_params(colors='#5a7a5e', labelsize=9)
    ax.legend(loc='upper left', facecolor='#0e1a10', edgecolor='#1e3a22')
    ax.grid(True, color='#1e3a22', alpha=0.3, linestyle='--')

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=120, bbox_inches='tight', facecolor='#0e1a10')
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def train_co2_model():
    india = _india_model1_train_frame()
    X = india[['year']].values
    y = india['co2_per_capita'].values
    model = LinearRegression()
    model.fit(X, y)
    return model


co2_model = train_co2_model()

# ── ML Model 2: Multi-feature CO₂ prediction (OWID) ───────
CO2_FEATURE_COLS = [
    'population',
    'coal_co2',
    'gas_co2',
    'oil_co2',
    'energy_per_capita',
    'co2_per_unit_energy',
]
CO2_FEATURE_LABELS = {
    'population': 'Population',
    'coal_co2': 'Coal CO₂ (Mt)',
    'gas_co2': 'Gas CO₂ (Mt)',
    'oil_co2': 'Oil CO₂ (Mt)',
    'energy_per_capita': 'Energy per capita',
    'co2_per_unit_energy': 'CO₂ per unit energy',
}


def _india_training_frame():
    return df_india[df_india['year'] >= ML_START_YEAR].copy()


def _project_features_for_years(years):
    """Extrapolate each OWID feature vs year for future prediction rows."""
    hist = _india_training_frame()
    rows = []
    for year in years:
        row = {'year': int(year)}
        for col in CO2_FEATURE_COLS:
            sub = hist.dropna(subset=['year', col])
            trend = LinearRegression()
            trend.fit(sub[['year']].values, sub[col].values)
            row[col] = float(trend.predict([[year]])[0])
        rows.append(row)
    return pd.DataFrame(rows)


def train_co2_multifeature_model():
    train_df = _india_training_frame().dropna(subset=CO2_FEATURE_COLS + ['co2_per_capita'])
    X = train_df[CO2_FEATURE_COLS].values
    y = train_df['co2_per_capita'].values
    model = LinearRegression()
    model.fit(X, y)
    return model, train_df


co2_multi_model, df_india_multi_train = train_co2_multifeature_model()


def multifeature_model_metrics():
    train_df = df_india_multi_train
    X = train_df[CO2_FEATURE_COLS].values
    y = train_df['co2_per_capita'].values
    y_pred = co2_multi_model.predict(X)
    return {
        'r2': round(r2_score(y, y_pred), 4),
        'mae': round(mean_absolute_error(y, y_pred), 3),
        'rmse': round(np.sqrt(mean_squared_error(y, y_pred)), 3),
        'mape': round(np.mean(np.abs((y - y_pred) / y)) * 100, 1),
        'n_samples': len(train_df),
    }


def predict_co2_multifeature(years):
    frame = _project_features_for_years(years)
    return co2_multi_model.predict(frame[CO2_FEATURE_COLS].values)


def make_multifeature_forecast_chart():
    train_df = df_india_multi_train
    hist_years = train_df['year'].values
    hist_co2 = train_df['co2_per_capita'].values
    hist_pred = co2_multi_model.predict(train_df[CO2_FEATURE_COLS].values)

    last_year = int(df_india['year'].max())
    future_years = np.arange(last_year + 1, 2036)
    future_co2 = predict_co2_multifeature(future_years)

    rmse = np.sqrt(mean_squared_error(hist_co2, hist_pred))

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor('#0e1a10')
    ax.set_facecolor('#080f0a')

    ax.plot(hist_years, hist_co2, 'o-', color='#00ff7f', linewidth=2.5,
            markersize=6, label=f'Actual ({int(hist_years.min())}-{int(hist_years.max())})')
    ax.plot(hist_years, hist_pred, '^-', color='#4dd9ff', linewidth=1.8,
            markersize=5, alpha=0.9, label='Model fit (multi-feature)')
    ax.plot(future_years, future_co2, 's--', color='#f5c518', linewidth=2,
            markersize=5, label='Predictions (features projected)')

    upper = [c + rmse for c in future_co2]
    lower = [c - rmse for c in future_co2]
    ax.fill_between(future_years, lower, upper, alpha=0.2, color='#f5c518',
                    label=f'±{rmse:.2f}t (1σ)')

    pred_2030 = float(predict_co2_multifeature([2030])[0])
    ax.plot(2030, pred_2030, 'o', color='#ff4d4d', markersize=12,
            markeredgecolor='white', markeredgewidth=2, label=f'2030: {pred_2030:.2f}t')

    feat_txt = ', '.join(CO2_FEATURE_LABELS[c] for c in CO2_FEATURE_COLS)
    ax.set_title('India CO₂ Forecast — Multi-Feature Regression (OWID)\n' + feat_txt,
                 color='#e8f5e9', fontsize=10, pad=10)
    ax.set_xlabel('Year', color='#5a7a5e', fontsize=10)
    ax.set_ylabel('CO₂ per capita (tonnes)', color='#5a7a5e', fontsize=10)
    ax.tick_params(colors='#5a7a5e', labelsize=9)
    ax.legend(loc='upper left', facecolor='#0e1a10', edgecolor='#1e3a22', fontsize=8)
    ax.grid(True, color='#1e3a22', alpha=0.3, linestyle='--')

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=120, bbox_inches='tight', facecolor='#0e1a10')
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


# ── ML Model 3: Polynomial regression on year (curved forecast) ──
YEAR_POLY_DEGREE = 2


def train_co2_year_polynomial_model():
    india = _india_training_frame()
    model = Pipeline([
        ('poly', PolynomialFeatures(degree=YEAR_POLY_DEGREE, include_bias=False)),
        ('lr', LinearRegression()),
    ])
    model.fit(india[['year']].values, india['co2_per_capita'].values)
    return model, india


co2_year_poly_model, df_india_year_poly_train = train_co2_year_polynomial_model()


def predict_co2_year_polynomial(years):
    X = np.array(years, dtype=float).reshape(-1, 1)
    return co2_year_poly_model.predict(X)


def predict_co2_polynomial(years):
    """Backward-compatible alias (Model 3: polynomial on year)."""
    return predict_co2_year_polynomial(years)


def polynomial_model_metrics():
    train_df = df_india_year_poly_train
    X = train_df[['year']].values
    y = train_df['co2_per_capita'].values
    y_pred = co2_year_poly_model.predict(X)
    return {
        'r2': round(r2_score(y, y_pred), 4),
        'mae': round(mean_absolute_error(y, y_pred), 3),
        'rmse': round(np.sqrt(mean_squared_error(y, y_pred)), 3),
        'mape': round(np.mean(np.abs((y - y_pred) / y)) * 100, 1),
        'poly_degree': YEAR_POLY_DEGREE,
    }


def make_polynomial_forecast_chart():
    train_df = df_india_year_poly_train
    hist_years = train_df['year'].values.astype(float)
    hist_co2 = train_df['co2_per_capita'].values
    hist_pred = co2_year_poly_model.predict(hist_years.reshape(-1, 1))

    last_year = int(df_india['year'].max())
    future_years = np.arange(last_year + 1, 2036, dtype=float)
    future_co2 = predict_co2_year_polynomial(future_years)

    rmse = np.sqrt(mean_squared_error(hist_co2, hist_pred))

    # Smooth curve so the polynomial shape is visible
    curve_years = np.linspace(hist_years.min(), 2035, 200)
    curve_co2 = predict_co2_year_polynomial(curve_years)

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor('#0e1a10')
    ax.set_facecolor('#080f0a')

    ax.plot(hist_years, hist_co2, 'o-', color='#00ff7f', linewidth=2.5,
            markersize=6, label=f'Actual ({int(hist_years.min())}-{int(hist_years.max())})')
    ax.plot(curve_years, curve_co2, '-', color='#cc88ff', linewidth=2.2,
            alpha=0.95, label=f'Polynomial fit (degree {YEAR_POLY_DEGREE} on year)')
    ax.plot(future_years, future_co2, 's--', color='#ff7744', linewidth=2,
            markersize=5, label='Polynomial predictions')

    upper = [c + rmse for c in future_co2]
    lower = [c - rmse for c in future_co2]
    ax.fill_between(future_years, lower, upper, alpha=0.2, color='#ff7744',
                    label=f'±{rmse:.2f}t (1σ)')

    pred_2030 = float(predict_co2_year_polynomial([2030])[0])
    ax.plot(2030, pred_2030, 'o', color='#ff4d4d', markersize=12,
            markeredgecolor='white', markeredgewidth=2, label=f'2030: {pred_2030:.2f}t')

    ax.set_title(
        f'India CO₂ Forecast — Polynomial on Year (degree {YEAR_POLY_DEGREE})\n'
        f'CO₂ = f(year, year²) · OWID India data',
        color='#e8f5e9', fontsize=10, pad=10,
    )
    ax.set_xlabel('Year', color='#5a7a5e', fontsize=10)
    ax.set_ylabel('CO₂ per capita (tonnes)', color='#5a7a5e', fontsize=10)
    ax.tick_params(colors='#5a7a5e', labelsize=9)
    ax.legend(loc='upper left', facecolor='#0e1a10', edgecolor='#1e3a22', fontsize=8)
    ax.grid(True, color='#1e3a22', alpha=0.3, linestyle='--')

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=120, bbox_inches='tight', facecolor='#0e1a10')
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def india_co2_display_rows():
    rows = []
    for _, row in df_india_recent.iterrows():
        rows.append({
            'year': int(row['year']),
            'co2_per_capita': round(float(row['co2_per_capita']), 3),
            'coal_co2_mt': round(float(row['coal_co2']), 1) if pd.notna(row.get('coal_co2')) else '—',
        })
    return rows


def state_usage_rows(limit=8):
    ranked = sorted(usage_by_state.items(), key=lambda x: x[1], reverse=True)[:limit]
    return [{'state': s, 'median_usage': round(v, 1)} for s, v in ranked]


def _build_energy_tips(ac, fridge, wash, lights, tv, fan, annual_co2, usage_high):
    """AI tips with ids — points come from user checking boxes (see frontend)."""
    tips = []

    if usage_high:
        tips.append({
            'id': 'usage_high',
            'icon': '⚠️',
            'text': f'Usage is high ({HIGH_USAGE_DAILY_KWH}+ kWh/day). Lower hours → Calculate again → then tick boxes for +2 pts each.',
            'pts': 0,
            'actionable': False,
        })

    if ac >= 6:
        tips.append({
            'id': 'reduce_ac',
            'icon': '❄️',
            'text': 'I will reduce AC by 1–2 hours per day',
            'pts': TIP_POINTS,
            'actionable': True,
        })
    if fan < 4 and ac >= 2:
        tips.append({
            'id': 'use_fan',
            'icon': '💨',
            'text': 'I will use ceiling fan more instead of AC when possible',
            'pts': TIP_POINTS,
            'actionable': True,
        })
    if lights >= 8:
        tips.append({
            'id': 'lights_off',
            'icon': '💡',
            'text': 'I will switch off lights in empty rooms',
            'pts': TIP_POINTS,
            'actionable': True,
        })
    if wash >= 1:
        tips.append({
            'id': 'wash_offpeak',
            'icon': '🫧',
            'text': 'I will run the washing machine in off-peak hours',
            'pts': TIP_POINTS,
            'actionable': True,
        })
    if tv >= 8:
        tips.append({
            'id': 'limit_tv',
            'icon': '📺',
            'text': 'I will limit TV to under 6 hours per day',
            'pts': TIP_POINTS,
            'actionable': True,
        })

    actionable = [t for t in tips if t.get('actionable')]
    if not usage_high and not actionable:
        tips.append({
            'id': 'keep_good',
            'icon': '✅',
            'text': 'Good habits — tick to confirm you will keep this usage level',
            'pts': TIP_POINTS,
            'actionable': True,
        })

    tips.append({
        'id': 'footprint_info',
        'icon': '🌱',
        'text': f'Your footprint ≈ {round(annual_co2, 2)} t CO₂/year (India avg ~2.1 t)',
        'pts': 0,
        'actionable': False,
    })

    return tips


def _energy_points_message(usage_high, usage_low):
    if usage_high:
        return 'High usage: earn 0 pts now. Reduce appliances, recalculate, then check AI actions (+2 pts each).'
    if usage_low:
        return f'Low usage: check each AI action you follow (+{TIP_POINTS} pts). Optional +{LOW_USAGE_BONUS_PTS} bonus button below.'
    return f'Check each AI action you will follow (+{TIP_POINTS} pts per box, once each).'


# ── Routes ────────────────────────────────────────────────
@app.route('/')
def index():
    states = sorted(usage_by_state.keys())
    co2_trend = make_co2_trend_chart()
    india_co2 = india_co2_display_rows()
    species_default = make_species_chart(1.9)
    pred_2030 = round(float(co2_model.predict([[2030]])[0]), 2)
    pred_2030_multi = round(float(predict_co2_multifeature([2030])[0]), 2)
    pred_2030_poly = round(float(predict_co2_year_polynomial([2030])[0]), 2)
    combined_chart = make_combined_forecast_chart()
    multifeature_chart = make_multifeature_forecast_chart()
    polynomial_chart = make_polynomial_forecast_chart()

    india_train = _india_model1_train_frame()
    X = india_train[['year']].values
    y = india_train['co2_per_capita'].values
    y_pred = co2_model.predict(X)
    last_y = int(india_train['year'].max())
    model_metrics = {
        'r2': round(r2_score(y, y_pred), 4),
        'mae': round(mean_absolute_error(y, y_pred), 3),
        'rmse': round(np.sqrt(mean_squared_error(y, y_pred)), 3),
        'mape': round(np.mean(np.abs((y - y_pred) / y)) * 100, 1),
        'history_years': f'{MODEL1_START_YEAR}–{last_y}',
        'train_years': f'{ML_START_YEAR}–{last_y}',
        'n_history': len(_india_model1_frame()),
        'n_samples': len(india_train),
    }
    model_metrics_multi = multifeature_model_metrics()
    model_metrics_poly = polynomial_model_metrics()
    co2_feature_labels = [CO2_FEATURE_LABELS[c] for c in CO2_FEATURE_COLS]

    return render_template(
        'index.html',
        states=states,
        co2_trend=co2_trend,
        species_chart=species_default,
        india_co2=india_co2,
        emission_factor=india_factor,
        pred_2030=pred_2030,
        pred_2030_multi=pred_2030_multi,
        pred_2030_poly=pred_2030_poly,
        combined_chart=combined_chart,
        multifeature_chart=multifeature_chart,
        polynomial_chart=polynomial_chart,
        model1_start_year=MODEL1_START_YEAR,
        model_metrics=model_metrics,
        model_metrics_multi=model_metrics_multi,
        model_metrics_poly=model_metrics_poly,
        co2_feature_labels=co2_feature_labels,
        uci_data=df_appliances.to_dict('records'),
        iucn_data=top_species_by_risk(1.9, n=12).to_dict('records'),
        df_tariff_rows=state_usage_rows(),
        appliance_watts=df_appliances.set_index('appliance')['watts'].to_dict(),
    )


@app.route('/calculate', methods=['POST'])
def calculate():
    data = request.json
    ac = float(data.get('ac', 0))
    fridge = float(data.get('fridge', 24))
    wash = float(data.get('wash', 0))
    lights = float(data.get('lights', 6))
    tv = float(data.get('tv', 0))
    fan = float(data.get('fan', 0))

    watts = df_appliances.set_index('appliance')['watts'].to_dict()
    hours = {
        'Air Conditioner': ac, 'Refrigerator': fridge,
        'Washing Machine': wash, 'Lights (10 LED)': lights,
        'Television': tv, 'Ceiling Fan': fan,
    }

    daily_kwh = sum(watts[a] * h / 1000 for a, h in hours.items())
    month_kwh = daily_kwh * 30
    co2_month = month_kwh * india_factor
    health = calc_planet_health(co2_month)
    annual_co2 = (co2_month * 12) / 1000

    labels = ['AC', 'Fridge', 'Wash', 'Lights', 'TV', 'Fan']
    values = [watts[a] * h / 1000 * 30 for a, h in hours.items()]
    chart = make_chart(labels, values, 'Monthly Energy (kWh) — energydata_complete.csv')
    sp_chart = make_species_chart(annual_co2)

    usage_high = daily_kwh >= HIGH_USAGE_DAILY_KWH or month_kwh >= HIGH_USAGE_MONTHLY_KWH
    usage_low = daily_kwh <= LOW_USAGE_DAILY_KWH

    tips = _build_energy_tips(
        ac=ac, fridge=fridge, wash=wash, lights=lights, tv=tv, fan=fan,
        annual_co2=annual_co2, usage_high=usage_high,
    )
    default_state = 'Karnataka' if 'Karnataka' in usage_by_state else sorted(usage_by_state.keys())[0]
    state = data.get('state') or default_state
    month_bill = calc_state_bill(month_kwh, state)

    return jsonify({
        'daily_kwh': round(daily_kwh, 2),
        'month_kwh': round(month_kwh, 2),
        'co2_month': round(co2_month, 1),
        'annual_co2': round(annual_co2, 2),
        'health': health,
        'chart': chart,
        'species_chart': sp_chart,
        'top_species': top_species_by_risk(annual_co2, n=12).to_dict('records'),
        'tips': tips,
        'usage_high': usage_high,
        'usage_low': usage_low,
        'tip_points': TIP_POINTS,
        'low_usage_bonus': LOW_USAGE_BONUS_PTS if usage_low and not usage_high else 0,
        'auto_pts': 0 if usage_high else (LOW_USAGE_BONUS_PTS if usage_low else MEDIUM_USAGE_AUTO_PTS),
        'points_message': _energy_points_message(usage_high, usage_low),
        'emission_factor': india_factor,
        'month_bill': month_bill,
    })


@app.route('/verify_bill', methods=['POST'])
def verify_bill():
    data = request.json
    prev = float(data.get('prev_units', 0))
    curr = float(data.get('curr_units', 0))
    state = data.get('state', 'Karnataka')

    prev_bill = calc_state_bill(prev, state)
    curr_bill = calc_state_bill(curr, state)
    saved_units = max(0, prev - curr)
    saved_money = max(0, prev_bill - curr_bill)
    extra_units = max(0, curr - prev)
    extra_money = max(0, curr_bill - prev_bill)
    usage_improved = curr < prev

    co2_saved = round(saved_units * india_factor, 1)
    co2_extra = round(extra_units * india_factor, 1)
    trees = max(1, round(co2_saved / 21)) if usage_improved else 0

    bonus_pts = min(100, round(saved_units * 2 + saved_money * 0.5))
    penalty_pts = min(50, round(extra_units * 2 + extra_money * 0.3))
    pts_delta = bonus_pts if usage_improved else (-penalty_pts if extra_units > 0 else 0)

    state_benchmark = usage_by_state.get(state, usage_national_median)

    fig, ax = plt.subplots(figsize=(6, 3))
    fig.patch.set_facecolor('#0e1a10')
    ax.set_facecolor('#080f0a')
    cats = ['Previous Month', 'Current Month']
    vals = [prev, curr]
    colors = ['#00c45a', '#ff4d4d'] if usage_improved else ['#ff4d4d', '#00c45a']
    bars = ax.bar(cats, vals, color=colors, width=0.4, edgecolor='none')
    ax.set_title(f'Units vs State Median ({state_benchmark:.0f}) — long_data_.csv',
                 color='#e8f5e9', fontsize=10, pad=8)
    ax.axhline(state_benchmark, color='#f5c518', linestyle='--', linewidth=1.5, label='State median usage')
    ax.tick_params(colors='#5a7a5e')
    for spine in ax.spines.values():
        spine.set_color('#1e3a22')
    ax.yaxis.grid(True, color='#1e3a22', linewidth=0.5)
    ax.legend(facecolor='#0e1a10', edgecolor='#1e3a22', labelcolor='#e8f5e9', fontsize=8)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f'{val:.0f} kWh', ha='center', color='#e8f5e9', fontsize=10, fontweight='bold')
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=120, bbox_inches='tight', facecolor='#0e1a10')
    plt.close()
    buf.seek(0)
    bill_chart = base64.b64encode(buf.read()).decode('utf-8')

    return jsonify({
        'saved_units': saved_units,
        'saved_money': saved_money,
        'extra_units': extra_units,
        'extra_money': extra_money,
        'prev_bill': prev_bill,
        'curr_bill': curr_bill,
        'co2_saved': co2_saved,
        'co2_extra': co2_extra,
        'trees': trees,
        'bonus_pts': bonus_pts,
        'penalty_pts': penalty_pts,
        'pts_delta': pts_delta,
        'usage_improved': usage_improved,
        'bill_chart': bill_chart,
        'state': state,
        'emission_factor': india_factor,
        'state_median_usage': round(state_benchmark, 1),
    })


@app.route('/datasets')
def datasets():
    return jsonify({
        'energy': df_appliances.to_dict('records'),
        'co2_india': india_co2_display_rows(),
        'species': df_species.head(20).to_dict('records'),
        'state_usage': state_usage_rows(20),
    })


if __name__ == '__main__':
    app.run(debug=True, port=5000)
