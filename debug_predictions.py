import sqlite3
import json
import sys
sys.path.insert(0, 'src')

# Check cached forecasts
print("=== CACHED FORECAST RISK LEVELS ===")
conn = sqlite3.connect('src/forecast_cache.db')
c = conn.cursor()
c.execute('SELECT district, forecast_data FROM forecasts')
results = c.fetchall()

if not results:
    print("No cached forecasts found! Run /api/update-now first")
else:
    for dist, data in results[:5]:
        forecasts = json.loads(data)
        risks = [f['risk_score'] for f in forecasts]
        risk_names = [f['risk_level'] for f in forecasts]
        dominant = max(set(risks), key=risks.count)
        print(f"{dist:15} Risks: {risks} -> Dominant: {dominant} ({['Low','Med','High'][dominant]})")

conn.close()

# Check if CASE_PROFILES loaded correctly
print("\n=== CASE PROFILES LOADED ===")
try:
    from app import CASE_PROFILES
    print(f"District encoding entries: {len(CASE_PROFILES.get('district_encoding', {}))}")
    print(f"Global risk weekly: {CASE_PROFILES.get('global_risk_weekly', {})}")
    
    # Check Colombo specifically
    colombo_high = CASE_PROFILES.get('district_month_risk_weekly', {}).get(('Colombo', 3, 2))
    print(f"\nColombo March High-Risk: {colombo_high} cases/week")
    
    gampaha_high = CASE_PROFILES.get('district_month_risk_weekly', {}).get(('Gampaha', 3, 2))
    print(f"Gampaha March High-Risk: {gampaha_high} cases/week")
except Exception as e:
    print(f"Error loading profiles: {e}")
