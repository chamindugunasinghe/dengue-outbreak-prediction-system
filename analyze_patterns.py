import csv
from collections import defaultdict

print("Loading historical data...")
rows = list(csv.DictReader(open('data/dengue_data.csv')))

# Group by district, month, risk level
by_district_month_risk = defaultdict(list)
for r in rows:
    district = r.get('District')
    month = r.get('Month')
    cases_str = r.get('Number_of_Cases', '')
    
    if district and month and cases_str.isdigit():
        cases = int(cases_str)
        risk = 0 if cases <= 3 else (1 if cases <= 9 else 2)
        by_district_month_risk[(district, int(month), risk)].append(cases)

print("\n=== MARCH HIGH-RISK HISTORICAL PATTERNS ===")
print("(What we should predict for high-risk districts in March)")
for district in ['Colombo', 'Gampaha', 'Kandy', 'Galle', 'Jaffna', 'Ampara', 'Puttalam']:
    key = (district, 3, 2)  # March, High risk
    if key in by_district_month_risk:
        values = by_district_month_risk[key]
        avg = sum(values) / len(values)
        print(f"{district:15} {avg:6.1f} cases/week  (from {len(values)} historical weeks)")

print("\n=== CURRENT WEEK PATTERN (Week 10-11, March) ===")
week_data = defaultdict(list)
for r in rows:
    if r.get('Month') == '3' and r.get('Week') in ['10', '11']:
        district = r.get('District')
        cases_str = r.get('Number_of_Cases', '')
        if district and cases_str.isdigit():
            week_data[district].append(int(cases_str))

for district in sorted(week_data.keys())[:10]:
    values = week_data[district]
    if values:
        avg = sum(values) / len(values)
        print(f"{district:15} {avg:6.1f} cases/week")
