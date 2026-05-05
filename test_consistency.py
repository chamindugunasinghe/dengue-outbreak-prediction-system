import requests
import time

districts_to_test = [
    ('Colombo', 6.9272, 80.7789),
    ('Galle', 6.0535, 80.2155),
    ('Kandy', 7.2906, 80.6337),
    ('Jaffna', 9.6615, 80.7740),
]

print("Testing 7-Day Forecast Consistency")
print("=" * 60)

for name, lat, lng in districts_to_test:
    response = requests.post('http://127.0.0.1:5000/api/forecast', 
        json={'lat': lat, 'lng': lng, 'district': name},
        timeout=10
    )
    
    data = response.json()
    if data['success']:
        scores = [f['risk_score'] for f in data['forecasts']]
        levels = [f['risk_level'] for f in data['forecasts']]
        
        # Check for violations (jump of 2 or more)
        violations = []
        for i in range(len(scores)-1):
            if abs(scores[i] - scores[i+1]) >= 2:
                violations.append(f"Day {i+1}→{i+2}: {scores[i]}→{scores[i+1]}")
        
        print(f"\n{name}:")
        print(f"  Risk sequence: {' → '.join(levels)}")
        print(f"  Score sequence: {scores}")
        
        if violations:
            print(f"  ❌ VIOLATIONS: {', '.join(violations)}")
        else:
            print(f"  ✅ SMOOTH (no jumps > 1)")
    
    time.sleep(0.5)

print("\n" + "=" * 60)
print("Summary: Realistic forecasts should show gradual transitions")
print("         (0→0→1 or 1→1→1 or 2→1→1, NOT 0→2→0)")
