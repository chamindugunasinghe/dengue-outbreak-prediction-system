import requests
import json

# Test forecast for Colombo
response = requests.post('http://127.0.0.1:5000/api/forecast', 
    json={'lat': 6.9272, 'lng': 80.7789, 'district': 'Colombo'},
    timeout=10
)

data = response.json()
if data['success']:
    print('Colombo 7-Day Forecast:')
    print('-' * 50)
    for f in data['forecasts']:
        print(f"Day {f['day']}: {f['date']} - {f['risk_level']} Risk (Score: {f['risk_score']})")
    print('-' * 50)
    
    # Check for wild swings
    scores = [f['risk_score'] for f in data['forecasts']]
    swings = 0
    for i in range(len(scores)-1):
        if abs(scores[i] - scores[i+1]) >= 2:
            swings += 1
            print(f"WARNING: Big swing from Day {i+1} to Day {i+2}: {scores[i]} -> {scores[i+1]}")
    
    print(f'\nWild swings (0->2 or 2->0): {swings}')
    print(f'Risk scores: {scores}')
    
    # Test another district
    print('\n' + '='*50)
    response2 = requests.post('http://127.0.0.1:5000/api/forecast', 
        json={'lat': 6.0535, 'lng': 80.2155, 'district': 'Galle'},
        timeout=10
    )
    data2 = response2.json()
    if data2['success']:
        print('Galle 7-Day Forecast:')
        print('-' * 50)
        scores2 = [f['risk_score'] for f in data2['forecasts']]
        for f in data2['forecasts']:
            print(f"Day {f['day']}: {f['date']} - {f['risk_level']} Risk")
        print(f'Risk scores: {scores2}')
else:
    print('Error:', data.get('error'))
