# Ministry Dashboard - User Guide

## 🎯 Purpose
The Ministry Dashboard provides Health Ministry officials with a comprehensive overview of predicted dengue cases for ALL 23 districts in Sri Lanka for the next 7 days. This enables proactive resource planning and allocation.

## 📊 What You Get

### **National Summary (Top Cards)**
1. **Total Predicted Cases**: Sum of all predicted cases across Sri Lanka for next week
2. **High Risk Districts**: Districts requiring immediate attention (predicted 61-150+ cases)
3. **Medium Risk Districts**: Districts needing moderate monitoring (21-60 cases)
4. **Low Risk Districts**: Districts with routine surveillance only (5-20 cases)

### **District Table**
Each row shows:
- **District Name**: All 23 districts
- **Predicted Cases**: Estimated number of dengue cases for next 7 days
- **Risk Level**: Low / Medium / High (color-coded badges)
- **Trend**: ↑ Increasing / → Stable / ↓ Decreasing
- **Peak Day**: Which day of the week will have highest risk (Day 1-7)

### **Priority Highlighting**
- Top 5 districts by predicted cases are highlighted with red background
- Automatically sorted by number of cases (highest first)

## 🔧 Features

### **1. Search**
- Type district name in search box to filter results
- Helps quickly locate specific districts

### **2. Sortable Columns**
- Click any column header to sort
- Click again to reverse sort order
- Useful for analyzing by different metrics

### **3. Export to CSV**
- Click "Export CSV" button
- Downloads file: `dengue_forecast_YYYY-MM-DD.csv`
- Use in Excel, reports, or other systems
- Contains all district data for sharing with teams

### **4. Refresh**
- Click "Refresh" button to reload latest predictions
- Data auto-updates daily at 2 AM
- Manual refresh available anytime

### **5. Bilingual Support**
- Toggle EN/SI button (top right)
- All content translates to Sinhala
- Language preference saved

## 🚀 How to Access

### **Option 1: From Landing Page**
1. Go to: `http://127.0.0.1:5000/`
2. Click **"Ministry Dashboard"** button (middle button)

### **Option 2: Direct URL**
- Navigate to: `http://127.0.0.1:5000/ministry-dashboard`

## 📋 Use Cases

### **Weekly Planning Meeting**
1. Open Ministry Dashboard
2. Review total predicted cases
3. Identify high-risk districts
4. Export CSV for meeting handout
5. Allocate resources to priority districts

### **Resource Allocation**
- Check **Priority Districts** (top 5)
- Review **Peak Day** to know when resources needed most
- Monitor **Trend** to see if situation worsening
- Plan medical supplies, staff deployment accordingly

### **Monitoring & Reporting**
- Take screenshot of dashboard for reports
- Export CSV for weekly briefings
- Compare week-over-week trends
- Share with regional health offices

## 📈 Understanding Predictions

### **Case Estimates**
Our AI model predicts risk levels (Low/Medium/High), which are converted to case estimates:
- **Low Risk (0)**: 5-20 cases per week
- **Medium Risk (1)**: 21-60 cases per week
- **High Risk (2)**: 61-150+ cases per week

These are **estimates** based on:
✓ 98.60% accurate ML model
✓ Weather forecasts (temperature, rainfall, wind)
✓ Historical dengue patterns
✓ Seasonal trends (monsoon periods)
✓ Environmental conditions

### **What the Trends Mean**
- **↑ Increasing**: Risk levels rising over the week - prepare for escalation
- **→ Stable**: Consistent risk throughout week - maintain current response
- **↓ Decreasing**: Risk levels declining - situation improving

### **Peak Day**
- Shows which day (1-7) will have highest risk
- Day 1 = Tomorrow, Day 7 = 7 days from now
- Plan interventions before peak day arrives

## 🔄 Data Updates

### **Automatic Updates**
- System updates ALL districts **daily at 2:00 AM**
- Uses latest weather forecasts
- No manual intervention needed

### **Manual Update**
If you need fresh data immediately:
1. Click **"Refresh"** button on dashboard
2. Or trigger via API: `POST /api/update-now`

### **Cache System**
- Forecasts cached for 12 hours
- Improves loading speed
- Reduces API calls
- "Last Updated" timestamp shown at bottom

## 🎨 Visual Indicators

### **Color Coding**
- 🔴 **Red badges/highlights**: High risk - immediate action required
- 🟡 **Yellow badges**: Medium risk - monitor closely
- 🟢 **Green badges**: Low risk - routine surveillance

### **Icons**
- 🔼 Up arrow: Risk increasing
- ➡️ Right arrow: Risk stable  
- 🔽 Down arrow: Risk decreasing

## 💡 Best Practices

### **For Health Ministry Officials**
1. **Check daily**: Review dashboard each morning
2. **Focus on priorities**: Top 5 districts get first attention
3. **Watch trends**: Increasing trends need rapid response
4. **Plan ahead**: Use Peak Day to schedule interventions
5. **Archive data**: Export CSV weekly for records

### **For Planning Officers**
1. Export CSV at start of week
2. Share with regional coordinators
3. Cross-reference with resource inventory
4. Schedule deployments before peak days
5. Track actual cases vs predictions for accuracy

### **For Emergency Response**
1. Monitor high-risk districts continuously
2. Pre-position resources in priority areas
3. Alert teams when trend changes to increasing
4. Coordinate with meteorology for weather updates

## 📞 Technical Details

### **API Endpoint**
```
GET /api/all-districts-forecast
```

Returns JSON with:
- Summary statistics
- All 23 districts data
- Daily forecasts for each district
- Timestamp of generation

### **Export Format**
CSV columns:
1. District
2. Predicted Cases (7-Day)
3. Risk Level
4. Trend
5. Peak Day

## ⚠️ Important Notes

1. **Predictions are estimates**: Actual cases may vary based on interventions, unreported cases, etc.
2. **Weather dependent**: Forecasts based on weather predictions which can change
3. **Use as planning tool**: Supplement with ground reports and surveillance data
4. **Not diagnostic**: For planning purposes only, not clinical diagnosis
5. **Model accuracy**: 98.60% on historical data, real-world performance may differ

## 🆕 Recent Test Results

**Latest Update (March 10, 2026)**:
- Total Predicted Cases: **5,647** for next week
- High Risk Districts: **1** (Puttalam)
- Medium Risk: **19** districts
- Low Risk: **3** districts

**Top Priority Districts**:
1. Puttalam (highest cases)
2. Ampara
3. Wayamba
4. Galle
5. Kegalle

## 📱 Browser Compatibility
- Chrome/Edge: ✅ Full support
- Firefox: ✅ Full support
- Safari: ✅ Full support
- Mobile browsers: ✅ Responsive design

## 🔐 Access Control
Currently system is open access. For production:
- Add authentication for Ministry users
- Implement role-based access
- Track user actions/downloads
- Add audit logging

---

**For technical support or questions:**
Contact the system administrator or refer to the main README.md file.

**Last Updated**: March 10, 2026
**Version**: 1.0
