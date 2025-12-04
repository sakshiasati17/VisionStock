# 📊 Real-Time Data Processing Guide for VisionStock

## 🔄 How Live Data Currently Works

### Current Flow:
```
User Action → Dashboard → Backend API → Database → Response → Display
```

1. **Dashboard fetches data** from backend endpoints:
   - `/api/detections` - All detection records
   - `/api/summary` - Statistics (total detections, SKUs, confidence)
   - `/api/planograms` - Expected inventory layouts

2. **Data is cached** for 30 seconds using `@st.cache_data(ttl=30)`
   - Reduces API calls
   - But means updates take up to 30 seconds to appear

3. **When you run a detection:**
   - Image uploaded → `/api/detect` endpoint
   - Backend processes image → Saves to PostgreSQL database
   - **Dashboard doesn't automatically refresh** to show new data

---

## 🚀 How to Enable Real-Time Updates

### Option 1: Auto-Refresh (Recommended - Simple)

Add auto-refresh to specific pages that need real-time data:

```python
import time

# In your page (e.g., Overview, Detection Records)
if st.checkbox("🔄 Auto-refresh (every 5 seconds)"):
    time.sleep(5)
    st.rerun()
```

**Pros:** Simple, user-controlled  
**Cons:** Uses more resources, requires user to enable

---

### Option 2: Manual Refresh Button

Add a refresh button that clears cache and reloads:

```python
if st.button("🔄 Refresh Data"):
    st.cache_data.clear()  # Clear all cached data
    st.rerun()  # Reload page
```

**Pros:** User controls when to refresh  
**Cons:** Manual action required

---

### Option 3: Reduce Cache TTL (Quick Fix)

Change cache time from 30 seconds to 5 seconds:

```python
@st.cache_data(ttl=5)  # Instead of ttl=30
def cached_api_request(endpoint: str, method: str = "GET", **kwargs):
    return api_request(endpoint, method=method, **kwargs)
```

**Pros:** Automatic, simple  
**Cons:** More API calls, slightly slower

---

### Option 4: Streamlit Auto-Refresh (Best for Demos)

Use Streamlit's built-in auto-refresh:

```python
import streamlit as st

# Add to top of page
auto_refresh = st.sidebar.checkbox("🔄 Auto-refresh", value=False)
if auto_refresh:
    refresh_interval = st.sidebar.slider("Refresh interval (seconds)", 1, 60, 5)
    time.sleep(refresh_interval)
    st.rerun()
```

---

### Option 5: WebSocket (Advanced - True Real-Time)

For true real-time updates without polling:

1. **Backend:** Add WebSocket endpoint using FastAPI WebSockets
2. **Frontend:** Use `streamlit-webrtc` or custom WebSocket client
3. **Database:** Use PostgreSQL LISTEN/NOTIFY for change events

**Pros:** True real-time, efficient  
**Cons:** Complex implementation, requires WebSocket support

---

## 📝 Implementation Steps

### Step 1: Choose Your Approach
- **For demos:** Option 4 (Auto-refresh checkbox)
- **For production:** Option 2 (Manual refresh) + Option 3 (Reduced TTL)
- **For true real-time:** Option 5 (WebSocket)

### Step 2: Add to Dashboard

Edit `dashboard/app.py` and add refresh functionality to pages that need it:
- Overview page (for summary stats)
- Detection Records page (for new detections)
- Inventory Analysis page (for planogram updates)

### Step 3: Test

1. Run a detection
2. Check if data appears automatically (or after refresh)
3. Verify database is being updated correctly

---

## 🔍 Current Data Sources

### What Data is Available:

1. **Detection Records** (`/api/detections`)
   - All detections from uploaded images
   - Filtered by SKU, shelf location, timestamp
   - Includes confidence scores, bounding boxes

2. **Summary Statistics** (`/api/summary`)
   - Total detections count
   - Unique SKUs detected
   - Average confidence score
   - Unique classes detected

3. **Planograms** (`/api/planograms`)
   - Expected product layouts
   - SKU counts per shelf
   - Product names and locations

### Database Schema:

- `detections` table: Stores all detection results
- `planograms` table: Stores expected inventory
- `model_versions` table: Stores model information
- `model_metrics` table: Stores training metrics

---

## 🛠️ Quick Implementation Example

Add this to your Overview page for auto-refresh:

```python
# At the top of Overview page section
col1, col2 = st.columns([3, 1])
with col2:
    auto_refresh = st.checkbox("🔄 Auto-refresh", value=False)
    if auto_refresh:
        refresh_seconds = st.selectbox("Interval", [5, 10, 30, 60], index=0)

# After fetching data
if auto_refresh:
    time.sleep(refresh_seconds)
    st.rerun()
```

---

## ✅ Recommended Setup for Your Demo

1. **Add auto-refresh checkbox** to Overview and Detection Records pages
2. **Reduce cache TTL** to 10 seconds (balance between real-time and performance)
3. **Add manual refresh button** for immediate updates
4. **Show last update timestamp** so users know when data was fetched

This gives you:
- ✅ Automatic updates every 10 seconds (if enabled)
- ✅ Manual refresh for immediate updates
- ✅ Efficient API usage (caching)
- ✅ Good user experience

---

## 📞 Need Help?

If you want me to implement any of these options, just let me know which approach you prefer!

