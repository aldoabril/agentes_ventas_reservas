from zoneinfo import ZoneInfo
try:
    tz = ZoneInfo("America/Lima")
    print(f"Timezone found: {tz}")
except Exception as e:
    print(f"Error: {e}")
