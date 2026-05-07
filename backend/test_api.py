import requests
import sys

try:
    print("Testing connection to http://127.0.0.1:8001/health...")
    r = requests.get("http://127.0.0.1:8001/health", timeout=5)
    print(f"Status: {r.status_code}")
    print(f"Response: {r.json()}")
    
    if r.status_code == 200:
        print("\n✅ Backend is REACHABLE via 127.0.0.1")
    else:
        print("\n❌ Backend answered with error")

except Exception as e:
    print(f"\n❌ FAILED to connect: {e}")
    sys.exit(1)
