import requests

print("Testing Frontend Proxy: http://localhost:5174/api/health")
try:
    # We are testing if the Frontend Server (5174) proxies to Backend (8001)
    r = requests.get("http://localhost:5174/api/health", timeout=5)
    print(f"Status: {r.status_code}")
    print(f"Response: {r.text}")
    
    if r.status_code == 200:
        print("\n✅ Proxy is WORKING")
    else:
        print("\n❌ Proxy Failed")

except Exception as e:
    print(f"\n❌ FAILED to connect: {e}")
