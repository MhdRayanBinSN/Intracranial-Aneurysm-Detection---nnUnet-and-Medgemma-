import requests

print("Testing UPLOAD to http://127.0.0.1:8001/analyze...")
try:
    # Create dummy file content
    files = [
        ('files', ('test1.dcm', b'0'*1000, 'application/dicom')),
        ('files', ('test2.dcm', b'1'*1000, 'application/dicom'))
    ]
    
    # Send POST request
    r = requests.post('http://127.0.0.1:8001/analyze', files=files, timeout=10)
    
    print(f"Status: {r.status_code}")
    if r.status_code == 200:
        print("Response received (Truncated):")
        print(str(r.json())[:200])
        print("\n✅ UPLOAD SUCCEEDED")
    else:
        print(f"\n❌ FAILED: {r.text}")

except Exception as e:
    print(f"\n❌ EXCEPTION: {e}")
