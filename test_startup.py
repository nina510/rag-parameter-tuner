#!/usr/bin/env python3
"""
测试应用启动脚本
用于诊断 Render 部署问题
"""
import os
import sys

# 设置环境变量（模拟 Render）
os.environ['PORT'] = '5000'
if not os.environ.get('OPENAI_API_KEY'):
    os.environ['OPENAI_API_KEY'] = 'test-key-for-startup-only'

print("=" * 50)
print("Testing application startup...")
print("=" * 50)

try:
    print("\n1. Testing imports...")
    import app
    print("   ✓ app module imported")
    
    print("\n2. Testing Flask app object...")
    flask_app = app.app
    print(f"   ✓ Flask app object: {type(flask_app)}")
    
    print("\n3. Testing routes...")
    with flask_app.test_client() as client:
        response = client.get('/api/health')
        print(f"   ✓ /api/health route: {response.status_code}")
        
        response = client.get('/')
        print(f"   ✓ / route: {response.status_code}")
    
    print("\n4. Testing Gunicorn import...")
    import gunicorn
    print("   ✓ gunicorn module available")
    
    print("\n" + "=" * 50)
    print("✓ All startup tests passed!")
    print("=" * 50)
    sys.exit(0)
    
except Exception as e:
    print(f"\n✗ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

