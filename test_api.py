
import subprocess
import time
import requests
import sys
import os
import signal
import json

BASE_URL = "http://localhost:8099"

def wait_for_server():
    print("Waiting for server...")
    for _ in range(30):
        try:
            requests.get(f"{BASE_URL}/api/status")
            return True
        except requests.ConnectionError:
            time.sleep(1)
    return False

def test_api():
    # Start server
    env = os.environ.copy()
    # Add app directory to PYTHONPATH so pv_manager package can be found
    env["PYTHONPATH"] = os.path.join(os.getcwd(), "app") + os.pathsep + env.get("PYTHONPATH", "")
    
    print("Starting server...")
    proc = subprocess.Popen(
        [sys.executable, "app/main.py"], 
        env=env, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE
    )
    
    try:
        if not wait_for_server():
            print("Server failed to start")
            # Kill and read output
            proc.terminate()
            out, err = proc.communicate()
            print("STDOUT:", out.decode())
            print("STDERR:", err.decode())
            sys.exit(1)
            
        print("Server started!")
        
        # Test Control API
        print("\nTesting Control API...")
        try:
            # Enable control
            resp = requests.post(f"{BASE_URL}/api/control", json={"active": True})
            resp.raise_for_status()
            print("Enable Control:", resp.json())
            assert resp.json()["active"] is True

            # Check status
            resp = requests.get(f"{BASE_URL}/api/status")
            resp.raise_for_status()
            print("Status (Control Active):", resp.json().get("control_active"))
            assert resp.json()["control_active"] is True

            # Disable control
            resp = requests.post(f"{BASE_URL}/api/control", json={"active": False})
            resp.raise_for_status()
            print("Disable Control:", resp.json())
            assert resp.json()["active"] is False
            
        except Exception as e:
            print(f"Control API failed: {e}")
            raise

        # Test Driver API
        print("\nTesting Driver API...")
        try:
            # Get Drivers
            resp = requests.get(f"{BASE_URL}/api/drivers")
            resp.raise_for_status()
            drivers = resp.json()["drivers"]
            print("Drivers:", json.dumps(drivers, indent=2))
            assert len(drivers) > 0
            goodwe = next((d for d in drivers if d["id"] == "goodwe"), None)
            assert goodwe is not None
            
            # Save Driver Config
            config_payload = {
                "driver_id": "goodwe",
                "entity_map": {
                    "operation_mode": "select.goodwe_mode",
                    "eco_mode_power": "number.goodwe_power",
                    "eco_mode_soc": "number.goodwe_soc",
                    "active_power": "sensor.goodwe_active_power"
                },
                "config": {
                    "max_power_kw": 10.0
                }
            }
            resp = requests.post(f"{BASE_URL}/api/settings/inverter-driver", json=config_payload)
            resp.raise_for_status()
            print("Save Driver Config:", resp.json())
            
            # Get Driver Config
            resp = requests.get(f"{BASE_URL}/api/settings/inverter-driver")
            resp.raise_for_status()
            saved_config = resp.json()
            print("Saved Config:", json.dumps(saved_config, indent=2))
            assert saved_config["driver_id"] == "goodwe"
            assert saved_config["config"]["max_power_kw"] == 10.0
            
        except Exception as e:
            print(f"Driver API failed: {e}")
            raise
            
        print("\nAll tests passed!")
        
    except Exception as e:
        print(f"\nTest failed: {e}")
        sys.exit(1)
    finally:
        print("Stopping server...")
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()

if __name__ == "__main__":
    test_api()
