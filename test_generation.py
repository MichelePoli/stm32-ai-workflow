import subprocess
import os

cmd = ["xvfb-run", "-a", "/home/mrusso/STM32CubeMX/STM32CubeMX", "-q", "/home/mrusso/stm32-ai-workflow/test_cubemx.script"]
print(f"Running: {' '.join(cmd)}")
try:
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    print(f"Exit code: {res.returncode}")
    print(f"STDOUT: {res.stdout}")
    print(f"STDERR: {res.stderr}")
except subprocess.TimeoutExpired:
    print("TIMEOUT after 300s")
except Exception as e:
    print(f"Error: {e}")
