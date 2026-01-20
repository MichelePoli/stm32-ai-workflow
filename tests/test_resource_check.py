
print("DEBUG: Test script started")
import unittest
import os
import shutil
from dataclasses import dataclass
import sys

try:
    from src.assistant.workflow2_ai import check_resource_constraints, resource_check_routing, get_mcu_limits
    print("DEBUG: Imports successful")
except Exception as e:
    print(f"DEBUG: Import failed: {e}")
    # Don't exit, let unittest define empty tests or something

# Mock MasterState
@dataclass
class MockState:
    analyze_success: bool = True
    analyze_report_dir: str = ""
    target: str = "stm32f401"
    compression: str = "medium"
    resource_check_result: str = ""
    ai_error_message: str = ""
    ram_usage: int = 0
    flash_usage: int = 0
    model_discovery_method: str = "search"
    search_iterations: int = 0

class TestResourceConstraints(unittest.TestCase):

    def setUp(self):
        self.test_dir = "./test_reports"
        os.makedirs(self.test_dir, exist_ok=True)
    
    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def create_report(self, ram_bytes, flash_bytes):
        content = f"""
        Summary of analysis:
        activations size   : {ram_bytes} bytes
        weights size       : {flash_bytes} bytes
        macc               : 1000000
        """
        with open(os.path.join(self.test_dir, "network_analyze_report.txt"), "w") as f:
            f.write(content)

    def test_limits_f4(self):
        f, r = get_mcu_limits("stm32f401")
        self.assertEqual(f, 256*1024)
        self.assertEqual(r, 64*1024)

    def test_ok_scenario(self):
        # F401 limits: 256KB (262144), 64KB (65536)
        # Usage: 100KB Flash, 30KB RAM -> OK
        self.create_report(30*1024, 100*1024)
        
        state = MockState(analyze_report_dir=self.test_dir, target="stm32f401")
        state = check_resource_constraints(state, {})
        
        self.assertEqual(state.resource_check_result, "ok")
        route = resource_check_routing(state)
        self.assertEqual(route, "run_validate")

    def test_warning_scenario(self):
        # Usage: 100KB Flash, 128KB RAM (2x limit) -> Warning
        self.create_report(128*1024, 100*1024)
        
        state = MockState(analyze_report_dir=self.test_dir, target="stm32f401")
        # Node executes: should update result AND compression
        state = check_resource_constraints(state, {})
        
        self.assertEqual(state.resource_check_result, "warning")
        self.assertEqual(state.compression, "high") # Changed in Node now!
        
        # Test routing: simple string return
        route = resource_check_routing(state)
        self.assertEqual(route, "run_generate")

    def test_critical_scenario(self):
        # Usage: 4MB Flash (16x limit) -> Critical
        self.create_report(30*1024, 4*1024*1024)
        
        state = MockState(analyze_report_dir=self.test_dir, target="stm32f401")
        # Node executes: should update result AND reset search
        state = check_resource_constraints(state, {})
        
        self.assertEqual(state.resource_check_result, "critical")
        self.assertEqual(state.model_discovery_method, "search")
        self.assertEqual(state.search_iterations, 0)
        
        # Test routing: simple string return
        route = resource_check_routing(state)
        self.assertEqual(route, "choose_predefined_taskbased_model")

if __name__ == '__main__':
    from unittest.mock import patch
    with patch('src.assistant.workflow2_ai.interrupt', return_value=""):
        # Run manually
        suite = unittest.TestLoader().loadTestsFromTestCase(TestResourceConstraints)
        unittest.TextTestRunner(verbosity=2).run(suite)
