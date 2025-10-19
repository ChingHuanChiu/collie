#!/usr/bin/env python3
"""
Easy test runner for the collie ML pipeline framework.

Quick usage examples:
    python tests/run_tests.py                    # Run all tests
    python tests/run_tests.py fast               # Run fast tests only
    python tests/run_tests.py models             # Test models only
    python tests/run_tests.py coverage           # Run with coverage report
    python tests/run_tests.py integration        # Run integration tests
    python tests/run_tests.py debug TestClass    # Debug specific test class
"""

import sys
import subprocess
import os
from pathlib import Path


class Colors:
    """ANSI color codes for pretty output."""
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    END = '\033[0m'


def print_header(text):
    """Print a colorful header."""
    print(f"\n{Colors.BLUE}{Colors.BOLD}{'='*60}{Colors.END}")
    print(f"{Colors.BLUE}{Colors.BOLD} {text:^56} {Colors.END}")
    print(f"{Colors.BLUE}{Colors.BOLD}{'='*60}{Colors.END}\n")


def print_success(text):
    """Print success message."""
    print(f"{Colors.GREEN}{Colors.BOLD}✅ {text}{Colors.END}")


def print_error(text):
    """Print error message."""
    print(f"{Colors.RED}{Colors.BOLD}❌ {text}{Colors.END}")


def print_info(text):
    """Print info message."""
    print(f"{Colors.YELLOW}ℹ️  {text}{Colors.END}")


def run_pytest(args, description):
    """Run pytest with given arguments."""
    # Change to project root directory
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    cmd = ["python", "-m", "pytest"] + args
    
    print_header(f"Running: {description}")
    print_info(f"Command: {' '.join(cmd)}")
    print_info(f"Working directory: {project_root}")
    
    try:
        result = subprocess.run(cmd, check=False)
        if result.returncode == 0:
            print_success(f"{description} completed successfully!")
            return True
        else:
            print_error(f"{description} failed!")
            return False
    except KeyboardInterrupt:
        print_error("Tests interrupted by user")
        return False
    except Exception as e:
        print_error(f"Error running tests: {e}")
        return False


def main():
    """Main test runner with simple commands."""
    if len(sys.argv) < 2:
        command = "all"
    else:
        command = sys.argv[1].lower()
    
    # Get additional arguments
    extra_args = sys.argv[2:] if len(sys.argv) > 2 else []
    
    # Define test configurations
    configs = {
        "all": {
            "args": ["tests/", "-v"],
            "desc": "All Tests"
        },
        "fast": {
            "args": ["tests/", "-v", "-m", "not slow"],
            "desc": "Fast Tests (excluding slow tests)"
        },
        "unit": {
            "args": ["tests/", "-v", "-m", "unit"],
            "desc": "Unit Tests Only"
        },
        "integration": {
            "args": ["tests/", "-v", "-m", "integration"],
            "desc": "Integration Tests Only"
        },
        "coverage": {
            "args": ["tests/", "-v", "--cov=collie", "--cov-report=html", "--cov-report=term-missing"],
            "desc": "Tests with Coverage Report"
        },
        "models": {
            "args": ["tests/test_models.py", "tests/core/test_*.py", "-v"],
            "desc": "Core Models and Components"
        },
        "transformer": {
            "args": ["tests/core/test_transformer.py", "-v"],
            "desc": "Transformer Component Tests"
        },
        "trainer": {
            "args": ["tests/core/test_trainer.py", "-v"],
            "desc": "Trainer Component Tests"
        },
        "evaluator": {
            "args": ["tests/core/test_evaluator.py", "-v"],
            "desc": "Evaluator Component Tests"
        },
        "pusher": {
            "args": ["tests/core/test_pusher.py", "-v"],
            "desc": "Pusher Component Tests"
        },
        "orchestrator": {
            "args": ["tests/core/test_orchestrator.py", "-v"],
            "desc": "Orchestrator Tests"
        },
        "events": {
            "args": ["tests/test_event.py", "-v"],
            "desc": "Event System Tests"
        },
        "debug": {
            "args": ["tests/", "-v", "-s", "--tb=long"],
            "desc": "Debug Mode (with print statements and detailed errors)"
        },
        "smoke": {
            "args": ["tests/", "-v", "-m", "smoke"],
            "desc": "Smoke Tests"
        },
        "parallel": {
            "args": ["tests/", "-v", "-n", "auto"],
            "desc": "Parallel Test Execution"
        }
    }
    
    # Handle special cases
    if command == "help" or command == "--help" or command == "-h":
        print_header("Collie Test Runner - Available Commands")
        print(f"{Colors.BOLD}Usage:{Colors.END} python tests/run_tests.py [command] [extra_args]\n")
        
        print(f"{Colors.BOLD}Quick Commands:{Colors.END}")
        for cmd, config in configs.items():
            print(f"  {Colors.GREEN}{cmd:12}{Colors.END} - {config['desc']}")
        
        print(f"\n{Colors.BOLD}Examples:{Colors.END}")
        print("  python tests/run_tests.py                    # Run all tests")
        print("  python tests/run_tests.py fast               # Run fast tests only") 
        print("  python tests/run_tests.py coverage           # Run with coverage")
        print("  python tests/run_tests.py debug TestTrainer  # Debug specific test")
        print("  python tests/run_tests.py models -k payload  # Test models with 'payload' in name")
        
        print(f"\n{Colors.BOLD}Pytest Arguments:{Colors.END}")
        print("  You can pass any pytest arguments after the command:")
        print("  python tests/run_tests.py all -k test_success --tb=short")
        return 0
    
    # Special handling for debug mode
    if command == "debug" and extra_args:
        configs["debug"]["args"].extend(["-k", " ".join(extra_args)])
        configs["debug"]["desc"] = f"Debug Mode - Testing: {' '.join(extra_args)}"
    
    # Check if command exists
    if command not in configs:
        print_error(f"Unknown command: {command}")
        print_info("Run 'python tests/run_tests.py help' to see available commands")
        return 1
    
    # Get configuration
    config = configs[command]
    test_args = config["args"] + extra_args
    
    # Run the tests
    success = run_pytest(test_args, config["desc"])
    
    # Print final status
    if success:
        print_success("All tests completed successfully!")
        if command == "coverage":
            print_info("Coverage report generated in htmlcov/index.html")
        return 0
    else:
        print_error("Some tests failed!")
        print_info("Run with 'debug' command for more detailed output")
        return 1


if __name__ == "__main__":
    sys.exit(main())
