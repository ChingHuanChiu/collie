"""
Test Runner Script

Provides convenient commands for running different test suites.
"""

import sys
import subprocess
from pathlib import Path


def run_command(cmd):
    """Run a command and return the exit code."""
    print(f"\n{'='*70}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'='*70}\n")
    
    result = subprocess.run(cmd)
    return result.returncode


def run_all_tests():
    """Run all tests."""
    return run_command(["pytest", "-v"])


def run_unit_tests():
    """Run unit tests only."""
    return run_command(["pytest", "tests/unit_tests/", "-v"])


def run_integration_tests():
    """Run integration tests only."""
    return run_command(["pytest", "tests/integration_tests/", "-v"])


def run_with_coverage():
    """Run tests with coverage report."""
    return run_command([
        "pytest",
        "--cov=collie",
        "--cov-report=html",
        "--cov-report=term-missing",
        "-v"
    ])


def run_fast_tests():
    """Run only fast tests (exclude slow marker)."""
    return run_command(["pytest", "-m", "not slow", "-v"])


def run_by_component(component):
    """Run tests for a specific component."""
    return run_command(["pytest", "-m", component, "-v"])


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Test Runner - Collie ML Pipeline Framework\n")
        print("Usage: python run_tests_new.py <command>\n")
        print("Available commands:")
        print("  all             - Run all tests")
        print("  unit            - Run unit tests only")
        print("  integration     - Run integration tests only")
        print("  coverage        - Run tests with coverage report")
        print("  fast            - Run fast tests only (exclude slow)")
        print("  <component>     - Run tests for specific component")
        print("                    (orchestrator, transformer, trainer, etc.)")
        print("\nExamples:")
        print("  python run_tests_new.py all")
        print("  python run_tests_new.py unit")
        print("  python run_tests_new.py orchestrator")
        return 1
    
    command = sys.argv[1].lower()
    
    if command == "all":
        return run_all_tests()
    elif command == "unit":
        return run_unit_tests()
    elif command == "integration":
        return run_integration_tests()
    elif command == "coverage":
        return run_with_coverage()
    elif command == "fast":
        return run_fast_tests()
    else:
        # Assume it's a component marker
        return run_by_component(command)


if __name__ == "__main__":
    sys.exit(main())
