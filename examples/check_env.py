#!/usr/bin/env python3
"""
Environment Variable Checker

Quick script to verify your .env file has the correct variable names.
"""

import os
import sys

from dotenv import load_dotenv

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("=" * 80)
print("ENVIRONMENT VARIABLE CHECKER")
print("=" * 80)
print()

# Load .env file
env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
if os.path.exists(env_path):
    print(f"✅ Found .env file at: {env_path}")
    load_dotenv(env_path)
else:
    print(f"❌ No .env file found at: {env_path}")
    print("   Please create one based on .env.example")
    sys.exit(1)

print()
print("Checking environment variables...")
print()

# Check for correct variable names (what config.py expects)
required_vars = {
    "ALPACA_API_KEY": os.getenv("ALPACA_API_KEY"),
    "ALPACA_SECRET_KEY": os.getenv("ALPACA_SECRET_KEY"),
    "PAPER": os.getenv("PAPER"),
}

# Also check for old variable names
old_vars = {"API_KEY": os.getenv("API_KEY"), "API_SECRET": os.getenv("API_SECRET")}

print("Expected variables (used by config.py):")
for var, value in required_vars.items():
    if value:
        display_value = value[:8] + "..." if len(value) > 8 else value
        if var == "PAPER":
            display_value = value
        print(f"  ✅ {var:25s} = {display_value}")
    else:
        print(f"  ❌ {var:25s} = NOT SET")

print()
print("Old variable names (not used by config.py):")
for var, value in old_vars.items():
    if value:
        display_value = value[:8] + "..." if len(value) > 8 else value
        print(f"  ⚠️  {var:25s} = {display_value} (should be ALPACA_{var})")
    else:
        print(f"  ✅ {var:25s} = NOT SET")

print()
print("=" * 80)

# Check if migration is needed
if old_vars["API_KEY"] and not required_vars["ALPACA_API_KEY"]:
    print("⚠️  MIGRATION NEEDED")
    print("=" * 80)
    print()
    print("Your .env file uses old variable names. Please update it:")
    print()
    print("OLD .env format:")
    print("  API_KEY=...")
    print("  API_SECRET=...")
    print()
    print("NEW .env format (required):")
    print("  ALPACA_API_KEY=...")
    print("  ALPACA_SECRET_KEY=...")
    print("  PAPER=True")
    print()
elif required_vars["ALPACA_API_KEY"] and required_vars["ALPACA_SECRET_KEY"]:
    print("✅ ALL CHECKS PASSED")
    print("=" * 80)
    print()
    print("Your .env file is correctly configured!")
    print("You can now run the smoke test:")
    print("  python examples/smoke_test.py")
else:
    print("❌ CONFIGURATION ERROR")
    print("=" * 80)
    print()
    print("Please create a .env file with:")
    print("  ALPACA_API_KEY=your_api_key_here")
    print("  ALPACA_SECRET_KEY=your_secret_key_here")
    print("  PAPER=True")
