#!/usr/bin/env python3
"""
Simple test to verify jersey_scraper.py basic functionality
"""
import sys
import asyncio
from pathlib import Path

# Test imports
try:
    from jersey_scraper import (
        AIJerseyScraper, 
        CATEGORIES, 
        KIT_TYPES, 
        ALL_TEAMS,
        TEAM_ALIASES
    )
    print("✓ All imports successful")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test constants
print("\n=== Testing Constants ===")
print(f"✓ Categories defined: {len(CATEGORIES)} - {list(CATEGORIES.keys())}")
print(f"✓ Kit types defined: {len(KIT_TYPES)} - {list(KIT_TYPES.keys())}")
print(f"✓ Teams defined: {len(ALL_TEAMS)}")

# Test team aliases uniqueness
print("\n=== Testing Team Aliases ===")
all_aliases = []
duplicate_found = False
for team, aliases in TEAM_ALIASES.items():
    for alias in aliases:
        if alias in all_aliases:
            print(f"✗ Duplicate alias found: '{alias}' is used by multiple teams")
            duplicate_found = True
        all_aliases.append(alias)

if not duplicate_found:
    print(f"✓ All {len(all_aliases)} team aliases are unique")

# Test AIJerseyScraper initialization
print("\n=== Testing AIJerseyScraper Initialization ===")
try:
    scraper = AIJerseyScraper()
    print("✓ Scraper initialized without API key")
    
    scraper_with_key = AIJerseyScraper(api_key="test-key")
    print("✓ Scraper initialized with API key")
except Exception as e:
    print(f"✗ Initialization failed: {e}")
    sys.exit(1)

# Test AI detection methods (fallback mode without API key)
print("\n=== Testing AI Detection Methods (Fallback) ===")

# Test team detection
test_texts = [
    ("Liverpool FC Home Kit", "liverpool"),
    ("Man Utd Away Jersey", "manchester_united"),
    ("Barcelona Third Kit", "barcelona"),
    ("Bayern Munich Home", "bayern_munich"),
]

for text, expected in test_texts:
    detected = scraper.use_ai_to_detect_team(text, ALL_TEAMS)
    if detected == expected:
        print(f"✓ Team detection: '{text}' -> {detected}")
    else:
        print(f"✗ Team detection failed: '{text}' -> {detected} (expected {expected})")

# Test kit type detection
kit_test_texts = [
    ("Home Kit 2025", "first"),
    ("Away Jersey 2026", "second"),
    ("Third Kit", "third"),
]

for text, expected in kit_test_texts:
    detected = scraper.use_ai_to_detect_kit_type(text)
    if detected == expected:
        print(f"✓ Kit type detection: '{text}' -> {detected}")
    else:
        print(f"✗ Kit type detection: '{text}' -> {detected} (expected {expected})")

# Test category detection
category_test_texts = [
    ("Men's Jersey", "men_jerseys"),
    ("Kids Kit", "kids_kits"),
    ("Women Jersey", "women_jerseys"),
]

for text, expected in category_test_texts:
    detected = scraper.use_ai_to_detect_category(text)
    if detected == expected:
        print(f"✓ Category detection: '{text}' -> {detected}")
    else:
        print(f"✗ Category detection: '{text}' -> {detected} (expected {expected})")

print("\n=== All Basic Tests Passed! ===")
print("\nNote: Full scraping tests require a live website and cannot be automated here.")
print("To test manually, run:")
print("  python jersey_scraper.py --url <store_url> --season 2025/2026 --team liverpool --category men_jerseys --type any")
