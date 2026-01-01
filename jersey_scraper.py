#!/usr/bin/env python3
"""
AI-powered Football Jersey Scraper

This tool scrapes football jerseys from online stores using AI to intelligently
detect team names and kit types.

Usage:
    python jersey_scraper.py --url <store_url> --season <season> [options]

Examples:
    python jersey_scraper.py --url https://www.minejerseys.ru --season 2025/2026 --team liverpool --category men_jerseys --type first
    python jersey_scraper.py --url https://www.minejerseys.ru --season 2025/2026 --team liverpool,barcelona --category any --type any
"""

import argparse
import asyncio
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Set
from urllib.parse import urljoin, urlparse

try:
    from playwright.async_api import async_playwright, Page, Browser
    import anthropic
except ImportError:
    print("Error: Required packages not installed.")
    print("Please run: pip install playwright anthropic")
    print("Then run: playwright install chromium")
    sys.exit(1)

# Constants
CATEGORIES = {
    "men_jerseys": ["men's", "mens", "גברים", "חולצות גברים", "male"],
    "men_long_jerseys": ["long sleeve", "ארוכות", "חולצות גברים ארוכות", "long"],
    "women_jerseys": ["women", "women's", "womens", "נשים", "חליפות נשים", "ladies", "female"],
    "kids_kits": ["kids", "children", "ילדים", "חליפות ילדים", "youth", "junior"],
    "shorts": ["shorts", "מכנסיים", "pants"]
}

KIT_TYPES = {
    "first": ["first", "home", "בית", "ראשי"],
    "second": ["second", "away", "חוץ", "שני"],
    "third": ["third", "שלישי", "alternate"]
}

ALL_TEAMS = [
    "liverpool", "manchester_united", "manchester_city", "arsenal", "chelsea",
    "tottenham", "barcelona", "real_madrid", "atletico_madrid", "juventus",
    "milan", "inter_milan", "bayern_munich", "borussia_dortmund", "psg"
]

TEAM_ALIASES = {
    "liverpool": ["liverpool", "lfc", "ליברפול"],
    "manchester_united": ["manchester united", "man utd", "man united", "mufc", "מנצ'סטר יונייטד"],
    "manchester_city": ["manchester city", "man city", "mcfc", "מנצ'סטר סיטי"],
    "arsenal": ["arsenal", "afc", "ארסנל"],
    "chelsea": ["chelsea", "cfc", "צ'לסי"],
    "tottenham": ["tottenham", "spurs", "thfc", "טוטנהאם"],
    "barcelona": ["barcelona", "barca", "fc barcelona", "ברצלונה"],
    "real_madrid": ["real madrid", "madrid", "rm", "ריאל מדריד"],
    "atletico_madrid": ["atletico madrid", "atletico", "atm", "אתלטיקו מדריד"],
    "juventus": ["juventus", "juve", "יובנטוס"],
    "milan": ["milan", "ac milan", "acm", "מילאן"],
    "inter_milan": ["inter milan", "inter", "אינטר"],
    "bayern_munich": ["bayern munich", "bayern", "fc bayern", "באיירן"],
    "borussia_dortmund": ["borussia dortmund", "dortmund", "bvb", "דורטמונד"],
    "psg": ["psg", "paris saint germain", "paris", "פריז"]
}


class AIJerseyScraper:
    """AI-powered jersey scraper that uses Claude to understand product information."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the scraper with optional Anthropic API key."""
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if self.api_key:
            self.client = anthropic.Anthropic(api_key=self.api_key)
        else:
            self.client = None
            print("Warning: No ANTHROPIC_API_KEY found. AI features will be limited.")
        
        self.playwright = None
        self.browser: Optional[Browser] = None
        self.context = None
        self.page: Optional[Page] = None
    
    async def start_browser(self):
        """Start the browser instance."""
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(headless=True)
        self.context = await self.browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        )
        self.page = await self.context.new_page()
    
    async def close_browser(self):
        """Close the browser instance."""
        if self.context:
            await self.context.close()
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()
    
    def use_ai_to_detect_team(self, text: str, teams: List[str]) -> Optional[str]:
        """Use AI to detect which team a text refers to."""
        if not self.client:
            # Fallback to simple matching
            text_lower = text.lower()
            for team in teams:
                for alias in TEAM_ALIASES.get(team, [team]):
                    if alias.lower() in text_lower:
                        return team
            return None
        
        try:
            team_list = ", ".join(teams)
            message = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=100,
                messages=[{
                    "role": "user",
                    "content": f"""Given this text: "{text}"
                    
Which of these football teams is it referring to? {team_list}

Respond with ONLY the exact team name from the list, or "none" if no match.
Examples: "Man Utd Home Kit" -> manchester_united, "Liverpool Away" -> liverpool"""
                }]
            )
            result = message.content[0].text.strip().lower()
            if result in teams:
                return result
            # Try fuzzy match
            for team in teams:
                if team.replace("_", " ") in result or result in team:
                    return team
            return None
        except Exception as e:
            print(f"AI detection error: {e}")
            # Fallback to simple matching
            text_lower = text.lower()
            for team in teams:
                for alias in TEAM_ALIASES.get(team, [team]):
                    if alias.lower() in text_lower:
                        return team
            return None
    
    def use_ai_to_detect_kit_type(self, text: str) -> Optional[str]:
        """Use AI to detect which kit type a text refers to."""
        if not self.client:
            # Fallback to simple matching
            text_lower = text.lower()
            for kit_type, keywords in KIT_TYPES.items():
                for keyword in keywords:
                    if keyword.lower() in text_lower:
                        return kit_type
            return None
        
        try:
            message = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=50,
                messages=[{
                    "role": "user",
                    "content": f"""Given this text: "{text}"
                    
Is this a first/home kit, second/away kit, or third kit?

Respond with ONLY one word: "first", "second", "third", or "unknown".
Examples: "Home Kit 2025" -> first, "Away Jersey" -> second, "Third Kit" -> third"""
                }]
            )
            result = message.content[0].text.strip().lower()
            if result in ["first", "second", "third"]:
                return result
            return None
        except Exception as e:
            print(f"AI detection error: {e}")
            # Fallback to simple matching
            text_lower = text.lower()
            for kit_type, keywords in KIT_TYPES.items():
                for keyword in keywords:
                    if keyword.lower() in text_lower:
                        return kit_type
            return None
    
    def use_ai_to_detect_category(self, text: str) -> Optional[str]:
        """Use AI to detect which category a text refers to."""
        if not self.client:
            # Fallback to simple matching - check more specific categories first
            text_lower = text.lower()
            
            # Priority order: more specific categories first
            priority_order = ["women_jerseys", "kids_kits", "men_long_jerseys", "shorts", "men_jerseys"]
            
            for category in priority_order:
                keywords = CATEGORIES[category]
                for keyword in keywords:
                    if keyword.lower() in text_lower:
                        return category
            return None
        
        try:
            message = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=50,
                messages=[{
                    "role": "user",
                    "content": f"""Given this text: "{text}"
                    
Which category does this belong to: men_jerseys, men_long_jerseys, women_jerseys, kids_kits, or shorts?

Respond with ONLY the category name or "unknown".
Examples: "Men's Jersey" -> men_jerseys, "Kids Kit" -> kids_kits, "Women Jersey" -> women_jerseys"""
                }]
            )
            result = message.content[0].text.strip().lower()
            if result in CATEGORIES.keys():
                return result
            return None
        except Exception as e:
            print(f"AI detection error: {e}")
            # Fallback to simple matching - check more specific categories first
            text_lower = text.lower()
            
            # Priority order: more specific categories first
            priority_order = ["women_jerseys", "kids_kits", "men_long_jerseys", "shorts", "men_jerseys"]
            
            for category in priority_order:
                keywords = CATEGORIES[category]
                for keyword in keywords:
                    if keyword.lower() in text_lower:
                        return category
            return None
    
    async def find_team_links(self, base_url: str, teams: List[str]) -> Dict[str, str]:
        """Find team pages on the website."""
        print(f"Scanning {base_url} for teams: {', '.join(teams)}")
        await self.page.goto(base_url, wait_until="networkidle", timeout=60000)
        
        # Wait for page to load
        await self.page.wait_for_timeout(2000)
        
        # Get all links on the page
        links = await self.page.query_selector_all("a")
        
        team_links = {}
        for link in links:
            try:
                href = await link.get_attribute("href")
                text = await link.inner_text()
                
                if not href or not text:
                    continue
                
                # Make absolute URL
                full_url = urljoin(base_url, href)
                
                # Use AI to detect team
                detected_team = self.use_ai_to_detect_team(text, teams)
                if detected_team and detected_team not in team_links:
                    team_links[detected_team] = full_url
                    print(f"Found {detected_team}: {full_url}")
            except Exception as e:
                continue
        
        return team_links
    
    async def scrape_team_page(self, url: str, team: str, season: str, 
                                categories: List[str], kit_types: List[str]) -> List[Dict]:
        """Scrape jerseys from a team page."""
        print(f"\nScraping {team} page: {url}")
        
        try:
            await self.page.goto(url, wait_until="networkidle", timeout=60000)
            await self.page.wait_for_timeout(2000)
        except Exception as e:
            print(f"Error loading page: {e}")
            return []
        
        # Get all product links
        product_links = []
        links = await self.page.query_selector_all("a")
        
        for link in links:
            try:
                href = await link.get_attribute("href")
                text = await link.inner_text()
                
                if not href:
                    continue
                
                # Check if it's a product link
                if not text or len(text.strip()) < 3:
                    continue
                
                # Make absolute URL
                full_url = urljoin(url, href)
                
                # Filter by season if specified
                if season and season not in text and season not in full_url:
                    continue
                
                # Detect category and kit type
                detected_category = self.use_ai_to_detect_category(text)
                detected_kit_type = self.use_ai_to_detect_kit_type(text)
                
                # Check if matches our filters
                if categories and detected_category not in categories:
                    continue
                
                if kit_types and detected_kit_type not in kit_types:
                    continue
                
                product_links.append({
                    "url": full_url,
                    "title": text.strip(),
                    "team": team,
                    "category": detected_category,
                    "kit_type": detected_kit_type
                })
                
            except Exception as e:
                continue
        
        print(f"Found {len(product_links)} matching products for {team}")
        return product_links
    
    async def scrape_product_images(self, product: Dict) -> List[str]:
        """Scrape all images from a product page."""
        url = product["url"]
        print(f"  Scraping images from: {product['title'][:50]}...")
        
        try:
            await self.page.goto(url, wait_until="networkidle", timeout=60000)
            await self.page.wait_for_timeout(2000)
        except Exception as e:
            print(f"  Error loading product page: {e}")
            return []
        
        # Find all product images
        image_urls = []
        images = await self.page.query_selector_all("img")
        
        for img in images:
            try:
                src = await img.get_attribute("src")
                if not src:
                    continue
                
                # Make absolute URL
                full_url = urljoin(url, src)
                
                # Filter out small images (likely icons/thumbnails)
                # and non-product images
                if any(skip in full_url.lower() for skip in ["logo", "icon", "banner", "social"]):
                    continue
                
                # Check image size if possible
                try:
                    width = await img.get_attribute("width")
                    height = await img.get_attribute("height")
                    if width and height:
                        if int(width) < 200 or int(height) < 200:
                            continue
                except:
                    pass
                
                if full_url not in image_urls:
                    image_urls.append(full_url)
            except Exception as e:
                continue
        
        print(f"  Found {len(image_urls)} images")
        return image_urls
    
    async def download_image(self, url: str, save_path: Path):
        """Download an image from URL."""
        try:
            response = await self.page.request.get(url)
            if response.ok:
                content = await response.body()
                save_path.parent.mkdir(parents=True, exist_ok=True)
                with open(save_path, "wb") as f:
                    f.write(content)
                return True
            else:
                print(f"  Failed to download: {url} (status: {response.status})")
                return False
        except Exception as e:
            print(f"  Error downloading {url}: {e}")
            return False
    
    async def scrape(self, store_url: str, season: str, teams: List[str],
                     categories: List[str], kit_types: List[str], output_dir: Path):
        """Main scraping method."""
        print(f"\n{'='*60}")
        print(f"AI Jersey Scraper")
        print(f"{'='*60}")
        print(f"Store: {store_url}")
        print(f"Season: {season}")
        print(f"Teams: {', '.join(teams)}")
        print(f"Categories: {', '.join(categories)}")
        print(f"Kit Types: {', '.join(kit_types)}")
        print(f"Output: {output_dir}")
        print(f"{'='*60}\n")
        
        await self.start_browser()
        
        try:
            # Find team links
            team_links = await self.find_team_links(store_url, teams)
            
            if not team_links:
                print("\nNo team pages found. The website might have a different structure.")
                print("Please check the URL and try again.")
                return
            
            # Scrape each team
            all_products = []
            for team, team_url in team_links.items():
                products = await self.scrape_team_page(
                    team_url, team, season, categories, kit_types
                )
                all_products.extend(products)
            
            if not all_products:
                print("\nNo matching products found.")
                return
            
            print(f"\n{'='*60}")
            print(f"Found {len(all_products)} total products. Downloading images...")
            print(f"{'='*60}\n")
            
            # Download images for each product
            for product in all_products:
                image_urls = await self.scrape_product_images(product)
                
                if not image_urls:
                    continue
                
                # Prepare save directory
                category = product["category"] or "unknown"
                team = product["team"]
                kit_type = product["kit_type"] or "unknown"
                
                save_dir = output_dir / category / team
                
                # Download images
                for idx, img_url in enumerate(image_urls, start=1):
                    extension = ".jpg" if ".jpg" in img_url else ".png"
                    if idx == 1:
                        save_path = save_dir / f"{kit_type}{extension}"
                    else:
                        save_path = save_dir / f"{kit_type}_{idx}{extension}"
                    
                    success = await self.download_image(img_url, save_path)
                    if success:
                        print(f"  ✓ Saved: {save_path}")
            
            print(f"\n{'='*60}")
            print(f"Scraping complete! Images saved to: {output_dir}")
            print(f"{'='*60}\n")
            
        finally:
            await self.close_browser()


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="AI-powered Football Jersey Scraper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Scrape Liverpool first kit men jerseys
  %(prog)s --url https://www.minejerseys.ru --season 2025/2026 \\
    --team liverpool --category men_jerseys --type first
  
  # Scrape multiple teams, all categories and types
  %(prog)s --url https://www.minejerseys.ru --season 2025/2026 \\
    --team liverpool,barcelona,real_madrid
  
  # Scrape all supported teams
  %(prog)s --url https://www.minejerseys.ru --season 2025/2026 --team any
        """
    )
    
    parser.add_argument(
        "--url",
        required=True,
        help="Store URL to scrape"
    )
    
    parser.add_argument(
        "--season",
        required=True,
        help="Season to filter (e.g., 2025/2026)"
    )
    
    parser.add_argument(
        "--team",
        default="any",
        help="Team name(s) - comma-separated or 'any' for all supported teams"
    )
    
    parser.add_argument(
        "--category",
        default="any",
        help="Category - one of: men_jerseys, men_long_jerseys, women_jerseys, kids_kits, shorts, or 'any'"
    )
    
    parser.add_argument(
        "--type",
        default="any",
        help="Kit type - one of: first, second, third, or 'any'"
    )
    
    parser.add_argument(
        "--output",
        default="./images",
        help="Output directory for images (default: ./images)"
    )
    
    parser.add_argument(
        "--api-key",
        help="Anthropic API key (can also set ANTHROPIC_API_KEY env var)"
    )
    
    return parser.parse_args()


async def main():
    """Main entry point."""
    args = parse_arguments()
    
    # Parse teams
    if not args.team or args.team.lower() == "any":
        teams = ALL_TEAMS
    else:
        teams = [t.strip().lower().replace(" ", "_") for t in args.team.split(",")]
    
    # Parse categories
    if not args.category or args.category.lower() == "any":
        categories = list(CATEGORIES.keys())
    else:
        categories = [c.strip().lower() for c in args.category.split(",")]
        # Validate categories
        invalid = [c for c in categories if c not in CATEGORIES]
        if invalid:
            print(f"Error: Invalid categories: {', '.join(invalid)}")
            print(f"Valid categories: {', '.join(CATEGORIES.keys())}")
            return
    
    # Parse kit types
    if not args.type or args.type.lower() == "any":
        kit_types = list(KIT_TYPES.keys())
    else:
        kit_types = [t.strip().lower() for t in args.type.split(",")]
        # Validate kit types
        invalid = [t for t in kit_types if t not in KIT_TYPES]
        if invalid:
            print(f"Error: Invalid kit types: {', '.join(invalid)}")
            print(f"Valid types: {', '.join(KIT_TYPES.keys())}")
            return
    
    # Create output directory
    output_dir = Path(args.output)
    
    # Create scraper and run
    scraper = AIJerseyScraper(api_key=args.api_key)
    await scraper.scrape(args.url, args.season, teams, categories, kit_types, output_dir)


if __name__ == "__main__":
    asyncio.run(main())
