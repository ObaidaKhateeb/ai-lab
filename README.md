# AI Football Jersey Scraper

An AI-powered tool that scrapes football jerseys from online stores using intelligent team and kit type detection.

## Features

- **AI-Powered Detection**: Uses Claude AI to intelligently detect:
  - Team names (e.g., "Man Utd" → Manchester United)
  - Kit types (e.g., "Home Kit" → first kit)
  - Categories (Men's, Women's, Kids, etc.)

- **Flexible Filtering**:
  - Multiple teams (comma-separated or "any" for all supported teams)
  - Multiple categories (men_jerseys, men_long_jerseys, women_jerseys, kids_kits, shorts)
  - Kit types (first/home, second/away, third, or any)
  - Season filtering

- **Organized Image Storage**: 
  - Images saved in: `./images/{category}/{team}/{type}.png`
  - Example: `./images/men_jerseys/liverpool/first.png`

## Installation

1. Clone the repository
2. Install dependencies:

```bash
pip install -r requirements.txt
playwright install chromium
```

3. Set up your Anthropic API key (optional but recommended for better AI detection):

```bash
export ANTHROPIC_API_KEY="your-api-key-here"
```

You can also pass it via `--api-key` flag.

## Usage

### Basic Usage

```bash
python jersey_scraper.py --url <store_url> --season <season> [options]
```

### Examples

**Scrape Liverpool first kit men jerseys:**
```bash
python jersey_scraper.py \
  --url https://www.minejerseys.ru \
  --season 2025/2026 \
  --team liverpool \
  --category men_jerseys \
  --type first
```

**Scrape multiple teams, all categories and types:**
```bash
python jersey_scraper.py \
  --url https://www.minejerseys.ru \
  --season 2025/2026 \
  --team liverpool,barcelona,real_madrid
```

**Scrape all supported teams:**
```bash
python jersey_scraper.py \
  --url https://www.minejerseys.ru \
  --season 2025/2026 \
  --team any
```

**Scrape with custom output directory:**
```bash
python jersey_scraper.py \
  --url https://www.minejerseys.ru \
  --season 2025/2026 \
  --team liverpool \
  --output ./my_jerseys
```

## Arguments

- `--url`: (Required) Store URL to scrape
- `--season`: (Required) Season to filter (e.g., 2025/2026)
- `--team`: Team name(s) - comma-separated or 'any' for all supported teams (default: any)
- `--category`: Category - men_jerseys, men_long_jerseys, women_jerseys, kids_kits, shorts, or 'any' (default: any)
- `--type`: Kit type - first, second, third, or 'any' (default: any)
- `--output`: Output directory for images (default: ./images)
- `--api-key`: Anthropic API key (alternatively set ANTHROPIC_API_KEY environment variable)

## Supported Teams

When using `--team any`, the tool searches for these teams:
- Liverpool
- Manchester United
- Manchester City
- Arsenal
- Chelsea
- Tottenham
- Barcelona
- Real Madrid
- Atletico Madrid
- Juventus
- Milan
- Inter Milan
- Bayern Munich
- Borussia Dortmund
- PSG

## Categories

- `men_jerseys`: Men's jerseys (short sleeve)
- `men_long_jerseys`: Men's long sleeve jerseys
- `women_jerseys`: Women's jerseys
- `kids_kits`: Kids/youth kits
- `shorts`: Shorts

## Kit Types

- `first`: First/Home kit
- `second`: Second/Away kit
- `third`: Third/Alternate kit
- `any`: All kit types

## How It Works

1. **Team Detection**: The scraper navigates to the store URL and uses AI to identify team pages based on link text
2. **Product Filtering**: For each team, it filters products by season, category, and kit type using AI analysis
3. **Image Extraction**: Enters each product page and downloads all product images
4. **Organized Storage**: Saves images in a structured folder hierarchy

## Notes

- The tool works best with an Anthropic API key for accurate AI detection
- Without an API key, it falls back to keyword matching (still functional but less accurate)
- The scraper respects website structure and uses proper delays to avoid overwhelming servers
- Images are saved with appropriate extensions (.png or .jpg) based on the source

## Troubleshooting

**No teams found:**
- Check if the store URL is correct
- The website structure might be different - try navigating manually first
- Ensure teams are listed on the homepage or accessible from it

**No products found:**
- Verify the season format matches the website (e.g., "2025/2026" vs "2025-2026")
- Check if the categories and types exist on the website
- Some teams might not have all kit types available

**Installation issues:**
- Make sure Python 3.8+ is installed
- Run `playwright install chromium` after installing requirements
- Check that you have sufficient disk space for browser installation

## License

This tool is for educational and personal use only. Please respect website terms of service and copyright laws when scraping content.
