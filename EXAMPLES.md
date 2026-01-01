# Usage Examples

This document provides detailed examples of how to use the AI Jersey Scraper tool.

## Prerequisites

1. Install dependencies:
```bash
pip install -r requirements.txt
playwright install chromium
```

2. (Optional but recommended) Set up Anthropic API key:
```bash
export ANTHROPIC_API_KEY="your-api-key-here"
```

## Basic Examples

### Example 1: Scrape a Single Team's First Kit
```bash
python jersey_scraper.py \
  --url https://www.minejerseys.ru \
  --season 2025/2026 \
  --team liverpool \
  --category men_jerseys \
  --type first
```

This will:
- Navigate to the store homepage
- Find Liverpool's team page
- Look for men's jerseys for the 2025/2026 season
- Filter for first/home kit only
- Download images to: `./images/men_jerseys/liverpool/first.png`

### Example 2: Scrape All Kits for a Team
```bash
python jersey_scraper.py \
  --url https://www.minejerseys.ru \
  --season 2025/2026 \
  --team barcelona \
  --category men_jerseys \
  --type any
```

This will download all kit types (first, second, third) for Barcelona:
- `./images/men_jerseys/barcelona/first.png`
- `./images/men_jerseys/barcelona/second.png`
- `./images/men_jerseys/barcelona/third.png`

### Example 3: Scrape Multiple Teams
```bash
python jersey_scraper.py \
  --url https://www.minejerseys.ru \
  --season 2025/2026 \
  --team liverpool,manchester_united,chelsea \
  --category men_jerseys \
  --type any
```

This will scrape men's jerseys (all types) for Liverpool, Manchester United, and Chelsea.

### Example 4: Scrape All Supported Teams
```bash
python jersey_scraper.py \
  --url https://www.minejerseys.ru \
  --season 2025/2026 \
  --team any \
  --category men_jerseys \
  --type first
```

This will scrape first kits for all 15 supported teams.

### Example 5: Scrape All Categories
```bash
python jersey_scraper.py \
  --url https://www.minejerseys.ru \
  --season 2025/2026 \
  --team liverpool \
  --category any \
  --type first
```

This will scrape Liverpool's first kit across all categories:
- `./images/men_jerseys/liverpool/first.png`
- `./images/men_long_jerseys/liverpool/first.png`
- `./images/women_jerseys/liverpool/first.png`
- `./images/kids_kits/liverpool/first.png`

### Example 6: Everything from a Team
```bash
python jersey_scraper.py \
  --url https://www.minejerseys.ru \
  --season 2025/2026 \
  --team real_madrid
```

When you omit `--category` and `--type`, they default to "any", so this will scrape:
- All categories (men, women, kids, shorts)
- All kit types (first, second, third)
- For Real Madrid

## Advanced Examples

### Example 7: Custom Output Directory
```bash
python jersey_scraper.py \
  --url https://www.minejerseys.ru \
  --season 2025/2026 \
  --team psg \
  --output ./my_custom_folder
```

Images will be saved to: `./my_custom_folder/{category}/{team}/{type}.png`

### Example 8: Using API Key via Command Line
```bash
python jersey_scraper.py \
  --url https://www.minejerseys.ru \
  --season 2025/2026 \
  --team arsenal \
  --api-key sk-ant-xxxxxxxxxxxxx
```

This passes the API key directly (useful if you don't want to set environment variable).

### Example 9: Different Season Format
```bash
python jersey_scraper.py \
  --url https://www.minejerseys.ru \
  --season "2024-2025" \
  --team juventus
```

The season parameter is flexible and will match however the website formats it.

## Understanding the Output

### Folder Structure
```
./images/
├── men_jerseys/
│   ├── liverpool/
│   │   ├── first.png
│   │   ├── second.png
│   │   └── third.png
│   ├── barcelona/
│   │   ├── first.png
│   │   └── second.png
│   └── ...
├── women_jerseys/
│   └── liverpool/
│       └── first.png
└── kids_kits/
    └── liverpool/
        ├── first.png
        └── second.png
```

### Multiple Images per Product
If a product has multiple images, they're saved as:
- `first.png` (main image)
- `first_2.png` (additional image)
- `first_3.png` (additional image)
- etc.

## Tips for Success

1. **Use the correct season format**: Check the website to see how they format seasons (e.g., "2025/2026" vs "2025-26" vs "25/26")

2. **Start small**: Test with a single team and category first before scraping everything

3. **Use AI features**: Set up the Anthropic API key for better team and kit detection

4. **Check the output**: After scraping, verify the images are correct and organized properly

5. **Be patient**: Scraping can take time, especially when scraping multiple teams and categories

## Troubleshooting

### No teams found
```bash
# Solution: Make sure the URL is the store homepage, not a specific team page
python jersey_scraper.py --url https://www.store.com --season 2025/2026 --team liverpool
```

### Wrong team detected
```bash
# Solution: Use the Anthropic API key for better AI detection
export ANTHROPIC_API_KEY="your-key"
python jersey_scraper.py --url ... --team liverpool
```

### Images not saving
```bash
# Solution: Check if you have write permissions in the output directory
python jersey_scraper.py --url ... --output ./images --team liverpool
```

## Team Name Reference

When using `--team`, use these exact names (or comma-separated combinations):

- liverpool
- manchester_united
- manchester_city
- arsenal
- chelsea
- tottenham
- barcelona
- real_madrid
- atletico_madrid
- juventus
- milan
- inter_milan
- bayern_munich
- borussia_dortmund
- psg

Or use `any` for all teams.

## Category Reference

When using `--category`, use these exact names:

- men_jerseys (short sleeve men's jerseys)
- men_long_jerseys (long sleeve men's jerseys)
- women_jerseys (women's jerseys)
- kids_kits (kids/youth kits)
- shorts

Or use `any` for all categories.

## Kit Type Reference

When using `--type`, use these exact names:

- first (home kit)
- second (away kit)
- third (third/alternate kit)

Or use `any` for all types.
