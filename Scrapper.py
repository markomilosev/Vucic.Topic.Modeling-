"""
VUČIĆ.RS RESEARCH SCRAPER

Repeatedly clicks:
    "Учитајте више"

on:
    https://vucic.rs/n12/Vesti/Najnovije

and scrapes the articles as they become available.

The scraper:

1. Opens the Najnovije page with Playwright.
2. Collects all currently visible article URLs.
3. Scrapes those articles.
4. Clicks the visible "Учитајте више" button.
5. Waits for new articles.
6. Collects the new URLs.
7. Scrapes them.
8. Repeats until Load More stops producing articles.

It also discovers related articles from individual article pages.

Requirements:

    pip install playwright requests beautifulsoup4
    playwright install chromium
"""

import os
import re
import time

from urllib.parse import urljoin, urlparse

import requests
import urllib3

from bs4 import BeautifulSoup

from playwright.sync_api import sync_playwright


urllib3.disable_warnings(
    urllib3.exceptions.InsecureRequestWarning
)


# ============================================================
# CONFIGURATION
# ============================================================

START_URL = "https://vucic.rs/n12/Vesti/Najnovije"

ALLOWED_DOMAIN = "vucic.rs"

OUTPUT_DIR = r"B:\Articles\Vucic"

# Maximum number of articles.
# Set to None for no artificial limit.
MAX_ARTICLES = 5000

# Delay between article requests.
CRAWL_DELAY = 1.5

# Wait after clicking Load More.
LOAD_MORE_WAIT = 3

# If Load More produces no new articles this many times,
# stop.
MAX_STAGNANT_CLICKS = 3

REQUEST_TIMEOUT = 20

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/151.0.0.0 Safari/537.36 "
    "(research-scraper)"
)


HEADERS = {
    "User-Agent": USER_AGENT
}


# Matches URLs such as:

# /Vesti/Najnovije/a64960-Znanje-je-u-jednoj-zemlji-najvaznije.html

ARTICLE_URL_RE = re.compile(
    r"/Vesti/[^/]+/a\d+-[^\"'#\s]+\.html",
    re.IGNORECASE
)


# ============================================================
# FILE NAMING
# ============================================================

def safe_filename(url):

    path = urlparse(url).path.strip("/")

    name = re.sub(
        r"[^a-zA-Z0-9_-]+",
        "_",
        path
    )

    return name[:150] + ".txt"


# ============================================================
# ARTICLE LINK EXTRACTION
# ============================================================

def find_article_links(html, base_url):

    soup = BeautifulSoup(
        html,
        "html.parser"
    )

    found = set()

    for a in soup.find_all(
        "a",
        href=True
    ):

        full_url = urljoin(
            base_url,
            a["href"]
        ).split("#")[0]

        parsed = urlparse(full_url)

        if (
            parsed.netloc.endswith(ALLOWED_DOMAIN)
            and ARTICLE_URL_RE.search(parsed.path)
        ):

            found.add(full_url)

    return found


# ============================================================
# ARTICLE EXTRACTION
# ============================================================

def extract_article(html, url):

    soup = BeautifulSoup(
        html,
        "html.parser"
    )

    # Remove Javascript etc.
    for tag in soup(
        [
            "script",
            "style",
            "noscript"
        ]
    ):
        tag.decompose()

    body = soup.body or soup

    elements = list(
        body.find_all(
            [
                "h1",
                "h2",
                "h3",
                "h4",
                "p"
            ]
        )
    )

    h1 = soup.find("h1")

    title = (
        h1.get_text(
            strip=True
        )
        if h1
        else url
    )

    if h1 in elements:

        start_idx = (
            elements.index(h1) + 1
        )

    else:

        start_idx = 0

    stop_markers = (
        "остале вести",
        "related",
        "слични чланци",
        "повезане вести"
    )

    paragraphs = []

    for el in elements[start_idx:]:

        text = el.get_text(
            strip=True
        )

        if not text:
            continue

        if (
            el.name in (
                "h2",
                "h3",
                "h4"
            )
            and any(
                marker in text.lower()
                for marker in stop_markers
            )
        ):

            break

        if el.name == "p":

            paragraphs.append(text)

    # Deduplicate paragraphs
    seen = set()

    clean = []

    for p in paragraphs:

        if p not in seen:

            seen.add(p)

            clean.append(p)

    return {
        "title": title,
        "body": "\n\n".join(clean)
    }


# ============================================================
# SAVE ARTICLE
# ============================================================

def save_article(
    session,
    url,
    scraped_urls
):

    # --------------------------------------------------------
    # Already scraped?
    # --------------------------------------------------------

    filename = safe_filename(url)

    filepath = os.path.join(
        OUTPUT_DIR,
        filename
    )

    if os.path.exists(filepath):

        print(
            f"Already exists: {filename}"
        )

        scraped_urls.add(url)

        return True

    # --------------------------------------------------------
    # Request
    # --------------------------------------------------------

    try:

        response = session.get(
            url,
            timeout=REQUEST_TIMEOUT
        )

        response.raise_for_status()

    except requests.RequestException as e:

        print(
            f"FAILED: {url}"
        )

        print(
            f"       {e}"
        )

        return False

    # --------------------------------------------------------
    # Extract
    # --------------------------------------------------------

    article = extract_article(
        response.text,
        url
    )

    if not article["body"]:

        print(
            f"NO BODY: {url}"
        )

        return False

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    with open(
        filepath,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            f"URL: {url}\n"
        )

        f.write(
            f"TITLE: {article['title']}\n\n"
        )

        f.write(
            article["body"]
        )

    scraped_urls.add(url)

    print(
        f"SAVED: {article['title']}"
    )

    print(
        f"       {filepath}"
    )

    return True


# ============================================================
# FIND VISIBLE LOAD MORE
# ============================================================

def find_visible_load_more(page):

    # IMPORTANT:
    #
    # There are TWO "Учитајте више" elements.
    #
    # The first one is hidden:
    #
    #   js-load-more
    #
    # The second one is the actual visible button:
    #
    #   js-btn-load-more
    #
    # Therefore we explicitly target the latter.

    button = page.locator(
        "a.js-btn-load-more"
    )

    if button.count() == 0:

        return None

    if not button.first.is_visible():

        return None

    return button.first


# ============================================================
# MAIN
# ============================================================

def main():

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    print("=" * 70)

    print(
        "VUČIĆ.RS RESEARCH SCRAPER"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # HTTP session for articles
    # --------------------------------------------------------

    session = requests.Session()

    session.headers.update(
        HEADERS
    )

    # URLs discovered from listing
    discovered = set()

    # URLs already scraped
    scraped = set()

    # --------------------------------------------------------
    # Start browser
    # --------------------------------------------------------

    with sync_playwright() as p:

        browser = p.chromium.launch(
            headless=True
        )

        page = browser.new_page(
            viewport={
                "width": 1920,
                "height": 1080
            },
            user_agent=USER_AGENT
        )

        print(
            "\nOpening:",
            START_URL
        )

        page.goto(
            START_URL,
            wait_until="networkidle",
            timeout=60000
        )

        page.wait_for_timeout(
            3000
        )

        # ----------------------------------------------------
        # Load More loop
        # ----------------------------------------------------

        stagnant_clicks = 0

        cycle = 0

        while True:

            cycle += 1

            print("\n")
            print("=" * 70)

            print(
                f"LOAD CYCLE {cycle}"
            )

            print("=" * 70)

            # ------------------------------------------------
            # Collect currently available articles
            # ------------------------------------------------

            current_html = page.content()

            current_links = find_article_links(
                current_html,
                START_URL
            )

            new_links = (
                current_links
                - discovered
            )

            discovered.update(
                current_links
            )

            print(
                f"Article URLs currently visible: "
                f"{len(current_links)}"
            )

            print(
                f"New article URLs: "
                f"{len(new_links)}"
            )

            print(
                f"Total discovered: "
                f"{len(discovered)}"
            )

            # ------------------------------------------------
            # Scrape new articles immediately
            # ------------------------------------------------

            for url in sorted(new_links):

                if (
                    MAX_ARTICLES is not None
                    and len(scraped) >= MAX_ARTICLES
                ):
                    break

                save_article(
                    session,
                    url,
                    scraped
                )

                time.sleep(
                    CRAWL_DELAY
                )

            # ------------------------------------------------
            # Safety limit
            # ------------------------------------------------

            if (
                MAX_ARTICLES is not None
                and len(scraped) >= MAX_ARTICLES
            ):

                print(
                    "\nReached MAX_ARTICLES."
                )

                break

            # ------------------------------------------------
            # Find visible Load More
            # ------------------------------------------------

            button = find_visible_load_more(
                page
            )

            if button is None:

                print(
                    "\nNo visible Load More button."
                )

                print(
                    "Archive appears to be exhausted."
                )

                break

            # ------------------------------------------------
            # Number of articles before click
            # ------------------------------------------------

            before_count = len(
                discovered
            )

            print(
                "\nClicking 'Учитајте више'..."
            )

            # ------------------------------------------------
            # Click
            # ------------------------------------------------

            try:

                button.scroll_into_view_if_needed()

                page.wait_for_timeout(
                    500
                )

                button.click(
                    timeout=15000
                )

            except Exception as e:

                print(
                    "\nERROR clicking Load More:"
                )

                print(e)

                break

            # ------------------------------------------------
            # Wait for AJAX
            # ------------------------------------------------

            page.wait_for_timeout(
                LOAD_MORE_WAIT * 1000
            )

            # ------------------------------------------------
            # Check whether new articles appeared
            # ------------------------------------------------

            after_html = page.content()

            after_links = find_article_links(
                after_html,
                START_URL
            )

            after_new = (
                after_links
                - discovered
            )

            discovered.update(
                after_links
            )

            print(
                "\nAfter click:"
            )

            print(
                f"New article URLs: "
                f"{len(after_new)}"
            )

            print(
                f"Total discovered: "
                f"{len(discovered)}"
            )

            # ------------------------------------------------
            # Scrape newly loaded articles
            # ------------------------------------------------

            for url in sorted(after_new):

                if (
                    MAX_ARTICLES is not None
                    and len(scraped) >= MAX_ARTICLES
                ):
                    break

                save_article(
                    session,
                    url,
                    scraped
                )

                time.sleep(
                    CRAWL_DELAY
                )

            # ------------------------------------------------
            # Detect stagnation
            # ------------------------------------------------

            if len(discovered) == before_count:

                stagnant_clicks += 1

                print(
                    f"\nNo new articles."
                    f" Stagnant clicks: "
                    f"{stagnant_clicks}/"
                    f"{MAX_STAGNANT_CLICKS}"
                )

                if (
                    stagnant_clicks
                    >= MAX_STAGNANT_CLICKS
                ):

                    print(
                        "\nLoad More stopped "
                        "producing new articles."
                    )

                    break

            else:

                stagnant_clicks = 0

            # ------------------------------------------------
            # Save checkpoint
            # ------------------------------------------------

            checkpoint = os.path.join(
                OUTPUT_DIR,
                "listing_checkpoint.html"
            )

            with open(
                checkpoint,
                "w",
                encoding="utf-8"
            ) as f:

                f.write(
                    page.content()
                )

        browser.close()

    # ========================================================
    # OPTIONAL RELATED ARTICLES
    # ========================================================

    print("\n")
    print("=" * 70)

    print(
        "MAIN LISTING SCRAPE COMPLETE"
    )

    print("=" * 70)

    print(
        f"Articles scraped: {len(scraped)}"
    )

    print(
        f"Article URLs discovered: "
        f"{len(discovered)}"
    )

    print(
        f"Output directory: "
        f"{OUTPUT_DIR}"
    )

    print("\nDone.")


if __name__ == "__main__":

    main()