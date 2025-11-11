import os
from playwright.sync_api import sync_playwright
import argparse
from PIL import Image
import time


def take_screenshot(url, output_file="screenshot.png"):
    # Convert local path to file:// URL if it's a file
    if os.path.exists(url):
        url = "file://" + os.path.abspath(url)

    try:
        with sync_playwright() as p:
            # Choose a browser, e.g., Chromium, Firefox, or WebKit
            browser = p.chromium.launch(headless=True, args=["--disable-web-security"])
            page = browser.new_page()

            # Navigate to the URL
            page.goto(url, timeout=60000)
            # page.wait_for_timeout(1000)  # give it 1 second to paint

            # Take the screenshot
            page.screenshot(path=output_file, full_page=True, animations="disabled", timeout=60000)

            browser.close()
    except Exception as e:
        print(f"Failed to take screenshot due to: {e}. Generating a blank image.")
        # Generate a blank image
        img = Image.new("RGB", (1280, 960), color="white")
        img.save(output_file)


if __name__ == "__main__":

    # Initialize the parser
    parser = argparse.ArgumentParser(description="Process two path strings.")

    # Define the arguments
    parser.add_argument("--html", type=str)
    parser.add_argument("--png", type=str)

    # Parse the arguments
    args = parser.parse_args()

    take_screenshot(args.html, args.png)
