"""
Regenerate docs/screenshot.png — a banner of the live app mid-generation.

Headless (no OS screen-recording permission needed). One-off deps:
    ./.venv/bin/pip install playwright pillow
    ./.venv/bin/playwright install chromium

Then, with the app running (./.venv/bin/python live.py):
    ./.venv/bin/python docs/shoot.py
"""

import os

from PIL import Image
from playwright.sync_api import sync_playwright

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "screenshot.png")
URL = "http://127.0.0.1:7861"

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 1160, "height": 1400}, device_scale_factor=2)
    page.goto(URL, wait_until="networkidle")
    page.wait_for_timeout(2500)                       # fonts + initial render
    page.get_by_role("button", name="Generate").first.click()
    page.wait_for_timeout(6000)                        # let it stream + fill the panels
    page.screenshot(path=OUT)
    browser.close()

# downscale to a sensible README width
im = Image.open(OUT)
w = 1400
im.resize((w, int(im.height * w / im.width)), Image.LANCZOS).save(OUT, optimize=True)
print("wrote", OUT)
