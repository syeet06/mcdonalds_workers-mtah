"""Screenshots of the viewer for the technical document (submission/figures/viewer_*.png).

The match video and the ball overlay are hidden before anything renders, so only our own interface
is captured (label bands, hovers, the playhead plot, the discrepancy report): the rules don't allow
publishing clipped video. Needs Playwright and Google Chrome, and a viewer running on PORT:

    python -m pipeline.viewer --no-browser --port 8781 &
    pip install playwright==1.55.0
    python submission/make_viewer_shots.py
"""
import os
import sys
from playwright.sync_api import sync_playwright

PORT = 8781
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
URL = f"http://127.0.0.1:{PORT}/?tracks=McDonald's_Workers.csv&frame={{}}"
HIDE = """
  const s = document.createElement('style');
  s.textContent = `video, #overlay { visibility: hidden !important; }
                   .stage { background: #0b0d12 !important; }`;
  document.addEventListener('DOMContentLoaded', () => document.head.appendChild(s));
"""


def ready(page):
    page.wait_for_function("document.getElementById('bLo') && document.getElementById('bLo').textContent.trim() !== '—'"
                           " && document.getElementById('bLo').textContent.trim() !== ''", timeout=120000)
    page.wait_for_timeout(1500)


def open_at(ctx, frame):
    page = ctx.new_page()
    page.add_init_script(HIDE)
    page.goto(URL.format(frame))
    ready(page)
    assert page.evaluate("getComputedStyle(document.getElementById('video')).visibility") == "hidden"
    return page


def hover_shot(ctx, frame, which, name):
    page = open_at(ctx, frame)
    page.dispatch_event(f"#{'bDeadWrap' if which == 'dead' else 'bLoWrap'}", "mouseenter")
    page.wait_for_timeout(800)
    # the whole hover at its natural size: unpinned from the band, no height cap, no scrolling
    page.evaluate("""() => { const t = document.getElementById('deadTip');
        Object.assign(t.style, {position: 'fixed', top: '0px', left: '0px', bottom: 'auto', maxHeight: 'none',
                                overflow: 'visible', width: 'max-content', maxWidth: 'none', zIndex: 99,
                                background: '#0b0d12', opacity: '1', backdropFilter: 'none'}); }""")
    page.wait_for_timeout(300)
    page.locator("#deadTip").screenshot(path=f"{OUT}/{name}")
    page.close()


with sync_playwright() as p:
    browser = p.chromium.launch(channel="chrome", headless=True)
    ctx = browser.new_context(viewport={"width": 1700, "height": 1400}, device_scale_factor=1)
    hover_shot(ctx, 15893, "dead", "viewer_dead_hover.png")      # just into the public stoppage
    hover_shot(ctx, 15780, "lo", "viewer_model_hover.png")       # the 1.3 s Barcelona touch we merge
    page = open_at(ctx, 12452)                                   # log-odds crossing zero and back
    page.locator(".lanes").first.screenshot(path=f"{OUT}/viewer_playhead.png")
    page.close()
    page = open_at(ctx, 15780)
    # discrepancy report against the public ground truth only
    opts = page.eval_on_selector_all("#repRef option", "os => os.map(o => [o.value, o.textContent])")
    gt = [v for v, t in opts if "public" in t.lower() or "ground" in t.lower()]
    print("reference options:", opts)
    page.select_option("#repPred", "McDonald's_Workers.csv")
    if gt:
        page.select_option("#repRef", gt[0])
    page.select_option("#repSort", "cost")
    page.wait_for_timeout(2000)
    box = page.locator("aside.report").bounding_box()
    page.screenshot(path=f"{OUT}/viewer_report.png", full_page=True,
                    clip={"x": box["x"], "y": box["y"], "width": box["width"], "height": 900})
    browser.close()
print("ok")
