"""Start-up checks for pipeline.submission: say what the run needs, find it, and stop early with a fix.

    python -m pipeline.startup          # run the checks on their own, without building anything

Standard library only, and imported before numpy, so that a wrong Python or a missing package gets a
plain message instead of a traceback from deep inside an import. Every check prints one line, so the
person running it can see what the terminal is doing before the pipeline itself starts.
"""

import importlib
import os
import platform
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import games  # noqa: E402  (standard library only, like this module)

PYTHON = (3, 12)
# (import name, pip name) for pipeline/requirements.txt.
PACKAGES = [("numpy", "numpy"), ("pandas", "pandas"), ("scipy", "scipy"), ("sklearn", "scikit-learn"),
            ("xgboost", "xgboost"), ("kloppy", "kloppy"), ("cv2", "opencv-python-headless"), ("av", "av")]
# The match files the pipeline reads, per recording (see "The match data" in README.md). The broadcast
# video is not read: only its tracking and homography, to find where the TV cut away from the pitch.
DATA = {"tactical": ["video", "atd_meta", "atd_raw", "homography", "smart_tagging"],
        "broadcast": ["atd_raw", "homography"]}
WEIGHTS = os.path.join(ROOT, "weights")
MODEL = os.path.join(WEIGHTS, "xgb_both_all7.json")
# Our detector's output on the tactical video. Not committed (frame-level data derived from the match
# video): built by ball_video.build() into the cache (common.BALL_VIDEO_PATH), or our copy, sent on
# request, dropped into weights/.
BALL_VIDEO_COMMITTED = os.path.join(WEIGHTS, "ball_video.npz")
CACHE = os.path.join(HERE, "cache")
BALL_VIDEO_CACHE = os.path.join(CACHE, "ball_video", "ball_video.npz")

OK, BAD = "✓", "✗"
_problems = []


def _out(text=""):
    sys.stdout.write(text + "\n")
    sys.stdout.flush()


def _line(ok, text):
    _out(f"      {OK if ok else BAD} {text}")


def _fail(text, fix):
    _line(False, text)
    _problems.append((text, fix))


def _size(path):
    n = os.path.getsize(path)
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024


def _short(path):
    return os.path.relpath(path, ROOT) if path.startswith(ROOT + os.sep) else path


def banner(title):
    rule = "═" * 78
    _out(f"\n{rule}\n  {title}\n{rule}")


def section(text):
    _out(f"\n  {text}")


def check_python():
    section("Python")
    have = sys.version_info[:2]
    text = f"Python {platform.python_version()} at {_short(sys.executable)}"
    if have == PYTHON:
        _line(True, text)
        return True
    _fail(text + f" (needs {PYTHON[0]}.{PYTHON[1]})",
          f"Run with Python {PYTHON[0]}.{PYTHON[1]}: newer versions such as 3.15 can't install the pinned scipy.\n"
          f"        If .venv was made with another Python, rebuild it from scratch:\n"
          f"          python3.12 -m venv --clear .venv && .venv/bin/pip install -r pipeline/requirements.txt")
    return False


def check_packages():
    section("Packages (pipeline/requirements.txt)")
    missing = []
    for module, pip_name in PACKAGES:
        try:
            mod = importlib.import_module(module)
        except Exception as error:      # xgboost raises XGBoostError, not ImportError, without libomp
            missing.append(pip_name)
            hint = ""
            if module == "xgboost" and "libomp" in str(error):
                hint = " (macOS: brew install libomp)"
            _line(False, f"{pip_name}: {type(error).__name__}{hint}")
            continue
        _line(True, f"{pip_name} {getattr(mod, '__version__', '')}")
    if missing:
        _problems.append((f"missing or broken: {', '.join(missing)}",
                          ".venv/bin/pip install -r pipeline/requirements.txt"
                          "   (and on macOS, brew install libomp for xgboost)"))


def check_data():
    from_env = bool(os.environ.get("MTAH_DATA_DIR"))
    section("Match data (the organisers' BAR-ATH bundle)")
    _out(f"      looking in {games.DATA_DIR}")
    _out(f"      ({'from MTAH_DATA_DIR' if from_env else 'the default ./data; set MTAH_DATA_DIR to look elsewhere'})")
    missing = []
    for source, assets in DATA.items():
        paths = games.asset_paths(source=source)
        for asset in assets:
            path = paths[asset]
            name = os.path.join(source, os.path.basename(path))
            if os.path.isfile(path):
                _line(True, f"{name:<44} {_size(path):>9}")
            else:
                _line(False, f"{name:<44} {'missing':>9}")
                missing.append(name)
    if missing:
        _problems.append((f"{len(missing)} match data file(s) not found in {games.DATA_DIR}",
                          "Put the organisers' match data folder (the one holding tactical/ and broadcast/) at\n"
                          "        ./data, or point to it:  export MTAH_DATA_DIR=\"/path/to/data\"\n"
                          "        See \"The match data\" in README.md for the exact files."))


def check_weights():
    section("Weights and video ball")
    if os.path.isfile(MODEL):
        _line(True, f"possession model     {_short(MODEL)}")
    else:
        _fail(f"possession model     {_short(MODEL)} missing", "git checkout weights/ (it is committed)")

    if os.path.isfile(BALL_VIDEO_CACHE):
        _line(True, f"video ball           {_short(BALL_VIDEO_CACHE)} (already in the cache)")
    elif os.path.isfile(BALL_VIDEO_COMMITTED):
        os.makedirs(os.path.dirname(BALL_VIDEO_CACHE), exist_ok=True)
        shutil.copyfile(BALL_VIDEO_COMMITTED, BALL_VIDEO_CACHE)
        _line(True, f"video ball           {_short(BALL_VIDEO_COMMITTED)} -> copied into the cache")
    else:
        _fail("video ball           not found",
              "build it (~11.6 h) with the video environment, or email chi.tsang.25@ucl.ac.uk or\n"
              "          sui.lim.25@ucl.ac.uk for our copy and put it at weights/ball_video.npz:\n"
              "          .venv-video/bin/python -c \"from pipeline import ball_video; ball_video.build()\"")


def describe_cache():
    section("Cache (pipeline/cache/)")
    built = [n for n in ("tracking.npz", "ball_filled.npz", "dead_ball.npz", "ball_seen.npz")
             if os.path.isfile(os.path.join(CACHE, n))]
    if len(built) == 4:
        _out("      found from an earlier run: the rebuild takes under a minute")
    else:
        _out("      first run: parsing the tracking and reading the video's audio and frames,")
        _out("      about 3-5 minutes; everything is cached, so later runs take under a minute")


def run(title="McDonald's Workers · ball possession pipeline"):
    """Every check, then exit with the list of fixes if any failed. Returns only when all passed."""
    banner(title)
    _out(f"  repository  {ROOT}")
    _out(f"  system      {platform.system()} {platform.release()} ({platform.machine()})")
    if check_python():
        check_packages()
    check_data()
    check_weights()
    if not _problems:
        describe_cache()
        _out(f"\n  {OK} all checks passed: building the labels\n")
        return
    _out(f"\n  {BAD} cannot start: {len(_problems)} problem(s) to fix first\n")
    for i, (text, fix) in enumerate(_problems, 1):
        _out(f"  {i}. {text}\n        {fix}\n")
    sys.exit(1)


if __name__ == "__main__":
    run()
