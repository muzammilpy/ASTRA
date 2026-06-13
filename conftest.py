"""
Pytest configuration – add the app directory to sys.path so that
internal imports (from core.config import ...) resolve correctly
both when running the server and when running tests.
"""

import sys
import os

# Make `app/` internals importable as top-level packages (core, routers, etc.)
APP_DIR = os.path.join(os.path.dirname(__file__), "app")
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)
