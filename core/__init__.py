"""
core/__init__.py -- Makes `core` a Python package.

Also re-exports the three submodules under their original flat names so that
app.py can use either:
    from core.safetrak_core import ...
    import core.safetrak_db as db
    import core.safetrak_reports as reports
"""
