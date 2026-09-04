from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py as _build_py


class build_py(_build_py):
    """Put voicing_runtime.pth at site-packages root so site.py runs it first."""

    def run(self):
        super().run()
        root = Path(__file__).resolve().parent
        src = root / "src" / "voicing_runtime.pth"
        if not src.is_file():
            src = root / "voicing_runtime.pth"
        dest = Path(self.build_lib) / "voicing_runtime.pth"
        dest.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")


setup(cmdclass={"build_py": build_py})
