# setup.py
from setuptools import setup, find_packages
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text("utf-8")


def get_version() -> str:
    """Read ``__version__`` from ffmpegcv/version.py (single source of truth)."""
    version_ns: dict = {}
    exec((this_directory / "ffmpegcv" / "version.py").read_text("utf-8"), version_ns)
    return version_ns["__version__"]


setup(
    name="ffmpegcv",  # 应用名
    version=get_version(),  # 版本号: 单一来源 ffmpegcv/version.py
    packages=find_packages(include=["ffmpegcv*"]),  # 包括在安装包内的 Python 包
    package_data={"ffmpegcv": ["py.typed"]},  # PEP 561 类型标记
    author="chenxf",
    author_email="cxf529125853@163.com",
    url="https://github.com/chenxinfeng4/ffmpegcv",
    long_description=long_description,
    long_description_content_type="text/markdown",
    # 添加依赖项
    python_requires=">=3.6",
    install_requires=[
        "numpy",
    ],
    extras_require={"cuda": ["pycuda"]},  # 定义一个名为cuda的可选依赖项，并指定pycuda
)
