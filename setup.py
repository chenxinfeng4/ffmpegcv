#setup.py
from setuptools import setup, find_packages
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text('utf-8')

setup(
    name='ffmpegcv', # 应用名
    version='0.3.2', # 版本号
    packages=find_packages(include=['ffmpegcv*']), # 包括在安装包内的 Python 包
    author='chenxf',
    author_email='cxf529125853@163.com',
    url='https://pypi.org/project/ffmpegcv/',
    long_description=long_description,
    long_description_content_type='text/markdown',

    # 添加依赖项
    python_requires='>=3.6',
    install_requires=[
        'numpy',
    ]
)
