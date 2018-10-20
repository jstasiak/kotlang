from setuptools import setup


if __name__ == '__main__':
    setup(
        name='kotlang',
        license='MIT',
        packages=['kotlang'],
        install_requires=[
            'llvmlite>=0.23.2',
            'click',
        ],
        extras_require={
            'dev': [
                'flake8',
                'flake8-import-order',
                'mypy>=0.620',
                'pytest',
                'pytest-cov',
            ],
        },
        entry_points={
            'console_scripts': [
                'kotc = kotlang.kotc_main:main',
            ],
        },
    )
