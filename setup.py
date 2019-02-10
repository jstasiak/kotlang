from setuptools import setup


if __name__ == '__main__':
    setup(
        name='kotlang',
        license='MIT',
        packages=['kotlang'],
        install_requires=['llvmlite>=0.23.2', 'click'],
        extras_require={
            'dev': [
                'black',
                'flake8',
                'flake8-import-order',
                # We need to avoid 0.650 because of https://github.com/python/mypy/pull/6097
                'mypy>=0.620,!=0.650',
                'pytest',
                'pytest-cov',
            ]
        },
        entry_points={'console_scripts': ['kotc = kotlang.kotc_main:main']},
    )
