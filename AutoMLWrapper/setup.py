from setuptools import setup, find_packages

setup(
    name='automlwrapper',
    version='0.1',
    description='A wrapper for AutoML libraries',
    packages=find_packages(),
    include_package_data=True,
    package_data={
        'automlwrapper': ['AutoGluon/*.yaml', 'AutoSklearn/*.yaml', 'AutoKeras/*.yaml', './*.yaml'],
    },
    install_requires=[
        "openai==0.28",
        "kaggle",
        "openml==0.12.0",
        "tabpfn",
    ]
)