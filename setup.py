import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt", 'r') as fr:
    req_pgks_list = fr.readlines()

setuptools.setup(
    name="rewardpredictive",
    version="0.0.1",
    author="Lucas Lehnert",
    author_email="lucas_lehnert@brown.edu",
    description="rewardpredictive",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/lucaslehnert/rewardpredictive",
    packages=find_packages(),
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=req_pgks_list,
    python_requires='==3.7'
)
