import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="distributedRL",
    version="0.0.1",
    author="cyoon1729",
    author_email="chrisyoon1729@gmail.com",
    description="distributed reinforcement learning frameworks",
    url="https://github.com/cyoon1729/distributedRL.git",
    packages=setuptools.find_packages(),
    python_requires=">=3.6",
    zip_safe=False,
)