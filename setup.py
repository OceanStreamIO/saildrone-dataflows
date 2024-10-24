from setuptools import setup, find_packages


def read_requirements(file_path):
    with open(file_path) as f:
        lines = f.read().splitlines()
    _requires = []
    _links = []
    for line in lines:
        if line.startswith("git+"):
            _links.append(line)
        else:
            _requires.append(line)
    return _requires, _links


install_requires, dependency_links = read_requirements('requirements.txt')

if __name__ == "__main__":
    setup(
        name="saildrone",
        version="0.1",
        packages=find_packages(),
        install_requires=install_requires
    )
