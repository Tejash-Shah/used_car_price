from setuptools import setup, find_packages


# Packages that are required for this module to be executed
def list_requirements(filename='requirements.txt'):
    with open(filename) as fd:
        return fd.read().splitlines()


setup(name='used_car_price',
      version='0.1',
      description='used_car_price_package',
      packages=find_packages(),
      install_requires=list_requirements(),
      include_package_data=True,
      zip_safe=False)
