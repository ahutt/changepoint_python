from setuptools import setup

setup(name='changepoint',
      version='0.1',
      description='Implements various mainstream and specialised changepoint methods for finding single and multiple changepoints within data. Many popular non-parametric and frequentist methods are included. Users should start by looking at the documentation for cpt_mean(), cpt_var() and cpt_meanvar().',
      url='https://github.com/ahutt/changepoint_python',
      author='Alix Hutt, Rebecca Killick, Robin Long (all at Lancaster University)',
      author_email='alix.hutt@yahoo.com',
      license='GPL',
      packages=['changepoint'],
      zip_safe=False)
