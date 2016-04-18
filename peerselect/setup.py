'''
    setup.py from the github CookieCutter
    project.

    Notes: First attempt at making a good package.

    This setup.py is an amalgmation of:
        https://github.com/audreyr/cookiecutter-pypackage
        https://the-hitchhikers-guide-to-packaging.readthedocs.org/en/latest/creation.html#setup-py-description
        http://docs.python-guide.org/en/latest/

    We are using Py.test which has some good docs and examples:
        http://pytest.org/latest/contents.html

    Useful examples of well setup projects are:
        https://github.com/mitsuhiko/flask
        https://github.com/scipy

    Remember that you have to register the package internally
    for it to import and for the tests to run:
        pip3 install -e . 

'''

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

'''
with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read().replace('.. :changelog:', '')
'''

requirements = [
    # TODO: put package requirements here
]

test_requirements = [
    'pytest'
]

setup(
    name='peerselect',
    version='0.01',
    description="Implementation and Experiments for Peer Selection",
    #long_description=readme + '\n\n' + history,
    author="Nicholas Mattei",
    author_email='nsmattei@gmail.com',
    url='https://github.com/XXXX',
    packages=[
        'peerselect',
    ],
    #package_dir={'{{ cookiecutter.repo_name }}':
    #             '{{ cookiecutter.repo_name }}'},
    #include_package_data=True,
    install_requires=requirements,
    #license="BSD",
    #zip_safe=False,
    #keywords='{{ cookiecutter.repo_name }}',
    #classifiers=[
    #    'Development Status :: 2 - Pre-Alpha',
    #    'Intended Audience :: Developers',
    #    'License :: OSI Approved :: BSD License',
    #    'Natural Language :: English',
    #    "Programming Language :: Python :: 2",
    #    'Programming Language :: Python :: 2.6',
    #    'Programming Language :: Python :: 2.7',
    #    'Programming Language :: Python :: 3',
    #    'Programming Language :: Python :: 3.3',
    #    'Programming Language :: Python :: 3.4',
    #],
    test_suite='tests',
    tests_require=test_requirements
)
