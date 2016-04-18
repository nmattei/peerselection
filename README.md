# Peerselection
README.md for PeerSelection (c) Nicholas Mattei and Data61/NICTA.

This is the full implementation and code used in developing our AAAI 2015 and 2016 ArXive paper on the (Exact) Dollar Partition Method.

For more information on this method, please see our publications.  Please cite these publications if you use this code in any academic capacity.

Strategyproof Peer Selection. Haris Aziz, Omer Lev, Nicholas Mattei, Jeffery S. Rosenschein, and Toby Walsh. arXiv:1604.03632.

URL: http://arxiv.org/abs/1604.03632

Strategyproof Peer Selection: Mechanisms, Analyses, and Experiments. Haris Aziz, Omer Lev, Nicholas Mattei, Jeffery S. Rosenschein, and Toby Walsh. 30th AAAI Conference on Artificial Intelligence (AAAI 2016), Feb. 2016.

PDF: http://www.nickmattei.net/docs/prize.pdf

This package will install the various peer selection functions mentioned in the papers along with all the supporting code and examples.  This can be used to either implement your own peer selection experiments or to verify the results from the papers.

## Main Components

- /experiments
    + Small scripts that show off some of the basic functionality of the library.  Includes the scripts for running the experiments in our papers (long_run.py)
- /notebooks
    + Notebooks for playing with the experiment harness and graphing some results
- /peerselect
    + The main code containing:
        * impartial.py: The implementation of the various mechanisms including Vanilla, Credible Subset, Dollar Partition, Exact Dollar Partition, Dollar Raffle, and Dollar Partition Raffle.
        * profile_generator.py: An adaptation of the code in PrefLib Tools to work for the domains of interest for peer selection.
- /tests
    + Tests for the code.  This is broken currently and the tests need to be fixed/updated/more coverage needs to be added.

# Dependencies

- Requires Python 3.3+, PyTest, NumPy.

- The notebooks in /notebooks require IPython and the full SciPy Stack (MatPlotLib, Pandas, etc.)

- To install use pip3 install -e . as we are still developing the library and you likely don't want to install it every time.  It may be best to do this in a virturalenv for any projects you are working on.

# TODO

1. Fix the testing directory (the tests do not pass currently, coverage is technically low).
2. Upload simple use case notebooks for showing basic experiment functionality.
3. Finish cleaning up the documentation in the main directory.
