.. toliman documentation master file, created by
   sphinx-quickstart on Tue Oct 31 19:12:10 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to toliman's documentation!
===================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

About 
=====

This documentation was written by `Benjamin Pope`_, so you can blame him for missing content.

.. _Benjamin Pope: benjaminpope.github.io


.. Indices and tables
.. ==================

.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`

Introduction
============

TOLIMAN is a telescope concept aimed at detecting planets around `Alpha Centauri`_ A and B via their effects on the relative astrometry_ between these two stars. We need to measure the separation (4 arcseconds) to better than 1 microarcsecond! 

We will do this using a `diffractive pupil`_, which imprints a characteristic diffraction pattern on the light before it ever encounters optical aberrations inside the telescope.

.. _astrometry: http://adsabs.harvard.edu/abs/2005PASP..117.1021S

.. _Alpha Centauri: https://en.wikipedia.org/wiki/Alpha_Centauri

.. _diffractive pupil: https://arxiv.org/abs/1304.0370


Software
========

This initial TOLIMAN simulation package is written in Python using poppy_, a physical optics simulation package. Most of the important detail is in `Jupyter notebooks`_, which have the advantage of being fun and interactive. We expect to migrate to more classical Python libraries when we need to move to more serious simulations.

.. _poppy: https://github.com/mperrin/poppy

.. _Jupyter notebooks: http://jupyter.readthedocs.io/en/latest/install.html