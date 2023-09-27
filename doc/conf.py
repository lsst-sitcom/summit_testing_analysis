"""Sphinx configuration file for an LSST stack package.

This configuration only affects single-package Sphinx documentation builds.
For more information, see:
https://developer.lsst.io/stack/building-single-package-docs.html
"""

from documenteer.conf.pipelinespkg import *  # noqa F403

project = "summit_testing_analysis"
html_theme_options["logotext"] = project  # type: ignore # noqa
html_title = project
html_short_title = project
