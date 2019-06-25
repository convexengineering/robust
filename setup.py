"""Python setup script for robust"""
from __future__ import print_function
from distutils.core import setup

description = """
generates and solves robust formulations for geometric and signomial programming. 
Requires installations of `GPkit <https://github.com/convexengineering/gpkit>`_. 
"""

license = """MIT License
Copyright (c) 2018 Convex Engineering
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE. """

setup(name='robust',
      version='0.0.0',
      description=description,
      url='https://github.com/convexengineering/robust',
      author='Ali Saab',
      author_email='saab@mit.edu',
      license=license,
      packages=["robust", "robust.data", "robust.feasibility_plots", "robust.signomial_simple_wing",
                "robust.simple_wing", "robust.solar_model", "robust.synthetic_model", "robust.testing",
                "robust.simulations"],
      package_data={"robust.data":['*.txt'], "robust.solar_model":['*.txt'], "robust.signomial_simple_wing":['*.txt'],
                    "robust.simple_wing":['*.txt']},
      install_requires=['gpkit', 'numpy', 'matplotlib', 'scipy'])
