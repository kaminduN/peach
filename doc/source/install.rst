How to Install
==============

There is no special procedure to install this library. Peach is a pure Python
module and, as it is, there is no need to compile any source code. However,
Peach is dependent on some packages that must be installed so that it works
properly. These are listed below:

  The Python programming language
    Available at http://www.python.org/, Python is a very concise and expressive
    general purpose language. It is used in projects that range from web to
    scientific applications, and is available for a lot of platforms, from
    Windows to Mac OS X and Linux. If you need to deal with scientific
    programming, you will like it a lot, so I suggest you to try it even if
    using this module is not of your interest.

  The NumPy package
    The numeric processing package for Python, available at
    http://www.scipy.org/NumPy. It is not, unfortunatelly, made available with
    the standard distribution of the language. But there are some Python
    distributions that embed it by default. Anyway, installers are provided for
    almost every platform, and installing it is not difficult and should pose no
    problem.

  The bitarray module
    It is available at http://pypi.python.org/pypi/bitarray/0.3.5. It is needed
    for genetic algorithms, stochastic optimization and other techniques. Its
    installation is a little more difficult than the other packages, but it is,
    nonetheless, very easy to install. A C compiler might be needed, but, if
    present, the process is handled automatically. Please, follow the guidelines
    in the module's page.

  The matplotlib package
    Although it is not required, it is strongly suggested that the Matplotlib
    module is used. It is used to plot 2D graphics, and their data model is
    compatible with the one used in Peach (the NumPy array), so you can easily
    plot your results when needed. Some of the scripts in the tutorial section
    make use of it, if available. You can find it at
    http://matplotlib.sourceforge.net/

  The PyQt toolkit
    This is also not a required package for the module, but some of the demos
    use it for animations and building user interfaces. If you want to run those
    demos, you must install the 4th version of PyQt, which can be found at
    http://www.riverbankcomputing.co.uk/. You might also need a library for
    plotting graphs inside user interfaces that is compatible with PyQt. This
    library is the PyQwt module in its 5th version, which can be found at
    http://pyqwt.sourceforge.net/


Local installation
------------------

You can have a global or local installation of Peach. A local installation is
done in a local directory and is available only for scripts residing on that
directory. A global installation is available for every user and every
application. The following steps should suffice to obtain a working local
installation of the module:

- First, make sure that the above listed modules are installed and working with
  your Python distribution. In general, all you need to do is download the
  installation file and run it; or, if you are using Linux, install the packages
  using your package manager;

- Unpack the Peach file in a folder. Within this folder, look for the ``peach``
  folder. Copy this folder to your application directory. You can test if these
  procedures worked by going to the Python command line interface in the
  application directory and typing:

  >>> import peach


Global installation
-------------------

To install Peach for every user and every application, following these steps
should suffice to obtain a working global installation of the module:

- First, make sure that the above listed modules are installed and working with
  your Python distribution. In general, all you need to do is download the
  installation file and run it; or, if you are using Linux, install the packages
  using your package manager;

- Unpack the Peach file in a folder. Within this folder, look for the ``peach``
  folder. Copy this folder to the ``site-packages`` folder on you Python
  installation. You can test if these procedures worked by going to the Python
  command line interface and typing:

  >>> import peach
