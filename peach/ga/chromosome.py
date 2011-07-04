################################################################################
# Peach - Computational Intelligence for Python
# Jose Alexandre Nalon
#
# This file: ga/chromosome.py
# Basic definitions for manipulating chromosomes
################################################################################

# Doc string, reStructuredText formatted:
__doc__ = """
Basic definitions and classes for manipulating chromosomes

This sub-package is a vital part of the genetic algorithms framework within the
module. This uses the ``bitarray`` module to implement a chromosome as an array
of bits. It is, thus, necessary that this module is installed in your Python
system. Please, check within the Python website how to install the ``bitarray``
module.

The class defined in this module is derived from ``bitarray`` and can also be
derived if needed. In general, users or programmers won't need to instance this
class directly -- it is manipulated by the genetic algorithm itself. Check the
class definition for more information.
"""

################################################################################
from bitarray import bitarray
import struct
import types


################################################################################
# Classes
################################################################################
class Chromosome(bitarray):
    '''
    Implements a chromosome as a bit array.

    Data is structured according to the ``struct`` module that exists in the
    Python standard library. Internally, data used in optimization with a
    genetic algorithm are represented as arrays of bits, so the ``bitarray``
    module must be installed. Please consult the Python package index for more
    information on how to install ``bitarray``. In general, the user don't need
    to worry about how the data is manipulated internally, but a specification
    of the format as in the ``struct`` module is needed.

    If the internal format of the data is specified as an ``struct`` format, the
    genetic algorithm will take care of encoding and decoding data from and to
    the optimizer. However, it is possible to specify, instead of a format, the
    length of the chromosome. In that case, the fitness function must deal with
    the encoding and decoding of the information. It is strongly suggested that
    you use ``struct`` format strings, as they are much easier. This second
    option is provided as a convenience.

    The ``Chromosome`` class is derived from the ``bitarray`` class. So, every
    property and method of this class should be accessible.
    '''
    def __new__(cls, fmt='', endian='little'):
        '''
        Allocates new memory space for the chromosome

        This function overrides the ``bitarray.__new__`` function to deal with
        the length of the chromosome. It should never be directly used, as it is
        automatically called by the Python interpreter in the moment of object
        creation.

        :Returns:
          A new ``Chromosome`` object.
        '''
        if type(fmt) == int:
            return bitarray.__new__(cls, fmt)
        elif type(fmt) == str:
            size = struct.calcsize(fmt) * 8
            return bitarray.__new__(cls, size)
        elif isinstance(fmt, bitarray):
            return bitarray.__new__(cls, fmt)


    def __init__(self, fmt=''):
        '''
        Initializes the chromosome.

        This method is automatically called by the Python interpreter and
        initializes the data in the chromosome. No data should be provided to be
        encoded in the chromosome, as it is usually better start with random
        estimates. This method, in particular, does not clear the memory used in
        the time of creation of the ``bitarray`` from which a ``Chromosome``
        derives -- so the random noise in the memory is used as initial value.

        :Parameters:
          fmt
            This parameter can be passed in two different ways. If ``fmt`` is a
            string, then it is assumed to be a ``struct``-format string. Its
            size is calculated and a ``bitarray`` of the corresponding size is
            created. Please, consult the ``struct`` documentation, since what is
            explained there is exactly what is used here. For example, if you
            are going to use the optimizer to deal with three-dimensional
            vectors of continuous variables, the format would be something
            like::

              fmt = 'fff'

            If ``fmt``, however, is an integer, then a ``bitarray`` of the given
            length is created. Note that, in this case, no format is given to
            the chromosome, and it is responsability of the programmer and the
            fitness function to provide for it.

            Default value is an empty string.
        '''
        if type(fmt) == int:
            self.__size = fmt
            self.format = None
        elif type(fmt) == str:
            self.__size = len(self)
            self.format = fmt
        elif isinstance(fmt, bitarray):
            self.__size = len(self)
            self.format = fmt.format
            '''Property that contains the chromosome ``struct`` format.'''


    def __get_size(self):
        return self.__size
    size = property(__get_size, None)
    '''Property that returns the chromosome size. Not writable.'''


    def decode(self):
        '''
        This method decodes the information given in the chromosome.

        Data in the chromosome is encoded as a ``struct``-formated string in a
        ``bitarray`` object. This method decodes the information and returns the
        encoded values. If a format string is not given, then it is assumed that
        this chromosome is just an array of bits, which is returned.

        :Returns:
          A tuple containing the decoded values, in the order specified by the
          format string.
        '''
        if self.format is None:
            return self
        return struct.unpack(self.format, self.tostring())


    def encode(self, values):
        '''
        This method encodes the information into the chromosome.

        Data in the chromosome is encoded as a ``struct``-formated string in a
        ``bitarray`` object. This method encodes the given information in the
        bitarray. If a format string is not given, this method raises a
        ``TypeError`` exception.

        :Parameters:
          values
            A tuple containing the values to be encoded in an order consistent
            with the given ``struct``-format.
        '''
        if self.format is None:
            raise TypeError, 'no encoding/decoding format available'
        tmp = bitarray()
        tmp.fromstring(struct.pack(self.format, *values))
        self[:] = tmp

################################################################################
