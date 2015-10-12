#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
This file is part of the PTA_ROQ.

PTA_ROQ is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

PTA_ROQ is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with PTA_ROQ.  If not, see <http://www.gnu.org/licenses/>.
"""

import numpy as np
import unittest

import logging
LOG = logging.getLogger("generator")


def _rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.

    Examples:
    >>> x = _rotation_matrix(np.array([0, 0, 1]), np.pi / 2)
    >>> x
    matrix([[  2.22044605e-16,  -1.00000000e+00,   0.00000000e+00],
            [  1.00000000e+00,   2.22044605e-16,   0.00000000e+00],
            [  0.00000000e+00,   0.00000000e+00,   1.00000000e+00]])


    """
    # pylint: disable=invalid-name,too-many-locals
    axis = np.asarray(axis)
    theta = np.asarray(theta)
    axis = axis/np.sqrt(np.dot(axis, axis))
    a = np.cos(theta / 2)
    b, c, d = - axis * np.sin(theta / 2)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


class CartesianBasisTestCase(unittest.TestCase):
    """Test the Greedy basis generator in a simple cartesian space.

    This test case tests the reduced basis generator for a simple
    Cartesian space, where the covariance matrix is just a simple
    identity matrix.
    """

    # pylint: disable=invalid-name

    def test_simple_cartesian(self):
        """ Test a non-degenerate 10D space.
        """
        params = VectorSpace(np.eye(10))
        params.extend(np.random.rand(100, 10))
        basis = generate_reduced_basis(params, 0.1)

        self.assertEqual(
            basis.reduced_basis.shape[0], 10,
            "Expected 10 reduced bases, got {}"
            .format(basis.reduced_basis.shape[0]))

    def test_a_degenerate_cartesian(self):
        """ Test a degenerate 11D space, where 1 dimension is zeros.
        """
        params = VectorSpace(np.eye(10))
        space = np.zeros((100, 11))
        space[:, :-1] = np.random.rand(100, 10)
        params.extend(space)
        basis = generate_reduced_basis(params, 0.1)

        self.assertEqual(
            basis.reduced_basis.shape[0], 10,
            "Expected 10 reduced bases, got {}"
            .format(basis.reduced_basis.shape[0]))

    def test_a_degenerate_cartesian_rotated(self):
        """ Test a degenerate 3D space, where 1 dimension is zeros, but
        the space is rotated.  """
        params = VectorSpace(np.eye(10))
        space = np.zeros((100, 3))
        space[:, :-1] = np.random.rand(100, 2)

        # Rotate the entire space
        for i in range(100):
            space[i] = _rotation_matrix([4, 4, 1], 1.2) @ space[i]

        params.extend(space)
        basis = generate_reduced_basis(params, 0.1)

        self.assertEqual(
            basis.reduced_basis.shape[0], 2,
            "Expected 2 reduced bases, got {}"
            .format(basis.reduced_basis.shape[0]))

class VectorSpace(object):
    """The parameter space class.

    The below shows the API of the basis generator and everything else
    needed to generate reduced basis.

    This is for representing the parameter space as a vector space with
    it's own inner product rule.

    Examples:

    >>> metric = np.eye(3)
    >>> space = VectorSpace(metric)  # Default is unity matrix
    >>> space
    VectorSpace(array([[ 1.,  0.,  0.],
                       [ 0.,  1.,  0.],
                       [ 0.,  0.,  1.]]))

    Extending an array and getting a vector from the space:

    >>> space.append(np.array([1, 0, 0]))
    >>> space[0]
    array([1, 0, 0])

    >>> space.extend(np.array([[0, 1, 0], [0, 2, 1]]))
    >>> space[1]
    array([0, 1, 0])

    >>> space[2]
    array([0, 2, 1])

    Get the covariance matrix associated with the space (i.e. the metric):
    >>> space.metric
    array([[ 1.,  0.,  0.],
           [ 0.,  1.,  0.],
           [ 0.,  0.,  1.]])

    Do a dot product in this vector space:

    >>> a = space.dot(space[2], space[1])   # return a real (or complex) number
    >>> a
    2.0

    or get the metric and do it manually:

    >>> a = space[1].T @ space.metric @ space[2]
    >>> a
    2.0

    Transform the input parameters by using a function:
    The function will be only used when returning the parameters, which
    means that if the transform functions need to be nested, the caller
    needs to do that himself.

    >>> space.transform = lambda x: 2 * x

    Now, let's repeat the same products as we did previously:

    >>> space.dot(space[2], space[1])   # return a real (or complex) number
    8.0

    Now, let's set the transformation function back to what it was:

    >>> space.transform = lambda x: x

    Get all norms of the vectors as a list:

    >>> norms = space.norms()
    >>> norms
    [1.0, 1.0, 5.0]

    or we can get a norm of an individual vector:

    >>> space.norm(1)
    1.0

    I need to be able to iterate through the vectors:
    >>> for vector in space:    # +ELLIPSIS
    ...    print(vector)
    [1 0 0]
    [0 1 0]
    [0 2 1]

    Get the shape of the space (i.e. the dimensionality of the space d*N,
    where d is the number of parameters and N is the number of vectors):

    >>> space.shape
    (3, 3)

    Get the number of vectors:
    >>> len(space)
    3

    """

    def __init__(self, metric):
        """We instantiate the parameter space and set the covariance
        matrix for calculating the inner product.

        :param metric: The matrix, which should have the same
            dimensionality as the length of a vector returned by the
            vector space.
        """

        self._space = []
        self.metric = metric

        # The following function is used in inner products.
        self.transform = lambda x: x

    def __repr__(self):
        name = "VectorSpace("
        matrix = "%r" % self.metric
        return name + ("\n" + " " * len(name)).join(matrix.split('\n')) + ")"

    def __getitem__(self, index):
        return self.transform(self._space[index])

    def __len__(self):
        return len(self._space)

    def __iter__(self):
        for i in self._space:
            yield i

    @property
    def shape(self):
        """Return the shape of the array as a tuple.

        The first element is the number of vectors and the second
        element is the number of elements in a vector.  Note, that this
        may depend on the actual `self.tranform` function as it can
        alter the number of elements in any vector in any way.
        """
        return (len(self), len(self[0]))

    def norm(self, index):
        """Get the norm of a particular vector with index.
        """
        return self.dot(self[index], self[index])

    def norms(self):
        """Get norms of all vectors in the vector space.
        """
        return [self.dot(a, a) for a in self]

    def append(self, vector):
        """Add a vector into the vector space.
        """
        self._space.append(vector)

    def extend(self, vector_subspace):
        """Extend the vector space by a 2D matrix of row vectors.
        """
        for vector in vector_subspace:
            self.append(vector)

    def dot(self, a, b):
        """ This is the definition of the inner product for some
        parameter space.  The inner product depends greatly on the
        covariance_matrix, which is passed to the class on
        initialising the parameter space.

        Parameters:
        -----------
        :param a: The first vector
        :param b: The second vector
        """
        # pylint: disable=invalid-name
        return a.T @ self.metric @ b


class SpaceReduce(object):
    """BasisGenerator is a greedy basis generator from a vector set.

    This class can extract principle basis vectors from either a simple
    vector space, or a more complex vector space defined by a template
    function and a set of parameters.  The template functions are
    usually useful when matching a signal in a noisy sample by use of
    Bayesian inference of parameters.  This greedy basis generator can
    be used in order to speed up the products between the signal
    template and the data.

    Note, that it is said, that because of rounding errors on a
    computer, a traditional Gram-Schmidt process is numerically
    unstable:
        https://en.wikipedia.org/wiki/Gram%E2%80%93Schmidt_process

    """

    def __init__(self, space, error_threshold):
        """The Basis generator object.

        :param parameter_space: A VectorSpace instance containing the
            parameters to generate vectors in a vector space of your choice.
        :param error_threshold: The error threshold when to stop the
            basis generation.

        """
        # Save the error threshold
        self._error_threshold = error_threshold
        self._space = space
        self._norms = abs(space)

        self.basis_indices = []
        self._overlap_coefficients = None
        self.grammian = None
        self.grammian_inverse = None

    def __iter__(self):
        """This is for creating an iterator to iterate over finding the next
        reduced basis which is in the set.

        This just executes a step of iteration where it first calculates
        the errors of projection of our vector space onto our basis set,
        then it checks if the maximum error is bellow the error
        threshold.  If it is not, then we find the vector, which has the
        highest projection error and then add it into our basis set.
        """

        # Start the iteration by adding the biggest vector of all
        index = self._norms.argmax()
        self.basis_indices.append(index)
        yield (self._space[index], 1)

        while True:
            self.update_grammian()
            overlap_errors = self.get_overlap_errors()
            index = np.argmax(overlap_errors)
            max_error = overlap_errors[index]

            if max_error < self._error_threshold:
                # We have finished iterating, break out of the loop and
                # do not yield anything else
                break

            # Yield a tuple, which is (vector, projection_error)
            self.basis_indices.append(index)
            yield (self._space[index], overlap_errors[index])

    def compute_overlap_coefficients(self):
        """This computes the overlap coefficients and stores them.

        We store the overlap coefficients, because the calculation takes
        time and we can cache the result for use in the next iteration.

        :returns: None

        """
        coefficients = np.zeros((len(self._space), 1))

        # Generate the basis to calculate the overlap coefficients.
        basis_to_use = self._space[self.basis_indices[-1]]

        # Use NumPy's iterator to calculate the overlap coefficients.
        iterator = np.nditer(
            coefficients, flags=['c_index'], op_flags=['writeonly'])
        while not iterator.finished:
            iterator[0] = self._space.dot(
                self._space[iterator.index], basis_to_use)

            iterator.iternext()

        # Add the coefficients to the list.
        if self._overlap_coefficients is None:
            self._overlap_coefficients = coefficients.reshape(
                1, coefficients.size)
        else:
            self._overlap_coefficients = np.append(
                self._overlap_coefficients,
                coefficients.reshape(1, coefficients.size),
                axis=0)

    def get_overlap_errors(self):
        """Calculate the errors of projecting VectorSpace onto our basis.

        This calculates the projection errors, which can then be used to
        find the vector in the vector space which is the least well
        represented by our basis.

        The dimensions of this array should be the same as the size of
        the parameter space.

        :returns: An array containing the projection errors.

        """
        overlap = np.array(self._overlap_coefficients)

        # Vectorize the error calculation function.
        overlap_errors = np.copy(self._norms)

        # Calculate the overlap errors by using the overlap
        # coefficients.
        iterator = np.nditer(
            overlap_errors, flags=['c_index'], op_flags=['readwrite'])

        while not iterator.finished:
            iterator[0] = iterator[0] - (
                overlap[iterator.index, :].T
                @ self.grammian_inverse
                @ overlap[iterator.index, :]
            )
            iterator.iternext()

        return overlap_errors

    def update_grammian(self):
        """Add the vector to the basis set.

        This adds a new basis vector into the list of our basis vector.
        After the vector is added, we recalculate the overlap
        coefficients between the basis vectors and the vectors in the
        `VectorSpace` we are traversing.  Then we recalculate the
        Grammian matrix so that we can calculate the errors of the
        projection of the vectors in the `VectorSpace` onto our basis
        set.

        :param index: The index in the `VectorSpace` in order to fetch
            the vector by passing the index into the `template` method.

        """
        index = self.basis_indices[-1]

        # Calculate the overlap_coefficients before we extend the
        # Grammian matrix
        self.compute_overlap_coefficients()

        # Do not recalculate all of the stuff. Save the old array, and
        # then initialise the new grammian with bigger dimensions and
        # set the old_grammian to be the top left bit.
        new_size = len(self.basis_indices)
        old_grammian = np.copy(self.grammian)
        self.grammian = np.zeros((new_size, new_size))

        if old_grammian is not None:
            self.grammian[:-1, :-1] = old_grammian

        # The bottom-right element the norm of the latest basis
        self.grammian[-1, -1] = self._space.norm(index)

        # Go through the the bottom row and through the right-most
        # column and just add all the elements by using the fact that
        # the Grammian is symmetric because of the property of the
        # inner-product in any vector space (even complex), that it is
        # commutative.
        for i in range(new_size - 1):
            coef = self._overlap_coefficients[-1, self.basis_indices[i]]
            self.grammian[-1, i] = coef
            self.grammian[i, -1] = coef

        # Calculate the inverse
        # FIXME: check in terms of speed if it's better to not calculate
        # the inverse, but use np.linalg.solve when we need to multiply
        # the inverse with something else.
        self.grammian_inverse = np.linalg.inv(self.grammian)


def generate_reduced_basis(input_space, target_error):
    """This function generates the reduced basis

    FIXME: finish the docstrings
    >> input_space = VectorSpace(covariance_matrix)
    >> input_space.transform = my_transformation_function
    >> input_space.extend(2D_array)

    >> reduced_basis = generate_reduced_basis(input_space, target_error)
    """
    reduced_basis_space = VectorSpace(input_space.metric)
    errors = np.empty((1, ), dtype=float)

    # The bellow starts the generator!
    for basis, error in SpaceReduce(input_space, target_error):
        errors.append(error)
        reduced_basis_space.append(basis)

    from collections import namedtuple
    # pylint: disable=invalid-name
    SimulationData = namedtuple(
        "SimulationData", "reduced_basis projection_error")

    return SimulationData(reduced_basis_space, errors)


if __name__ == "__main__":
    import doctest
    doctest.testmod()
