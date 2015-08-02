#!/usr/bin/python3.4
# -*- coding: utf-8 -*-

import numpy as np
import unittest

import logging
LOG = logging.getLogger("generator")


class BasisGenerator(object):
    """BasisGenerator
    
    This class is responsible for generating the Reduced Basis in a
    parameter space.  This parameter space can be either a simple vector
    space or something more complex like a waveform space.

    """

    def __init__(self, parameter_space, error_threshold):
        """TODO: to be defined1.

        :s: TODO

        """
        # Save the error threshold
        self.error_threshold = error_threshold
        self.params = parameter_space

        self.__initialised = False
        self.__error = np.array([1])

        self.__basis_indices = np.array([], dtype=int)
        self.__basis = None
        self.__overlap_coefficients = None
        self.__grammian = None


        index = self.params.norm.argmax()
        self.add_basis(index)

        self.__initialised = True

    @property
    def __grammian_inverse(self):
        return np.linalg.inv(self.__grammian)

    def _compute_overlap_coefficients(self):
        """This computes the overlap coefficients and stores them.

        We store the overlap coefficients, because the calculation takes
        time and we can cache the result for use in the next iteration.

        :returns: None

        """
        coefficients = np.zeros(
            (self.params.parameter.shape[0], 1))

        basis_to_use = self.__basis[-1]

        # Use NumPy's iterator
        it = np.nditer(coefficients, flags=['c_index'], op_flags=['writeonly'])
        while not it.finished:
            it[0] = self.params.braket(
                self.params.template(it.index), basis_to_use)
            it.iternext()

        if self.__overlap_coefficients is None:
            self.__overlap_coefficients = coefficients.reshape(
                1, coefficients.size)
        else:
            self.__overlap_coefficients = np.append(
                self.__overlap_coefficients,
                coefficients.reshape(1, coefficients.size),
                axis=0)

    def get_overlap_errors(self):
        """TODO: Docstring for _compute_overlap_errors.

        :returns: TODO

        """
        overlap_matrix = np.matrix(self.__overlap_coefficients)

        # Vectorize the error calculation function.
        overlap_errors = np.copy(self.params.norm)

        it = np.nditer(overlap_errors, flags=['c_index'],
                       op_flags=['readwrite'])
        while not it.finished:
            it[0] = it[0] - overlap_matrix[:,it.index].transpose() * \
                self.__grammian_inverse * overlap_matrix[:,it.index]
            it.iternext()

        return overlap_errors

    def add_basis(self, index):
        """Add the vector to the basis set.

        This also recalculates the grammian matrix.

        :arg1: TODO
        :returns: TODO

        """
        vector = self.params.template(index)

        self.__basis_indices = np.append(self.__basis_indices, index)
        if self.__basis is None:
            self.__basis = vector.reshape(1, vector.size)
        else:
            self.__basis = np.append(
                self.__basis,
                vector.reshape(1, vector.size),
                axis=0)

        new_size = self.__error.size
        old_grammian = np.copy(self.__grammian)
        self.__grammian = np.zeros((new_size, new_size))

        if old_grammian is not None:
            self.__grammian[:-1, :-1] = old_grammian

        self.__grammian[-1, -1] = self.params.norm[index]

        self._compute_overlap_coefficients()

        for i in range(new_size - 1):
            coef = self.__overlap_coefficients[-1, self.__basis_indices[i]]
            self.__grammian[-1, i] = coef
            self.__grammian[i, -1] = coef

    def iterate(self):
        """TODO: Docstring for _do_iteration.

        :arg1: TODO
        :returns: boolean. True if the iteration was the last one.

        """
        overlap_errors = self.get_overlap_errors()

        if overlap_errors.max() < self.error_threshold:
            return True

        index = np.argmax(overlap_errors)
        self.__error = np.append(self.__error, overlap_errors[index])

        self.add_basis(index)

    def run(self, monitor=False):
        """
        Do the generation
        """
        if monitor:
            LOG.info("The error for the computations is")

        calculation_done = False
        while not calculation_done:
            calculation_done = self.iterate()
            if monitor:
                LOG.info(self.__error[-1])

        return self.__basis


class VectorSpace(object):
    """Docstring for ParameterSpace. """

    def __init__(self, parameters, covariance_matrix):
        """Instansiate a parameter space, which has it's own inner product rule and the Grammian"""

        self.parameter = parameters
        self.__covariance_matrix = covariance_matrix
        self.norm = np.zeros(len(parameters))

        # Calculate all the norms when assigning
        for i in range(self.norm.size):
            temp = self.template(i)
            self.norm[i] = self.braket(temp, temp)

    def template(self, index):
        """
        This command is given the index in the parameter space and
        it gives back the template vector.
        """
        return self.parameter[index]

    def braket(self, a, b):
        """ This is the definition of the inner product for some
        parameter space.  The inner product depends greatly on the
        covariance_matrix, which is passed to the class on
        initialising the parameter space.

        Parameters:
        -----------
        :param a: The first vector
        :param b: The second vector
        """
        return (np.matrix(a)
            * self.__covariance_matrix * np.matrix(b).transpose()).item()


class CartesianBasisTestCase(unittest.TestCase):
    """Test the Greedy basis generator in a simple cartesian space.

    This test case tests the reduced basis generator for a simple
    Cartesian space, where the covariance matrix is just a simple
    identity matrix.
    """

    def test_simple_cartesian(self):
        """ Test a non-degenerate 10D space.
        """
        params = VectorSpace(np.random.rand(100,10), np.eye(10))

        generator = BasisGenerator(params, 0.1)
        basis = generator.run(monitor=True)

        self.assertEqual(
            basis.shape[0], 10,
            "Expected 10 reduced bases, got {}"
            .format(basis.shape[0]))

    def test_a_degenerate_cartesian(self):
        """ Test a degenerate 11D space, where 1 dimension is zeros.
        """
        space = np.zeros((100,11))
        space[:,:-1] = np.random.rand(100,10)
        params = VectorSpace(space, np.eye(11))

        generator = BasisGenerator(params, 0.1)
        basis = generator.run(monitor=True)

        self.assertEqual(
            basis.shape[0], 10,
            "Expected 10 reduced bases, got {}"
            .format(basis.shape[0]))

    def __construct_a_rotation_matrix(self, axis, theta):
        """
        Return the rotation matrix associated with counterclockwise rotation about
        the given axis by theta radians.
        """
        axis = np.asarray(axis)
        theta = np.asarray(theta)
        axis = axis/np.sqrt(np.dot(axis, axis))
        a = np.cos(theta/2)
        b, c, d = -axis * np.sin(theta/2)
        aa, bb, cc, dd = a*a, b*b, c*c, d*d
        bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
        return np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                         [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                         [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])

    def test_a_degenerate_cartesian_rotated(self):
        """ Test a degenerate 3D space, where 1 dimension is zeros, but the space is rotated.
        """
        space = np.zeros((100,3))
        space[:,:-1] = np.random.rand(100,2)

        # Rotate the entire space
        for i in range(100):
            space[i] = np.dot(
                self.__construct_a_rotation_matrix([4, 4, 1], 1.2), space[i])

        params = VectorSpace(space, np.eye(3))

        generator = BasisGenerator(params, 0.1)
        basis = generator.run(monitor=True)

        self.assertEqual(
            basis.shape[0], 2,
            "Expected 2 reduced bases, got {}"
            .format(basis.shape[0]))


if __name__ == "__main__":
    unittest.main()
