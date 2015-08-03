#!/usr/bin/python3.4
# -*- coding: utf-8 -*-

import numpy as np
import unittest

import logging
LOG = logging.getLogger("generator")


class BasisGenerator(object):
    """BasisGenerator is a greedy basis generator from a vector set.

    This class can extract principle basis vectors from either a simple
    vector space, or a more complex vector space defined by a template
    function and a set of parameters.  The template functions are
    usually useful when matching a signal in a noisy sample by use of
    Bayesian inference of parameters.  This greedy basis generator can
    be used in order to speed up the products between the signal
    template and the data.

    """

    def __init__(self, parameter_space, error_threshold):
        """The Basis generator object.

        :param parameter_space: A VectorSpace instance containing the
            parameters to generate vectors in a vector space of your choice.
        :param error_threshold: The error threshold when to stop the
            basis generation.

        """
        # Save the error threshold
        self._error_threshold = error_threshold
        self._parameter_space = parameter_space

        self.__initialised = False
        self.__error = np.array([1])

        self.__basis_indices = np.array([], dtype=int)
        self.__basis = None
        self.__overlap_coefficients = None
        self.__grammian = None


        index = self._parameter_space.norm.argmax()
        self.add_basis(index)

        self.__initialised = True

    def _compute_overlap_coefficients(self):
        """This computes the overlap coefficients and stores them.

        We store the overlap coefficients, because the calculation takes
        time and we can cache the result for use in the next iteration.

        :returns: None

        """
        coefficients = np.zeros(
            (self._parameter_space.parameter.shape[0], 1))

        # Generate the basis to calculate the overlap coefficients.
        basis_to_use = self._parameter_space.template(
            self.__basis_indices[-1])

        # Use NumPy's iterator to calculate the overlap coefficients.
        iterator = np.nditer(
            coefficients, flags=['c_index'], op_flags=['writeonly'])
        while not iterator.finished:
            iterator[0] = self._parameter_space.inner_product(
                self._parameter_space.template(iterator.index),
                basis_to_use)
            iterator.iternext()

        # Add the coefficients to the list.
        if self.__overlap_coefficients is None:
            self.__overlap_coefficients = coefficients.reshape(
                1, coefficients.size)
        else:
            self.__overlap_coefficients = np.append(
                self.__overlap_coefficients,
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
        overlap_matrix = np.matrix(self.__overlap_coefficients)

        # Vectorize the error calculation function.
        overlap_errors = np.copy(self._parameter_space.norm)

        # Calculate the overlap errors by using the overlap
        # coefficients.
        iterator = np.nditer(
            overlap_errors, flags=['c_index'], op_flags=['readwrite'])
        while not iterator.finished:
            iterator[0] = iterator[0] - \
                overlap_matrix[:, iterator.index].transpose() * \
                self.__grammian_inverse * overlap_matrix[:, iterator.index]
            iterator.iternext()

        return overlap_errors

    def add_basis(self, index):
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
        # Save the indices for future use
        self.__basis_indices = np.append(self.__basis_indices, index)

        # Calculate the overlap_coefficients before we extend the
        # Grammian matrix
        self._compute_overlap_coefficients()

        # Do not recalculate all of the stuff. Save the old array, and
        # then initialise the new grammian with bigger dimensions and
        # set the old_grammian to be the top left bit.
        new_size = self.__error.size
        old_grammian = np.copy(self.__grammian)
        self.__grammian = np.zeros((new_size, new_size))

        if old_grammian is not None:
            self.__grammian[:-1, :-1] = old_grammian

        # The bottom-right element the norm of the latest basis
        self.__grammian[-1, -1] = self._parameter_space.norm[index]

        # Go through the the bottom row and through the right-most
        # column and just add all the elements by using the fact that
        # the Grammian is symmetric because of the property of the
        # inner-product in any vector space (even complex), that it is
        # commutative.
        for i in range(new_size - 1):
            coef = self.__overlap_coefficients[-1, self.__basis_indices[i]]
            self.__grammian[-1, i] = coef
            self.__grammian[i, -1] = coef

        # Calculate the inverse
        self.__grammian_inverse = np.linalg.inv(self.__grammian)

    def iterate(self):
        """Do a one step of iteration.

        This just executes a step of iteration where it first calculates
        the errors of projection of our vector space onto our basis set,
        then it checks if the maximum error is bellow the error
        threshold.  If it is not, then we find the vector, which has the
        highest projection error and then add it into our basis set.

        :returns: boolean. True if the iteration was the last one.

        """
        overlap_errors = self.get_overlap_errors()

        if overlap_errors.max() < self._error_threshold:
            # We have finished iterating.
            return True

        index = np.argmax(overlap_errors)
        self.__error = np.append(self.__error, overlap_errors[index])

        self.add_basis(index)

        # We have not finished the calculation
        return False

    def run(self, monitor=False):
        """Run the generator.
        
        :param monitor: Whether to log the progress.

        :returns: The basis vectors.
        """
        if monitor:
            LOG.info("The error for the computations is")

        calculation_done = False
        while not calculation_done:
            calculation_done = self.iterate()
            if monitor:
                LOG.info(self.__error[-1])

        # Generate the basis
        return np.array([
            self._parameter_space.template(i)
            for i in self.__basis_indices])


class VectorSpace(object):
    """The parameter space class.

    This is for representing the parameter space as a vector space with
    it's own inner product rule.

    """

    def __init__(self, parameters, covariance_matrix):
        """We instantiate the parameter space and set the covariance
        matrix for calculating the inner product.

        :param parameters: The 2D array representing the parameter space.
        :param covariance_matrix: The matrix, which should have the same
            dimensionality as the length of a vector returned by the
            `template` function of this class.
        """

        self.parameter = parameters
        self.__covariance_matrix = covariance_matrix
        self.norm = np.zeros(len(parameters))

        # Calculate all the norms when assigning
        for i in range(self.norm.size):
            temp = self.template(i)
            self.norm[i] = self.inner_product(temp, temp)

    def template(self, index):
        """
        This command is given the index in the parameter space and
        it gives back the template vector.
        
        :param index: The index of the parameter in the parameter space.
        """
        return self.parameter[index]

    def inner_product(self, a, b):
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
