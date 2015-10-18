#!/usr/bin/env python
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

This is still WIP!

Most of the formulae appearing in this module are taken from Ellis et
al. paper, which I have read a while ago.

"""

import numpy as np
from collections import namedtuple


STEADY_SOURCE = True


# Named tuples, which are equivalent to C structs, which are immutable
# and used to store collections of constants.  They are really fast, so
# passing them as function parameters is really efficient!
MeasurementProperties = namedtuple(
    "MeasurementProperties", "t_final dt_min dt_max")

NoiseVariance = namedtuple(
    "NoiseVariance", "white red powerlaw")

SourceParameters = namedtuple(
    "SourceParameters",
    "mass distance inclination phase polarisation theta phi frequency")

Pulsar = namedtuple(
    "Pulsar", "distance direction noise_variance")


class UnitVector:
    """The unit vector, which will return useful unit vectors when
    calculating antena patterns.
    """

    def __init__(self, theta, phi):
        """Initialise.

        :theta: TODO
        :phi: TODO

        """
        self._theta = theta
        self._phi = phi

    @property
    def omega(self):
        """This is the unit vector pointing from the GW source to the
        solar system barycenter (SSB)
        """
        return np.array([
            - np.sin(self._theta) * np.cos(self._phi),
            - np.sin(self._theta) * np.sin(self._phi),
            - np.cos(self._theta)])

    @property
    def polarisation_m(self):
        """One of the polarization tensor transformation vectors.
        """
        return np.array([-np.sin(self._phi), np.cos(self._phi), 0])

    @property
    def polarisation_n(self):
        """One of the polarization tensor transformation vectors.
        """
        return np.array([
            - np.cos(self._theta) * np.cos(self._phi),
            - np.cos(self._theta) * np.sin(self._phi),
            np.sin(self._theta)])


class PulsarGrid:
    """This is the class for storing all of the pulsars and properties
    about them.

    >>> pulsar_grid = PulsarGrid(
    ...     10,
    ...     coordinate_range=((10, 20), (0, np.pi), (-np.pi, np.pi)),
    ...     noise_amplitudes=(0, 0, 0))

    >>> len(pulsar_grid)
    10

    >>> pulsar_grid[0]
    Pulsar

    >>>

    """
    # pylint: disable=too-few-public-methods

    def __init__(self, number_of_pulsars, coordinate_range,
                 noise_amplitudes):
        self._pulsars = []
        # Number of dimensions
        dim = 3

        # Verify input
        for coord_range in coordinate_range:
            assert coord_range[0] < coord_range[1], \
                "The range should be given in a format: (min, max)"

        for __ in range(number_of_pulsars):
            # Place the pulsars in a 3D space.  Since we want the pulsars to
            # be randomly distributed, we use a random number generator,
            # which will generate numbers in the range of [0, 1).
            coords = np.random.rand(dim)

            # Transform the random number distributions to span the ranges
            # defined by the user. Also scale the noises so that we can
            # define different ratios of white, red and power law noise.
            for i in range(dim):
                coords[i] = (
                    coords[i] * (
                        coordinate_range[i][1] - coordinate_range[i][0])
                    + coordinate_range[i][0])

            distance, theta, phi = coords

            direction = np.array([
                np.sin(theta) * np.cos(phi),
                np.sin(theta) * np.sin(phi),
                np.cos(theta)])

            # Three kinds of noises, the actual realisations of them should
            # not be coorelated
            noise_variance = NoiseVariance(
                white=np.random.rand() * noise_amplitudes[0],
                red=np.random.rand() * noise_amplitudes[1],
                power_law=np.random.rand() * noise_amplitudes[2])

            self._pulsars.append(Pulsar(distance, direction, noise_variance))

    def __getitem__(self, index):
        return self._pulsars[index]

    def __len__(self):
        return len(self._pulsars)

    def __iter__(self):
        for pulsar in self._pulsars:
            yield pulsar


def antenna_pattern(source_parameters, u_p):
    """
    define antenna pattern functions as in (9) and (10)
    The F[0] term is Fplus and F[1] is Fcross
    """
    # Initialise unit vector according to the given parameters
    unit_source_vector = UnitVector(
        source_parameters.theta, source_parameters.phi)

    one = unit_source_vector.polarisation_m @ u_p
    two = unit_source_vector.polarisation_n @ u_p
    omega = unit_source_vector.omega @ u_p

    return (
        0.5 * (one ** 2 - two ** 2) / (1 + omega),
        one * two / (1 + omega))


def frequency(source_parameters, pulsar_time):
    """
    Define some required functions.

    """
    # The below snippet is for quickly evolving source. We are assume
    # that our source is slowly evolving.
    if STEADY_SOURCE:
        return source_parameters.frequency
    else:
        return (
            source_parameters.frequency**(-8/3)
            - 256/5 * source_parameters.mass**(5/3) * pulsar_time)**(-3/8)


def phase_term(source_parameters, pulsar_time):
    r"""
    Calculate the phase term, which is expressed as :math:`\Phi'(t)`, which is
    :math:`\Phi(t) - \Phi_0`, so there is only a part of formula from
    the Ellis et al. equation (14).

    However, the formula has the following:
    .. math::

        \Phi(t) - \phi_n

    where
    .. math::

        \phi_n = \phi_0 + \Phi_0

    so we essentially have:
    .. math::

        \Phi(t) - \phi_0 - \Phi_0 = \Phi'(t) - \phi_0

    The phase term returns :math:`\Phi'(t)`, so in here we need to subtract the
    extra parameter.

    """
    # The below snippet is for quickly evolving source. We are assume
    # that our source is slowly evolving.
    if STEADY_SOURCE:
        # If the source is assumed to be steady, then we do a Taylor
        # expansion of the main function and the below is the first
        # derivative term.
        return source_parameters.frequency * pulsar_time
    else:
        return (
            (source_parameters.frequency**(-5/3) -
             frequency(source_parameters, pulsar_time)**(-5/3)) /
            (32 * source_parameters.mass ** (5/3)))


def gw_contribution(source_parameters, pulsar_time):
    """
    define the GW contributions to the timing residuals as in (12) and
    (13) The first term is the plus, the second term is the cross as in
    the F function.
    """
    phase = (phase_term(source_parameters, pulsar_time)
             - source_parameters.phase)

    # pylint: disable=invalid-name
    a = - np.sin(2 * phase) * (1 + np.cos(source_parameters.inclination) ** 2)
    b = - np.cos(2 * phase) * np.cos(source_parameters.inclination)
    c = (source_parameters.mass**(5/3)
         / source_parameters.distance
         / frequency(source_parameters, pulsar_time) ** (1/3))

    return (
        c * (a * np.cos(2 * source_parameters.polarisation) +
             b * np.sin(2 * source_parameters.polarisation)),
        c * (a * np.sin(2 * source_parameters.polarisation) -
             b * np.cos(2 * source_parameters.polarisation)))


def amplitude(source_parameters):
    """Calculate the amplitudes a as shown in Ellis et al equation (18).
    """
    # Alias some variablesfor easier calculations.
    phase, polarisation = source_parameters.phase, source_parameters.polarisation
    zeta = source_parameters.mass ** (5/3) / source_parameters.distance
    cos_inclination = np.cos(source_parameters.inclination)

    # The formulas have 2 * polarisation in them, so do that before I calculate
    # the cos and sines
    # pylint: disable=invalid-name
    a, b = 1 + cos_inclination**2, 2 * cos_inclination
    cc = np.cos(2 * phase) * np.cos(2 * polarisation)
    cs = np.cos(2 * phase) * np.sin(2 * polarisation)
    sc = np.sin(2 * phase) * np.cos(2 * polarisation)
    ss = np.sin(2 * phase) * np.sin(2 * polarisation)

    # The below is just a simple vector with for components.
    return (zeta * (a * np.array([cc, -sc, cs, -ss]) +
                    b * np.array([ss, cs, -sc, -cc])))


def basis(source_parameters, time, u_p):
    """Define the time dependent basis A as show in Ellis et al equation (19).
    """

    # Do so initial calculations for easier to understand code.
    freq = frequency(source_parameters, time)
    phase = phase_term(source_parameters, time)
    sin_phase = np.sin(2 * phase)
    cos_phase = np.cos(2 * phase)
    f_plus, f_cross = antenna_pattern(source_parameters, u_p)

    # The below is is essentially a matrix because of the fact that the
    # frequency can be a time series and the *_phase can be time series
    # as well.  This means that we need to do the manipulation of the
    # output carefully and the it's better not to put freq term outside
    # the np.array definition.
    return np.array([
        freq ** (-1/3) * sin_phase * f_plus,
        freq ** (-1/3) * cos_phase * f_plus,
        freq ** (-1/3) * sin_phase * f_cross,
        freq ** (-1/3) * cos_phase * f_cross])


def pulsar_term(observation_time, source_parameters, distance_to_pulsar, u_p):
    """
    Define the pulsar term as in the eq (17)
    """
    v_s = UnitVector(source_parameters.theta, source_parameters.phi)

    pulsar_time = observation_time - distance_to_pulsar * (1 + v_s.omega @ u_p)
    f_plus, f_cross = antenna_pattern(source_parameters, u_p)
    s_plus, s_cross = gw_contribution(source_parameters, pulsar_time)

    return f_plus * s_plus + f_cross * s_cross


def individual_source_contribution(
        time, source_parameters, distance_to_pulsar, u_p):
    """
    Define the residual as a function of parameters.

    Part of the equation (16) from Ellis et al.
    """
    # Add the other contributions, note that the function is vectorised,
    # as the return value of the function `basis` is a matrix.
    return (
        amplitude(source_parameters) @ basis(source_parameters, time, u_p)
        + pulsar_term(time, source_parameters, distance_to_pulsar, u_p))


def noise(time, variance):
    """Define the noise term, which combines all types of noises.

    :param time: A time series to generate everything.

    :param variance: The variances for different times of noise. An
        instance of NoiseVariance.

    """
    noise_series = 0

    # Implement the white  noise
    if variance.white != 0:
        noise_series = np.random.rand(len(time)) * np.sqrt(variance.white)

    # TODO: Implement the red and power law noises as well as shown in
    # the van Haasteren et al. paper from 2009
    #
    # I found this: http://stackoverflow.com/questions/918736 , but do
    # not know yet how to apply it.

    # Turn these two types of noise of
    if variance.red != 0:
        raise NotImplementedError("Red noise is not implemented yet")

    if variance.powerLaw != 0:
        raise NotImplementedError("Red noise is not implemented yet")

    # Add all the noise contributions
    return noise_series


def residual(time, sources, pulsar):
    """Calculate the residual

    :param time: A time series to generate everything.

    :param sources: A

    :param pulsar: The properties of the pulsar.

    """
    signal = 0

    # Add contributions to the signal from all GW sources
    for source_parameters in sources:
        signal += individual_source_contribution(
            time, source_parameters, pulsar.distance, pulsar.direction)

    # Add noise to the signal and return the result
    return signal + noise(time, pulsar.noise_variance)


def generate_measurement_schedule(measurement_properties):
    """
    :param measurement_properties:

    """
    assert measurement_properties.dt_max > measurement_properties.dt_min, \
        "The dt_max needs to be higher than dt_min"

    ddt = measurement_properties.dt_max - measurement_properties.dt_min
    maximum_number_of_measurements = \
        measurement_properties.t_final / measurement_properties.dt_min

    # Use cum sum to get the time since the start of the experiment
    schedule = np.cumsum(
        # Here we generate a surpluss of random numbers in the range
        # of [0, 1) and then multiply them by ddt and add dt_min
        # which will give a bunch of random numbers in the range of
        # dt_min and dt_max.
        np.random.rand(maximum_number_of_measurements) * ddt
        + measurement_properties.dt_min)

    # Limit the size of the array only to include values, which are
    # smaller than t_final and append to the final array.
    return schedule[schedule < measurement_properties.t_final]


def data_generation(sources, pulsars, schedule_properties,
                    add_gw_background=False):
    """This function actually generates all of the data
    """
    residuals = []

    # Copy the structure of the array for the time log
    for pulsar in pulsars:
        if add_gw_background:
            raise NotImplementedError("TODO")

        time_series = generate_measurement_schedule(schedule_properties)
        residuals.append(residual(time_series, sources, pulsar))

    return residuals
