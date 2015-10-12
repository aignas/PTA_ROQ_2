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
"""

import numpy as np
cos = np.cos
sin = np.sin


class UnitVector(object):
    """Docstring for UnitVector """

    def __init__(self, theta, phi):
        """The unit vector, which will return cartesian 

        :theta: TODO
        :phi: TODO

        """
        self._theta = theta
        self._phi = phi

        self.Omega = np.array([
            - sin(self._theta) * cos(self._phi),
            - sin(self._theta) * sin(self._phi),
            - cos(self._theta)])

        self.m = np.array([
            -sin(self._phi),
            cos(self._phi),
            0])

        self.n = np.array([
            - cos(self._theta) * cos(self._phi),
            - cos(self._theta) * sin(self._phi),
            sin(self._theta)])


class PulsarGrid(object):
    """Docstring for PulsarGrid. """

    def __init__(self, number_of_pulsars, pulsar_distance_range, noise_amplitudes):
        """TODO: to be defined1.

        :number_of_pulsars: TODO
        :pulsar_distance_range: This should be in spherical polar coordinates.
        :noise_amplitudes: TODO

        """
        self.pulsar_number = number_of_pulsars

        # Only 3 dimensions
        self.pulsar_coordinates = np.random.rand(number_of_pulsars, 3)

        # Three kinds of noises, the actual realisations of them should
        # not be coorelated
        self.white_noise, self.red_noise, self.power_law_noise = \
            np.random.rand(number_of_pulsars, 3)

        # Transform the random number distributions to span the ranges
        # defined by the user. Also scale the noises so that we can
        # define different ratios of white, red and power law noise.
        for i in range(3):
            if pulsar_distance_range[i, 0] > pulsar_distance_range[i, 1]:
                raise RuntimeError(
                    "The range should be given in a format: (min, max)")

            # Tranform the range
            self.pulsar_coordinates[:, i] *= (
                pulsar_distance_range[i, 1] - pulsar_distance_range[i, 0])
            self.pulsar_coordinates[:, i] += pulsar_distance_range[i, 0]

            # Transform the noise
            self.noise[:, i] *= noise_amplitudes[i]

        # Calculate a cartesian unit vectors:
        self.unit_vectors = np.array([
            sin(self.pulsar_coordinates[:, 1]) * cos(self.pulsar_coordinates[:, 2]),
            sin(self.pulsar_coordinates[:, 1]) * sin(self.pulsar_coordinates[:, 2]),
            cos(self.pulsar_coordinates[:, 0])])


def antennaPattern (theta, phi, u_p):
    """
    define antenna pattern functions as in (9) and (10)
    The F[0] term is Fplus and F[1] is Fcross
    """
    # Initialise unit vector according to the given parameters
    v = UnitVectors(theta, phi)

    d1 = np.dot(v.m, u_p)
    d2 = np.dot(v.n, u_p)
    d3 = np.dot(v.Omega, u_p)

    return np.ndarray([
        0.5 * ( d1 + d2 ) * ( d1 - d2 ) / ( 1 + d3 ),
        ( d1 * d2 ) / ( 1 + d3 )
    ])


def omega(time, fundamental_frequency, effective_mass):
    """
    Define some required functions.

    FIXME we can probably do some approximations here and there (See the
    Ellis et al. paper p3).
    """
    return (fundamental_frequency**(-8/3) 
        - 256/5 * effective_mass**(5/3) * time)**(-3/8)


def Phi (time, fundamental_frequency, effective_mass):
    """
    Calculate the phase term.

    FIXME: the Cornell et al. paper has a constant term here.  I believe,
    that the constant term here should be as well, but I am wondering if
    the constant term is the same as \Phi_0.
    """
    return 1/(32 * effective_mass**(5/3)) * (
        fundamental_frequency**(-5/3) 
        - omega(time, fundamental_frequency, effective_mass)**(-5/3))


def gWContribution (omega0, M, t, iota, psi, zeta, Phi0):
    """
    define the GW contributions to the timing residuals as in (12) and
    (13) The first term is the plus, the second term is the cross as in
    the F function.
    """

    gw = np.zeros(2)

    a = sin( 2 * (Phi(t, omega0, M) - Phi0) ) * (1 + cos(iota)**2)
    b = cos( 2 * (Phi(t, omega0, M) - Phi0) ) * cos(iota)
    gw = np.zeros(2)

    gw[0] = zeta / omega(t, omega0, M)**(1/3) * ( - a * cos(2*psi) - 2 * sin(2*psi) )
    gw[1] = zeta / omega(t, omega0, M)**(1/3) * ( - a * sin(2*psi) - 2 * cos(2*psi) )

    return gw


def amplitude(zeta, iota, phi, psi):
    """
    define the coefficients amplitudes as shown in the (18)
    """
    a = np.zeros(4)
    a[0] = zeta * ( (1 + cos(iota)**2) * cos (phi) * cos (2*psi) \
                + 2 * cos(iota) * sin (phi) * sin (2*psi) )

    a[1] = - zeta * ( (1 + cos(iota)**2) * sin (phi) * cos (2*psi) \
                + 2 * cos(iota) * cos (phi) * sin (2*psi) )

    a[2] = zeta * ( (1 + cos(iota)**2) * cos (phi) * sin (2*psi) \
                + 2 * cos(iota) * sin (phi) * cos (2*psi) )

    a[3] = - zeta * ( (1 + cos(iota)**2 ) * sin (phi) * sin (2*psi) \
                + 2 * cos(iota) * cos (phi) * cos (2*psi) )

    return a


def basis (omega0, M, theta, phi, t, u_p):
    """
    Define the time dependent basis functions as shown in the equation (19)
    """
    A = np.zeros(4)
    F = antennaPattern(theta, phi, u_p)
    A[0] = F[0] * omega(t, omega0, M)**(-1/3) * sin (2 * Phi(t, omega0, M))
    A[1] = F[0] * omega(t, omega0, M)**(-1/3) * cos (2 * Phi(t, omega0, M))
    A[2] = F[1] * omega(t, omega0, M)**(-1/3) * sin (2 * Phi(t, omega0, M))
    A[3] = F[1] * omega(t, omega0, M)**(-1/3) * cos (2 * Phi(t, omega0, M))

    return A

def pulsarTerm (t, M, D, iota, Phi0, psi, theta, phi, omega0, L, u_p):
    """
    Define the pulsar term as in the eq (17)
    """
    v = UnitVectors(theta, phi)
    tp = t - L * (1 + np.dot(v.Omega(), u_p))
    zeta = M**(5/3)/D

    F = antennaPattern(theta, phi, u_p)
    s = gWContribution (omega0, M, tp, iota, psi, zeta, Phi0)
    return np.dot(F,s)

def noise (double t, np.ndarray variance):
    """
    Define the noise term
    """
    cdef double white, red, powerLaw

    # Implement the white  noise
    white = np.random.rand() * sqrt(variance[0])

    # FIXME Implement the red and power law noises as well as shown in the van Haasteren
    # et al. paper from 2009
    # I found this: http://stackoverflow.com/questions/918736 , but do not know yet how
    # to apply it.

    # Turn these two types of noise of
    red = 0
    powerLaw = 0

    # Add all the noise contributions
    return white + red + powerLaw

def individualSource (double t, np.ndarray params, double L, np.ndarray u_p):
    """
    Define the residual as a function of parameters
    """
    cdef double M, D, iota, Phi0, psi, theta, phi, omega0, zeta, p
    cdef np.ndarray a, A

    # Unpack all the parameters:
    M, D, iota, Phi0, psi, theta, phi, omega0 = params.tolist()

    zeta = M**(5/3)/D
    a = amplitude(zeta, iota, phi, psi)
    A = basis (omega0, M, theta, phi, t, u_p)
    p = pulsarTerm (t, M, D, iota, Phi0, psi, theta, phi, omega0, L, u_p)
    return np.dot(a,A) + p

# Define the residual as a function of parameters
def residual (double t, np.ndarray sources, double L, np.ndarray u_p, np.ndarray variance):
    cdef double Signal = 0
    cdef int i

    # Add contributions to the signal from all GW sources
    for i in range(0,sources.shape[0],8):
        Signal += individualSource(t, sources[i:i+8], L, u_p)

    # Add noise to the signal
    Signal += noise (t, variance)

    return Signal

def genSchedule (np.ndarray schedule, double t_final, double dt_min, double dt_max):
    cdef double t
    cdef np.ndarray u_p
    cdef np.ndarray dates_out = np.array([]), index_out = np.array([])
    cdef int N = schedule.shape[0]
    collectData = True
    cdef int i

    # Copy the structure of the array for the time log
    dates = [np.array([])]*N

    # Start collecting the data
    while collectData:
        collectData = False

        for i in range(N):
            t = schedule[i]

            if t > t_final:
                continue
            else:
                # The data is being collected, add to the log
                dates[i] = np.append(dates[i], t)
                # Do not stop generating data if there is at least one pulsar, which had
                # not "gone" in to the future
                collectData = True

            # Update a schedule for this pulsar. We randomise the next measurement a bit
            schedule[i] += np.random.rand() * abs(dt_max - dt_min) + dt_min

    # FIXME Spit out the data in a format we need
    # Contract the data into a one vector
    for i in range(N):
        dates_out = np.append(dates_out, dates[i])
        index_out = np.append(index_out, [i]*dates[i].shape[0])

    return np.append([index_out], [dates_out], axis=0)

# Generate the actual data.
def dataGeneration (np.ndarray schedule, np.ndarray sources, pulsars, 
        addNoise=True, addGWB=False):
    cdef double t, L
    cdef np.ndarray u_p, a = np.array([]), noise = np.zeros(3)
    cdef int N = pulsars.getNumber()
    cdef int i

    # Copy the structure of the array for the time log

    # Start collecting the data
    for i in range(schedule.shape[1]):

        # Set some temporary values
        L = pulsars.getLength(schedule[0,i])
        u_p = pulsars.getUnitVector(schedule[0,i])

        if addNoise:
            noise = pulsars.getNoise(schedule[0,i])

        a = np.append(a, residual(schedule[1,i], sources, L, u_p, noise))

    return a

# Calculate the GWB terms in the covariance matrix members.
def covarianceMatrixMemberGWB (i, j, a, b, A, f, gamma, tau, N, C):
    cdef double alpha, sum, sum_member
    cdef int k

    alpha = 3/2 * C * log(C) - C/4 + 1/2
    # a simple delta function implementation
    if a == b:
        alpha += 1/2

    # Here I use slightly more memory by storring each_member before summing it, but
    # this way I do not have to calculate horrible factorials and it should speed things
    # up a bit
    # This function calculates N terms and then truncates the series
    sum_member = - 1 / (1 + gamma)
    sum = sum_member
    for k in range(N):
        sum_member *= - (f * tau)**2 / ((2*k + 1) * (2*k + 2)) * (2*k - 1 - gamma) \
                      / (2*k + 1 - gamma)
        sum += sum_member

    return A**2 * alpha / ((2 * np.pi)**2 * f**(1 + gamma)) \
           / (np.gamma(-1 - gamma) * sin(-np.pi * gamma / 2) * (f*tau)**(1 + gamma) - sum)

# Calculate white noise terms
def covarianceMatrixMemberWN (i,j,a,b,N):
    # r is the return value and N should be the noise amplitude
    cdef double r = 0

    if a==b and i==j:
        r = N**2

    return r

# Calculate red noise Lorentzian terms
def covarianceMatrixMemberLor (i,j,a,b,N,f,tau):
    # r is the return value and N should be the noise amplitude
    cdef double r = 0
    
    if a==b:
        r = N**2 * exp(-f*tau)

    return r

# Calculate the power law spectral noise
def covarianceMatrixMemberPowLaw (i, j, a, b, A, f, tau, gamma, N):
    cdef double r = 0
    cdef int k

    if a==b:
        # Here I use a similar technique to the one explained in
        # covarianceMatrixMemberGWB
        sum_member = 1 / (1 - gamma)
        sum = sum_member
        for k in range(N):
            sum_member *= - (f * tau)**2 / ((2*k + 1) * (2*k + 2)) * (2*k + 1 - gamma) \
                          / (2*k + 3 - gamma)
            sum += sum_member

        r =  A**2 / (f**(gamma - 1)) \
             * (np.gamma(1 - gamma) * sin(np.pi * gamma /2) * (f * tau)**( gamma -1) - sum)
    
    # Return the computed value
    return r
