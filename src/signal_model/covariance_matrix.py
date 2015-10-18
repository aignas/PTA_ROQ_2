
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
    r = 0

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
