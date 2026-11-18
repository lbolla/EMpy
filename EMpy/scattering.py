import numpy
import EMpy

__author__ = "Julien Hillairet"


class Field:
    """Class to describe an electromagnetic field.

    A EM field is a vector field which is the combination of an electric field E and a magnetic field H.
    These fields are defined on some space point r

    @ivar E: Electric vector on point M{r} in the cartesian coordinate system M{(Ex,Ey,Ez)}
    @type E: 2d numpy.ndarray of shape (3,N)
    @ivar H: Magnetic vector on point M{r} in the cartesian coordinate system M{(Mx,My,Mz)}
    @type H: 2d numpy.ndarray of shape (3,N)
    @ivar r: Position vector in cartesian coordinates M{(x,y,z)}
    @type r: 2d numpy.ndarray of shape (3,N)

    """

    def __init__(self, E=None, H=None, r=None):
        """Initialize the Field object with an Electric, Magnetic and Position vectors."""
        self.E = numpy.zeros((3, 1)) if E is None else E
        self.H = numpy.zeros((3, 1)) if H is None else H
        self.r = numpy.zeros((3, 1)) if r is None else r


"""
##############################################################
utils functions for field manipulations
##############################################################
"""


def stack(X, Y, Z):
    """Stack 3 vectors of different lengths into one vector.

    stack 3 vectors of different length to one vector of type
    M{P(n) = [x(n); y(n); z(n)]} with M{n=[1:Nx*Ny*Nz]}

    Examples
    ========
        >>> import numpy
        >>> X = numpy.arange(4); Y = numpy.arange(3); Z = numpy.arange(2)
        >>> S = field.stack(X,Y,Z)
        >>> print S
        [[ 0.  0.  0.  0.  0.  0.  1.  1.  1.  1.  1.  1.  2.  2.  2.  2.  2.  2.  3.  3.  3.  3.  3.  3.]
         [ 0.  0.  1.  1.  2.  2.  0.  0.  1.  1.  2.  2.  0.  0.  1.  1.  2.  2.  0.  0.  1.  1.  2.  2.]
         [ 0.  1.  0.  1.  0.  1.  0.  1.  0.  1.  0.  1.  0.  1.  0.  1.  0.  1.  0.  1.  0.  1.  0.  1.]]

    Checking the size of the returned array
        >>> numpy.size(S,axis=1) == numpy.size(X,axis=0)*numpy.size(Y,axis=0)*numpy.size(Z,axis=0)
        True

    @param X: first vector.of length Nx
    @type X: numpy.ndarray of shape (1, Nx).
    @param Y: second vector.of length Ny
    @type Y: numpy.ndarray of shape (1, Ny).
    @param Z: third vector.of length Nz
    @type Z: numpy.ndarray of shape (1, Nz).

    @return: stacked array.
    @rtype: numpy.ndarray of shape (3, Nx*Ny*Nz)

    """
    X = numpy.asarray(X).ravel()
    Y = numpy.asarray(Y).ravel()
    Z = numpy.asarray(Z).ravel()

    Nx = X.size
    Ny = Y.size
    Nz = Z.size

    XX = numpy.repeat(X, Ny * Nz)
    YY = numpy.tile(numpy.repeat(Y, Nz), Nx)
    ZZ = numpy.tile(Z, Nx * Ny)

    return numpy.vstack((XX, YY, ZZ))


def matlab_dot(a, b):
    """
    "classical" dot product (as defined in octave or matlab)

    Example
    =======
        >>> a=numpy.array([1,2,3]).repeat(6).reshape((3,6))
        >>> print a
        [[1 1 1 1 1 1]
         [2 2 2 2 2 2]
         [3 3 3 3 3 3]]
        >>> field.matlab_dot(a,a)
        array([14, 14, 14, 14, 14, 14])

    where M{14=1*1+2*2+3*3}

    @param a: first vector of length N
    @type  a: numpy.ndarray of shape (M,N)
    @param b: second vector of length N
    @type  b: numpy.ndarray of shape (M,N)

    @return: "classical" dot product between M{a} and M{b}
    @rtype: numpy.ndarray of shape (1,N)

    """
    return numpy.sum(a * b, axis=0)


"""
##########################################
Common functions for calculating EM fields
##########################################
"""


def currentsScatteringKottler(P, J, M, Q, dS, f, epsr=1):
    """
    Compute the scattered fields on some point M{P} in cartesian coordinates M{P(Px,Py,Pz)}
    by electric (and even magnetic) currents densities M{J}, M{M}
    defined on a surface M{Q(Qx,Qy,Qz)}.
    M{dS} corresponds to the surface element M{dS} of surface M{Q}

    TODO:
     - Maybe an option to compute only main expression in farfield
     - Optimisation ? (At the present time, python version is slower than the matlab code)

    @param P: observation points M{P(Px,Py,Pz)}.
    @type  P: numpy.ndarray of shape (3,NbP)
    @param J: electric current density on M{Q}
    @type  J: numpy.ndarray of shape (3,NbQ)
    @param M: magnetic current density on M{Q}
    @type  M: numpy.ndarray of shape (3,NbQ)
    @param Q: radiating (surface) points M{Q(Qx,Qy,Qz)}.
    @type  Q: numpy.ndarray of shape (3,NbQ)
    @param dS: surface elements of the radiating surface
    @type dS: numpy.ndarray of shape (1,NbQ)
    @param f: frequency
    @type  f: float
    @param epsr: relative permittivity (dielectric constant)  (defaut : 1, vacuum)
    @type  epsr: float

    @return: EM Field on M{P}
    @rtype: L{Field}


    """
    P = numpy.asarray(P)
    Q = numpy.asarray(Q)
    J = numpy.asarray(J)
    M = numpy.asarray(M)
    dS = numpy.asarray(dS)

    # test if the parameters size are correct
    if (
        P.ndim != 2
        or Q.ndim != 2
        or J.ndim != 2
        or M.ndim != 2
        or P.shape[0] != 3
        or Q.shape[0] != 3
        or J.shape[0] != 3
        or M.shape[0] != 3
    ):
        raise ValueError("Bad parameters size : P, Q, J, M must have shape (3, N).")

    if dS.ndim > 2 or (dS.ndim == 2 and 1 not in dS.shape):
        raise ValueError("Bad parameters size : dS must be 1D or shape (1, NbQ).")
    dS = dS.ravel()

    if not (Q.shape[1] == J.shape[1] == M.shape[1] == dS.size):
        raise ValueError(
            "Bad parameters size : number of columns between Q, J, M and dS must be equal."
        )

    lambda0 = EMpy.constants.c / f
    n_med = numpy.sqrt(numpy.asarray(epsr, dtype=complex))
    if n_med.real - n_med.imag < 0 or (n_med.real - n_med.imag == 0 and n_med.real < 0):
        n_med = -n_med
    lambdam = lambda0 / n_med
    # k0 = 2*numpy.pi/lambda0
    km = 2 * numpy.pi / lambdam
    Z0 = EMpy.constants.eta0
    NbP = numpy.size(P, axis=1)

    # preallocation
    EMF_P = Field(
        numpy.zeros((3, NbP), dtype=complex), numpy.zeros((3, NbP), dtype=complex), P
    )

    # for all observation point
    # PB = EMpy.utils.ProgressBar()
    for ind in numpy.arange(NbP):
        # distance between scattering and observation point
        QP = P[:, ind : ind + 1] - Q
        r = numpy.sqrt(numpy.sum(QP**2, axis=0))
        mask = r > 0
        if not numpy.any(mask):
            continue
        r_nz = r[mask]
        # unit vector
        r1 = QP[:, mask] / r_nz
        J_nz = J[:, mask]
        M_nz = M[:, mask]
        dS_nz = dS[mask]
        # integrand expression shortcuts
        kmr = km * r_nz
        kmr_1 = 1 / kmr
        kmr_2 = kmr_1**2
        aa = 1 - 1j * kmr_1 - kmr_2
        bb = -1 + 3 * 1j * kmr_1 + 3 * kmr_2
        cc = 1 - 1j * kmr_1
        # scalar product
        r1dotJ = matlab_dot(r1, J_nz)
        r1dotM = matlab_dot(r1, -M_nz)
        # Kottler EM fields
        EMF_P.E[:, ind] = (
            km
            / (4 * numpy.pi * 1j)
            * numpy.sum(
                (
                    Z0 / n_med * (aa * J_nz + bb * (r1dotJ * numpy.ones((3, 1))) * r1)
                    + cc * numpy.cross(r1, -M_nz, axis=0)
                )
                * numpy.exp(-1j * kmr)
                / r_nz
                * dS_nz,
                axis=1,
            )
        )
        EMF_P.H[:, ind] = (
            -km
            / (4 * numpy.pi * 1j)
            * numpy.sum(
                (
                    n_med / Z0 * (-aa * M_nz + bb * (r1dotM * numpy.ones((3, 1))) * r1)
                    - cc * numpy.cross(r1, J_nz, axis=0)
                )
                * numpy.exp(-1j * kmr)
                / r_nz
                * dS_nz,
                axis=1,
            )
        )
        # waitbar update
        # PB.update((ind+1)*100.0/NbP)

    return EMF_P
