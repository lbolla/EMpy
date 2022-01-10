from builtins import object
#-*- coding: UTF-8 -*-
import numpy
import EMpy_gpu

__author__ = 'Julien Hillairet'

class Field(object):
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

  def __init__(self,E=numpy.zeros((3,1)), H=numpy.zeros((3,1)), r=numpy.zeros((3,1))):
    """Initialize the Field object with an Electric, Magnetic and Position vectors."""
    self.E = E
    self.H = H
    self.r = r


""" 
##############################################################
utils functions for field manipulations
##############################################################
"""
def stack(X,Y,Z):  
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
  Nx = numpy.size(X,axis=0)
  Ny = numpy.size(Y,axis=0)
  Nz = numpy.size(Z,axis=0)
  
  XX = numpy.reshape(numpy.transpose(numpy.ones((Ny*Nz, 1)) * X), (1, Ny*Nz*Nx))
  YY = numpy.tile(numpy.reshape(numpy.transpose(numpy.ones((Nz, 1)) * Y), (1, Ny*Nz)), (1, Nx))
  ZZ = numpy.tile(Z, (1, Nx*Ny)) 
  
  return numpy.vstack((XX,YY,ZZ))


def matlab_dot(a,b):
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
  return numpy.sum(a*b,axis=0)


"""
########################################## 
Common functions for calculating EM fields
########################################## 
"""
def currentsScatteringKottler(P,J,M,Q,dS,f,epsr=1):
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
  # test if the parameters size are correct
  if numpy.size(P,axis=0)!=3 or \
     numpy.size(Q,axis=0)!=3 or \
     numpy.size(J,axis=0)!=3 or \
     numpy.size(M,axis=0)!=3:
    EMpy_gpu.utils.warning('Bad parameters size : number of rows must be 3 for vectors P,Q,J,M')
  
  if not numpy.size(Q,axis=1)==numpy.size(J,axis=1)==numpy.size(M,axis=1)==numpy.size(dS,axis=1):
    EMpy_gpu.utils.warning('Bad parameters size : number of columns between Q,J,M and dS must be equal')
  
  lambda0 = EMpy_gpu.constants.c/f  
  lambdam = lambda0/numpy.sqrt(epsr)
  #k0 = 2*numpy.pi/lambda0
  km = 2*numpy.pi/lambdam 
  Z0 = 120*numpy.pi
  NbP = numpy.size(P,axis=1)
  NbQ = numpy.size(Q,axis=1)
  
  # preallocation
  EMF_P = Field(numpy.zeros((3,NbP), dtype=complex), \
                numpy.zeros((3,NbP), dtype=complex), P)
  
  # for all observation point
  #PB = EMpy.utils.ProgressBar()
  for ind in numpy.arange(NbP):
    # distance between scattering and observation point
    QP = P[:,ind].reshape((3,1)) * numpy.ones((1,NbQ)) - Q
    r =  numpy.sqrt(numpy.sum(QP**2,axis=0))
    # unit vector
    r1 = QP/r
    # integrand expression shorcuts
    kmr = km*r
    kmr_1 = 1/kmr
    kmr_2 = kmr_1**2
    aa = 1 - 1j*kmr_1 - kmr_2
    bb = -1 + 3*1j*kmr_1 + 3*kmr_2
    cc = 1 - 1j*kmr_1
    # scalar product
    r1dotJ = matlab_dot(r1, J)
    r1dotM = matlab_dot(r1, -M)
    # Kottler EM fields
    EMF_P.E[:,ind] = km/(4*numpy.pi*1j) \
               * numpy.sum((Z0/numpy.sqrt(epsr)*(aa*J + bb*(r1dotJ*numpy.ones((3,1)))*r1) \
               + cc*numpy.cross(r1,-M,axis=0))*numpy.exp(-1j*km*r)/r*dS,axis=1)
    EMF_P.H[:,ind] = -km/(4*numpy.pi*1j) \
               * numpy.sum((numpy.sqrt(epsr)/Z0*(-aa*M + bb*(r1dotM*numpy.ones((3,1)))*r1) \
               - cc*numpy.cross(r1,J,axis=0))*numpy.exp(-1j*km*r)/r*dS,axis=1)
    # waitbar update
    #PB.update((ind+1)*100.0/NbP)
    
  return EMF_P
