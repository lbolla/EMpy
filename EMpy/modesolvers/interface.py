from builtins import range
from builtins import object
import numpy

class ModeSolver(object):
    
    def solve(self, *argv):
        raise NotImplementedError()        

class Mode(object):
    
    def get_x(self, x=None, y=None):
        raise NotImplementedError()

    def get_y(self, x=None, y=None):
        raise NotImplementedError()
    
    def intensity(self, x=None, y=None):
        raise NotImplementedError()

    def TEfrac(self, x=None, y=None):
        raise NotImplementedError()

    def overlap(self, x=None, y=None):
        raise NotImplementedError()

    def get_fields_for_FDTD(self, x, y):
        raise NotImplementedError()

    def save_for_FDTD(self, mode_id='', x=None, y=None):
        """Save mode's fields on a staggered grid."""        
        Ex_FDTD, Ey_FDTD, Ez_FDTD, Hx_FDTD, Hy_FDTD, Hz_FDTD = self.get_fields_for_FDTD(x, y)
        Ex_FDTD.real.T.tofile('ex' + mode_id + '.dat', sep=' ')
        Ex_FDTD.imag.T.tofile('iex' + mode_id + '.dat', sep=' ')
        Ey_FDTD.real.T.tofile('ey' + mode_id + '.dat', sep=' ')
        Ey_FDTD.imag.T.tofile('iey' + mode_id + '.dat', sep=' ')
        Ez_FDTD.real.T.tofile('ez' + mode_id + '.dat', sep=' ')
        Ez_FDTD.imag.T.tofile('iez' + mode_id + '.dat', sep=' ')
        Hx_FDTD.real.T.tofile('hx' + mode_id + '.dat', sep=' ')
        Hx_FDTD.imag.T.tofile('ihx' + mode_id + '.dat', sep=' ')
        Hy_FDTD.real.T.tofile('hy' + mode_id + '.dat', sep=' ')
        Hy_FDTD.imag.T.tofile('ihy' + mode_id + '.dat', sep=' ')
        Hz_FDTD.real.T.tofile('hz' + mode_id + '.dat', sep=' ')
        Hz_FDTD.imag.T.tofile('ihz' + mode_id + '.dat', sep=' ')

    def plot(self, x=None, y=None):
        raise NotImplementedError()

def overlap(m1, m2, x=None, y=None):
    return m1.overlap(m2, x, y)

def interface_matrix(solver1, solver2, x=None, y=None):

    neigs = solver1.nmodes
    
    O11 = numpy.zeros((neigs, neigs), dtype=complex)
    O22 = numpy.zeros((neigs, neigs), dtype=complex)
    O12 = numpy.zeros((neigs, neigs), dtype=complex)
    O21 = numpy.zeros((neigs, neigs), dtype=complex)

    for i in range(neigs):
        for j in range(neigs):
            
            O11[i, j] = overlap(solver1.modes[i], solver1.modes[j], x, y)
            O22[i, j] = overlap(solver2.modes[i], solver2.modes[j], x, y)
            O12[i, j] = overlap(solver1.modes[i], solver2.modes[j], x, y)
            O21[i, j] = overlap(solver2.modes[i], solver1.modes[j], x, y)

    return (O11, O22, O12, O21)    

