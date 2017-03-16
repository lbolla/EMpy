# -*- coding: utf-8 -*-
"""
Transfer matrix example: Laser Etch Monitor/End-point detection.
Built for Electromagnetic Python by Lorenzo Bolla: http://lbolla.github.io/EMpy/

Simulate a laser etch monitor (aka. laser endpoint detection):
    Calculate the reflectivity of the stack as the stack is progressively 
    thinned by etching from one side. Finally, plot reflectivity vs. etch depth
    at a single wavelength, as is often used to determine end-points during 
    reactive ion/plasma etching.
    For more info, see Oxford Plasma's page on the technique:
        http://www.oxfordplasma.de/technols/li.htm
    
    Author: Demis D. John, March 2017
"""


''' Load some modules '''
##########################
import EMpy
import n_k as rix    # file of refractive indices, takes wavelengths in Microns!!

import pylab
import numpy
from copy import deepcopy   # to make copies of MultiLayer objects, instead of only references
import sys  # for progress bar (sys.stdout)



''' Simulation Parameters '''
#############################
wl_lasermon = 670e-9      # laser monitor wavelength
EtchStep = 10e-9          # how much to etch before acquiring new laser monitor reflectivity


# `PlotEachSim` option, where spectrum of each etch depth will be plotted
#PlotEachSim = False
wl_span = 50e-9   # plot span for full spectrum plot if `PlotEachSim`

wls = numpy.array([ wl_lasermon - 1e-9,  wl_lasermon,  wl_lasermon + 1e-9   ])



''' Define some helper functions '''
####################################

def pairs_to_Multilayer(layers, ReverseLayers=False):
    '''Convert List of (index,thickness) pairs into E.M.py Multilayer object.
    This allows you to use List math, such as addition, multiplication etc. for repeating layers.
    
    Parameters
    ----------
    `index` should be a function of one parameter, wavelength (meters), 
        eg. `lambda wl: 1.447 + 0.01/wl**2`

    `thickness` in meters.

    ReverseLayers: {True | False}
        Reverse order of layers?
    
    
    Examples
    --------
    > GaAs = lambda x: 3.51148 
    > AlGaAs95 = lambda x: n.AlGaAs(0.95, x)   # function from file `n.py`
    > air = lambda x: 1.0     # constant vs. wavelength

    # build from top to bottom - ie. topmost layer (closest to laser monitor) first. Substrate last.
    > layers =     \
    >     [ ( air, numpy.inf )  ] +     \
    >     [ (GaAs, 60e-9) , (AlGaAs95, 300e-9) ] * 5 +    \
    >     [ (GaAs, numpy.inf) ]
    
    > iso_layers = pairs_to_Multilayer( layers )
    '''
    if ReverseLayers: layers = layers.reverse()
    
    n, d = map(  numpy.array,  zip(*layers)  )    # unzip the layers into separate arrays for index & thickness

    EM_multilayer = EMpy.utils.Multilayer()
    
    for i in range(n.size):
        EM_multilayer.append(    
            EMpy.utils.Layer(
                EMpy.materials.IsotropicMaterial(
                    name='lyr %i'%(i),  # name for this material
                    n0=EMpy.materials.RefractiveIndex( n0_func=n[i] )
                ),
                d[i]
            )# layer
        )# append
    # end for(n)
    return EM_multilayer
#end pairs_to_Multilayer()



def get_multilayer_thickness( multilayer ):
    #import numpy as np
    t = []
    for l in multilayer:
        t.extend(   [l.thickness] if ( not numpy.isinf(l.thickness) ) else []   )
    #end for
    return numpy.sum(t)
#end get_multilayer_thickness()


def find_nearest(a, a0):
    "Return Element in ndArray `a` closest to the scalar value `a0`"
    idx = numpy.abs(a - a0).argmin()
    return a.flat[idx]

def arg_find_nearest(a, a0):
    "Return index to element in ndArray `a` with value closest to the scalar value `a0`"
    idx = numpy.abs(a - a0).argmin()
    return idx


def count_noninf(multilayer):
    ''' return number of non-infinite layers in a Multilayer'''
    out=0
    for x in multilayer:
        out = out+0 if numpy.isinf(x.thickness) else out+1
    return out
#end num_noninf()


def arg_inf(multilayer):
    ''' return index to infinite layers in a Multilayer'''
    out=[]
    for i,x in enumerate(multilayer):
        if numpy.isinf(x.thickness):
            out.append( i )
    return out
#end arg_inf()




def rix2losses(n, wl):
    """Return real(n), imag(n), alpha, alpha_cm1, alpha_dBcm1, given a
    complex refractive index.  Power goes as: P = P0 exp(-alpha*z)."""
    nr = numpy.real(n)
    ni = numpy.imag(n)
    alpha = 4 * numpy.pi * ni / wl
    alpha_cm1 = alpha / 100.
    alpha_dBcm1 = 10 * numpy.log10(numpy.exp(1)) * alpha_cm1

    return nr, ni, alpha, alpha_cm1, alpha_dBcm1


def loss_cm2rix(n, alpha_cm1):
    """Return complex refractive index, given real index (n) and absorption coefficient (alpha_cm1) in cm^-1."""
    nr = n.real()
    ni = 100* alpha_cm1 * wl /(numpy.pi * 4)
    return (nr + 1j*ni)


def loss_m2rix(n, alpha_m1):
    """Return complex refractive index, given real index (n) and absorption coefficient (alpha_m1) in m^-1."""
    nr = n.real()
    ni = alpha_m1 * wl /(numpy.pi * 4)
    return (nr + 1j*ni)




''' Define some materials '''
####################################

''' All materials use a function for refractive index, for loop simplicity. '''
air = lambda x: 1.0     # constant vs. wavelength

# convert RIX functions to microns!
GaAs = lambda x: (3.51148 - 0.151j)        # absorption = 11.5 cm^-1, not implemented
AlGaAs95 = lambda x: rix.AlGaAs(x, 0.95)    # Refractive index function from file
GaSb = lambda x: rix.GaSb( x * 1e6,  k=True )     # function from file, convert meters --> microns, request complex


# DBR mirror periods
wl_center = 1550e-9
dGaAs = wl_center / GaAs(wl_center).real / 4
dAlGaAs95 = wl_center / AlGaAs95(wl_center).real / 4





''' Create dictionaries for each set of layers & angles of incidence '''
# Build from top to bottom - ie. topmost layer (closest to laser monitor) first. Substrate last.
# Must include infinite-thickness layers on top & bottom for air & substrate, respectively.

layers0 = {}

# concatenating lists of (Index,Thickness) pairs to enable periodic structures.
#   Make sure to include infinite-thickness layers on ends
layers0['layers'] =     \
    [ ( air, numpy.inf ),  ] + \
    [ (GaAs, dGaAs) , (AlGaAs95, dAlGaAs95) ] * 3 + \
    [ (GaAs, 551.75e-9) ] + \
    [ (AlGaAs95, dAlGaAs95), (GaAs, dGaAs) ] * 3 + \
    [ (AlGaAs95, 300e-9) ] + \
    [ ( GaAs, 500e-9) ] + \
    [ ( GaAs, numpy.inf) ]

layers0['angle'] = 0    # angle in degrees
theta_inc = EMpy.utils.deg2rad( layers0['angle'] )  # incidence angle


# convert layers to E.M.py MultiLayer object
iso_layers = pairs_to_Multilayer( layers0['layers'] )

# get index to laser-mon wavelength within `wls` array
wlidx = arg_find_nearest(wls, wl_lasermon)

# setup etching loop
EtchStep_current = EtchStep
go = True
i = -1
layers0['etchedlayers'] = [ ]   # 1st stack is unetched
layers0['solution'] = [ ]       # Multilayer solution object
layers0['Rlaser'] = [ ]         # y-axis data
layers0['EtchSteps'] = [ ]      # x-axis data
layers0['rix'] = [ ]            # refractive index data



idxtemp = 0    # 0 if etching away from first layer in list, -1 if etching from last layer appended

while go is True:
    ''' keep reducing thickness/removing layers until last layer is too thin.'''
    i=i+1;
    
    sys.stdout.write('.'); sys.stdout.flush();	# print a small progress bar
    

    if count_noninf(iso_layers) > 0:
        '''at least one infinite-thickness layer should still be left'''

        if i<=0:    
            EtchStep_current = 0;   # analyze unetched structure first
            indexno = idxtemp
        else:
            while numpy.isinf(  iso_layers[idxtemp].thickness  ):
                idxtemp  = idxtemp + 1      # point to non-infinte layers
            #end while
            indexno = idxtemp
        #end if(i)
        
        if iso_layers[indexno].thickness < EtchStep_current:
            ''' next layer is thinner than etch increment'''
            EtchStep_current = EtchStep_current - iso_layers[indexno].thickness    
            iso_layers.pop( indexno )    # remove this layer
            # "   removed one layer, %i layers left"%(len(iso_layers))

        elif iso_layers[indexno].thickness > EtchStep_current:
            '''etch increment ends within next layer'''
            iso_layers[indexno].thickness = iso_layers[indexno].thickness - EtchStep_current
            layers0['etchedlayers'].append( deepcopy(iso_layers) )      # add this layer stack to the list
            if i <= 0:
                layers0['EtchSteps'].append( 0 )
            else:
                layers0['EtchSteps'].append( layers0['EtchSteps'][-1] + EtchStep )  # Add x-axis point
                EtchStep_current = EtchStep     # reset EtchStep_current
            layers0['rix'].append(  layers0['etchedlayers'][-1][idxtemp].mat.n(wl_lasermon).real  )  # get RefrIndex in this layer
            
            # "** reached one EtchStep. %i EtchSteps saved. Solving..."%len(layers0['etchedlayers'])
            
            # solve for reflectivity at laser monitor wavelength
            layers0['solution'].append(  
                EMpy.transfer_matrix.IsotropicTransferMatrix(  layers0['etchedlayers'][-1], theta_inc   ).solve(wls)  
            )#append
            layers0['Rlaser'].append( layers0['solution'][-1].Rs[wlidx] )
    else:
        go = False
#end while

print('\n')



## Plots:
fig2, [ax2,ax3] = pylab.subplots(nrows=2, ncols=1, sharex=True)
ax2.set_title( 'Reflectivity at $\lambda = %0.1fnm$'%(wls[wlidx] * 1e9)  )

# plot reflectivity vs. depth
ax3.plot(    numpy.array( layers0['EtchSteps'] ) * 1e9, 
             numpy.array( layers0['Rlaser'] ) * 100,
             '-',
        )
ax3.set_xlabel( 'Etch Depth (nm)' )
ax3.set_ylabel( 'Laser Reflectivity (%)' )
ax3.grid(True)


# plot refractive index vs. depth
ax2.plot(   numpy.array( layers0['EtchSteps'] ) * 1e9, 
            layers0['rix'],
            '-g',
        )
ax2.set_ylabel( 'Refractive Index' )
ax2.grid(True)

fig2.show()

