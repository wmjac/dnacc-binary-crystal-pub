#!/usr/bin/env python3
# -*- coding: utf-8 -*-

###################################################################################
# This code is a modified interface to be used with /src/ available at
# github.com/smarbach/DNACoatedColloidsInteractions
# to calculate like and unlike particle interactions for binary colloidal crystals
# for the systems studied, with a feature to account for like-particle attractions
# Also generates LAMMPS compatible potential files
###################################################################################

import sys
sys.path.append('../src/')
from EnergyProfileCall import returnResultEnergy, noiseProfile
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline, splrep, BSpline, make_smoothing_spline

###############################################################################
# input parameters
###############################################################################

# colloid parameters

optcolloidcolloid = 1   # if colloid-colloid interaction set to 1
                        # else this is colloid-flat surface interaction
radius = float(sys.argv[1])*0.001            # radius of colloid in microns
PSdensity  = 1.055      # density (g/cm^3) of PS particle at Room T (22 C)
dilatationC = 1          # account for colloid material dilatation (should only be 1 if colloid is Polystyrene)
PSCharge  = -0.019      # charge (C/m^2) of (top) colloid

# general experimental parameters

peo_mw = float(sys.argv[2]) # PEO molecular weight in kDa
criticalHeight = 20         # HALF depth of focus (nm) or maximal height of colloids
                            # that determines wether they are bound or unbound
slabThick  = 250            # Total accessible height for colloids in microns
                            #(height of water slide)
saltConcentration  = 0.5  # Salt concentration in mol/L
optglassSurfaceFlag = 1     # 1 if surface is not covered with PS fine layer (for vdW calculation)
gravity  = 9.802            # gravity in NYC
cF127  = 0.5                # (w/v %) surfactant concentration (for effective increased length of brushes)
temperature = [float(sys.argv[3])]          # relevant temperatures for the energy profile calculation
fmix = float(sys.argv[4])   # Fraction of a* strands on A particles / a strands on B particles

print(sys.argv[1:])
# brush properties


tetherSeq  = "GTCTACC"#"ACCGCA"   # sticky DNA sequence
NDNAt  = 65             # TOTAL number of DNA bases on top surface (colloid), including sticky part
NDNAb  = 65             # TOTAL number of DNA bases on bottom surface
if (peo_mw==6.5):
    NPEOt = 148 	# No. of PEO units/monomers
elif (peo_mw==11):
    NPEOt = 250
elif (peo_mw==34):
    NPEOt = 772
elif (peo_mw==67):
    NPEOt = 1521
else:
    print("PEO weight not registered\n")
NPEOb  = NPEOt            # TOTAL number of polymer units (for example PEO units) on bottom
areat  = 1/(0.03)       # area (nm^2) available for 1 strand on top surface: 30,000 (\mu m)^-2 = 0.03 nm^-2
areab  = 1/(0.03)       # area (nm^2) available for 1 strand on bottom surface
ft_like = 2*fmix*(1-fmix)           # fraction of complimentary strands for A-A binding : = 0, if fmix = 0
ft_unlike = fmix**2 + (1-fmix)**2   # fraction of complimentary strands for A-B binding: = 1, if fmix = 0
fb_like = ft_like
fb_unlike = ft_unlike
DNAmodel = 'ssDNA'      # or 'dsDNA' according to the nature of the non-sticky part
persistencePEO  = 0.368 # persistence length (nm) of additional polymer (here PEO)
DNACharge  = 0          # charge per base of DNA, (real from 0 to 1). Leave to 0 to avoid 
                        # lengthy electrostatic calculations, especially at high salt concentration
                        # also these calculations are potentially unstable.

# model parameters

optdeplFlag = 0     # 1 to calculate depletion interactions
depletionTypeC = 'other' # 'F127' or 'other'
Ragg = 1            # specify agregation radius in nm if you want another type of depletion
optvdwFlag  = 2     # 1 to calculate van der Waals interactions
slideType = 'PS' # 'Glass' for Glass facing PS; otherwise 'PSonGlass' for PS (80nm) on Glass facing PS, or 'PS' for PS facing PS or 'other'
hamakerC = 3e-12    # in which case you need to specify the hamaker constant
mushroomFlag  = 0   # 1 if brush is low density, ~ mushroom brush, otherwise 0
porosity = 0.       # (real from 0 to 1) partial penetration of micelles in brush
#modelAccuracy = 0   # 1 if you want high model accuracy (long calculation - disabled for now)
wPEO = 0.0978       # polymer excluded volume (can be adjusted to obtain measured brush height)

optNoise  = 0#1               # apply a noise kernel to prediction
optShot  = 1                # shot noise (1) or gaussian kernel (0)
photonTarget  = 1000         # target photon number for shot noise
penetrationDepth  = 100     # penetration depth of the TIRM in nm
gaussianWidth  = 10         # gaussian kernel width in nm



###############################################################################
# calculate pair potentials
###############################################################################

if ((fmix != 0) and (fmix != 1)):
    result_like = returnResultEnergy(optcolloidcolloid,radius,criticalHeight,slabThick, \
                      saltConcentration,tetherSeq,NDNAt,NDNAb,NPEOt,NPEOb, \
                      areat,areab,ft_like,fb_like,persistencePEO,wPEO,DNACharge,PSCharge, \
                      optglassSurfaceFlag,PSdensity,gravity,optdeplFlag, \
                      cF127,optvdwFlag,mushroomFlag,porosity,DNAmodel,slideType,temperature, \
                          depletionType = depletionTypeC, aggRadius = Ragg, hamaker = hamakerC, \
                          dilatation = dilatationC)
result_unlike = returnResultEnergy(optcolloidcolloid,radius,criticalHeight,slabThick, \
                      saltConcentration,tetherSeq,NDNAt,NDNAb,NPEOt,NPEOb, \
                      areat,areab,ft_unlike,fb_unlike,persistencePEO,wPEO,DNACharge,PSCharge, \
                      optglassSurfaceFlag,PSdensity,gravity,optdeplFlag, \
                      cF127,optvdwFlag,mushroomFlag,porosity,DNAmodel,slideType,temperature, \
                          depletionType = depletionTypeC, aggRadius = Ragg, hamaker = hamakerC, \
                          dilatation = dilatationC)
print('The potential profile is calculated.')
    
    
    
if optNoise == 1:
    print('Now calculating the noise distortion, this may take some time')
    result = noiseProfile(result,optShot,photonTarget,penetrationDepth,gaussianWidth,temperature)
    
allheights =   result_unlike['allheights']
if ((fmix != 0) and (fmix != 1)):
    phiDna_like = result_like['phiBridge']
else:
    phiDna_like = np.zeros(len(allheights))

phiDna_unlike = result_unlike['phiBridge']
phiSter = result_unlike['phiSter']
phiVdW = result_unlike['phiVdW']


###############################################################################
# OUTPUT columns: <separation (metres)>, <A-B DNA binding (kBT)>, <A-A/A-B steric repulsion (kBT)>, <A-A/A-B VdW attraction (kBT)>, <A-A DNA binding (kBT)>
###############################################################################

for i in range(len(allheights)):
    f=open("full_pot_r_%dnm_mw_%gk_mix_%g_t_%g.txt" %(radius*1000, peo_mw, fmix, temperature[0]),"a+")
    f.write('{:.16e}'.format(allheights[i])+'\t'+'{:.16e}'.format(phiDna_unlike[i])+'\t'+'{:.16e}'.format(phiSter[i])+'\t'+'{:.16e}'.format(phiVdW[i])+'\t'+'{:.16e}'.format(phiDna_like[i])+'\n') 
    f.close()

###############################################################################################
# LAMMPS compatible potential files (Distance in lj units (sigma=2R), from centers of particles
###############################################################################################
# Rename potentials for convenience
unlike = np.zeros(len(allheights))
like = np.zeros(len(allheights))
for i in range(len(allheights)):
	unlike[i] = phiDna_unlike[i] + phiSter[i] + phiVdW[i]
	like[i] = phiDna_like[i] + phiSter[i] + phiVdW[i]
n = 900					 # Vary this based on distance at which potential is high enough to be irrelevant to simulations
h = allheights[-n:] 
h = np.insert(h,0,-2*radius*1e-6)        # extrapolate from min. separation to surface of colloid
unlike = unlike[-n:]
unlike = np.insert(unlike,0,20000)       # 20000 kBT was found high enough to be irrelevant to simulations and allow for smooth spline interpolation
like = like[-n:]
like = np.insert(like,0,20000)
hf = np.linspace(0.001,4,5000)		 # Grid spacing for LAMMPS potential, in units of sigma=2R
spl_unlike = CubicSpline((h*1e9/(2*radius*1000) + 1), unlike) # Spline fitting done in nm units for better numerical stability
spl_like = CubicSpline((h*1e9/(2*radius*1000) + 1), like)
pot_unlike = spl_unlike(hf)
pot_like = spl_like(hf)
force_unlike = -np.gradient(pot_unlike)
force_like = -np.gradient(pot_like)
sz = (len(hf),4)                         # LAMMPS potential file format: index, distance (units of sigma), potential, force
fit_like = np.zeros(sz)
fit_unlike = np.zeros(sz)
for i in range(len(hf)):
	fit_like[i,0] = i+1
	fit_unlike[i,0] = i+1
	fit_like[i,1] = hf[i]
	fit_unlike[i,1] = hf[i]
	fit_like[i,2] = pot_like[i]
	fit_unlike[i,2] = pot_unlike[i]
	fit_like[i,3] = force_like[i]
	fit_unlike[i,3] = force_unlike[i]
file_like = "lammps_pot_like_r_%d_mw_%g_mix_%g_t_%g.txt" %(radius*1000, peo_mw, fmix, temperature[0])
file_unlike = "lammps_pot_unlike_r_%d_mw_%g_mix_%g_t_%g.txt" %(radius*1000, peo_mw, fmix, temperature[0])
np.savetxt(file_like, fit_like, fmt=["%d","%.10f","%.10f","%.10f"], delimiter=' ', header = '#Like potential for r=%dnm, Mw=%gk, alpha=%g, T=%g C\n\nLike_pot\nN 5000 R 0.001 4\n\n' %(radius*1000,peo_mw,fmix,temperature[0]), comments='')
np.savetxt(file_unlike, fit_unlike, fmt=["%d","%.10f","%.10f","%.10f"], delimiter=' ', header = '#Unlike potential for r=%dnm, Mw=%gk, alpha=%g, T=%g C\n\nUnlike_pot\nN 5000 R 0.001 4\n\n' %(radius*1000,peo_mw,fmix,temperature[0]), comments='')
print("All done, good to run some simulations!\n")
