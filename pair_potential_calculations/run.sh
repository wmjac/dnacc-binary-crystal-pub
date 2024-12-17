#!/bin/bash

PS=300		# PS particle radius in nm
MW=6.5		# PEO molecular wright in kDa (only takes 6.5, 11, 34, 67)
T=25		# Temperature in celsius
mix=0.1		# Fraction of mixed strands

python3 binary_crystal_potentials.py $PS $MW $T $mix
