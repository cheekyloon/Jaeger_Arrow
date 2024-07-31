#!/usr/bin/python3

# This script loads CTD data in .mat format

# import modules
import scipy.io

# directory path 
dirF    = '/Users/sandy/Documents/ISW_projects/Jaeger_Arrow/Instruments/'
# CTD file name
fileCTD = 'CTD_far_field.mat'

# load CTD mat file
matCTD  = scipy.io.loadmat(dirF + fileCTD)

# extract data
zF14      = matCTD['zb_F14'][0]
rhoF14    = matCTD['rhob_F14'][0]
zPMZA06   = matCTD['PMZA_jun_z'][0]
rhoPMZA06 = matCTD['PMZA_jun_rho'][0]

