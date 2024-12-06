import os
import sys
import shutil
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import use
import flopy 
import pyemu

use('qt5agg')

if sys.platform.startswith('win'):
    bin_path = os.path.join("bin","win")
    
elif sys.platform.startswith('linux'):
    bin_path = os.path.join("bin","linux")


elif sys.platform.lower().startswith('dar') or sys.platform.lower().startswith('mac'):
    bin_path = os.path.join("bin","mac")
    
else:
    raise Exception('***ERROR: OPERATING SYSTEM UNKOWN***')


def build_flowmodel(new_d, delrc=10):
    if os.path.exists(new_d):
        shutil.rmtree(new_d)
    os.makedirs(new_d)
    # after moore and doherty 2005:
    # 500 m × 800 m rectangular domain of a single-layer
    # groundwater model of flow in a confined aquifer of
    # 10 m thickness. A fixed inflow of 0.1 m3 d−1 m−1
    # occurs through the upper boundary of the model;
    # heads are fixed at 0 m along the lower boundary.
    # A hydraulic conductivity field with a log average
    # value of zero was generated using a log exponential
    # variogram with a range (3 times the coefficient
    # in the exponent in the variogram equation) of
    # 600 m and a sill of 0.2. Diffuse recharge is zero.

    sim = flopy.mf6.MFSimulation(sim_name="model", exe_name="mf6", version="mf6", sim_ws=new_d,
                                 memory_print_option="ALL",continue_=True)

    nper = 1
    sp_len = 1000000
    tdis_pd = [(sp_len, 1, 1.0) for _ in range(nper)]
    tdis = flopy.mf6.ModflowTdis(sim, pname="tdis", time_units="DAYS", nper=nper, perioddata=tdis_pd)

    ims = flopy.mf6.ModflowIms(sim, pname="ims", complexity="SIMPLE",
                               inner_dvclose=1e-3, outer_dvclose=1e-3, outer_maximum=1000, inner_maximum=1000)

    model_nam_file = "model.nam"    
    newtonoptions = []
    gwf = flopy.mf6.ModflowGwf(sim, modelname="model", model_nam_file=model_nam_file, save_flows=True,newtonoptions=newtonoptions)

    top = 0
    botm = -10
    ylen = 800
    xlen = 500
    delr = delc = delrc
    assert ylen % delc == 0 # are these the right way round (not that it matters)
    assert xlen % delr == 0
    nrow = ylen // delc
    ncol = xlen // delr
    dis = flopy.mf6.ModflowGwfdis(gwf, nlay=1, nrow=nrow, ncol=ncol, delr=delr, delc=delc, 
    	                          top=top, botm=botm,idomain=1)

    wel_spd = [(0,0,j,0.1*delr) for j in range(ncol)]
    wel = flopy.mf6.ModflowGwfwel(gwf,stress_period_data={0:wel_spd})

    ghb_spd = [(0,nrow-1,j,0.0,100) for j in range(ncol)]
    ghb = flopy.mf6.ModflowGwfghb(gwf,stress_period_data={0:ghb_spd})

    headfile = "model.hds"
    head_filerecord = [headfile]
    budgetfile = "model.cbb"
    budget_filerecord = [budgetfile]
    saverecord = {kper:[("HEAD", "ALL"), ("BUDGET", "ALL")] for kper in range(nper)}
    printrecord = {kper:[("HEAD", "ALL")] for kper in range(nper)}
    oc = flopy.mf6.ModflowGwfoc(gwf, saverecord=saverecord, head_filerecord=head_filerecord,
                                budget_filerecord=budget_filerecord, printrecord=printrecord)

    ic = flopy.mf6.ModflowGwfic(gwf,strt=top)
    npf = flopy.mf6.ModflowGwfnpf(gwf, icelltype=0, k=1.0)
    sim.set_all_data_external()
    sim.write_simulation()

    for bin_name in os.listdir(bin_path):
        shutil.copy2(os.path.join(bin_path,bin_name),os.path.join(new_d,bin_name))

    pyemu.os_utils.run("mf6",cwd=new_d)


if __name__ == "__main__":
    build_flowmodel("base_model")