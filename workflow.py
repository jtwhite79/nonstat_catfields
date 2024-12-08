import os
import sys
import shutil
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import flopy 
import pyemu


if sys.platform.startswith('win'):
    bin_path = os.path.join("bin","win")
    
elif sys.platform.startswith('linux'):
    bin_path = os.path.join("bin","linux")


elif sys.platform.lower().startswith('dar') or sys.platform.lower().startswith('mac'):
    bin_path = os.path.join("bin","mac")
    
else:
    raise Exception('***ERROR: OPERATING SYSTEM UNKOWN***')


def build_flowmodel(new_d):
    if os.path.exists(new_d):
        shutil.rmtree(new_d)
    os.makedirs(new_d)

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
    nrow = 800
    ncol = 500
    delr = delc = 10.0

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


def build_ptmodel(org_d,new_d=None):

    if new_d is None:
        new_d = org_d+"_pt"
    if os.path.exists(new_d):
        shutil.rmtree(new_d)
    shutil.copytree(org_d,new_d)

    sim = flopy.mf6.MFSimulation.load(sim_ws=new_d)
    gwf = sim.get_model()

    nrow = gwf.dis.nrow.data
    ncol =gwf.dis.ncol.data
    nodes = []
    k = 0
    i = 100
    for j in range(0,ncol,20):
        nodes.append(k * nrow * ncol + i * ncol + j)


    # create modpath files
    mpnamf = "model_mp"

    # create basic forward tracking modpath simulation
    mp = flopy.modpath.Modpath7.create_mp7(
        modelname=mpnamf,
        trackdir="forward",
        flowmodel=gwf,
        model_ws=new_d,
        rowcelldivisions=1,
        columncelldivisions=1,
        layercelldivisions=1,
        nodes=nodes,
        exe_name="mp7",
    )

    # write modpath datasets
    mp.write_input()

    pyemu.os_utils.run("mp7 model_mp.mpsim",cwd=new_d)

    
def quick_domain_plot(ws):
    sim = flopy.mf6.MFSimulation.load(sim_ws=ws)
    gwf = sim.get_model()
    hds = flopy.utils.HeadFile(os.path.join(ws,"model.hds"),model=gwf)
    totim = np.cumsum(sim.tdis.perioddata.array["perlen"])
    df = load_pathline(os.path.join(ws,'model_mp.mppth'),totim)
    print(df)

    x = gwf.modelgrid.xcellcenters
    y = gwf.modelgrid.ycellcenters
    fig,ax = plt.subplots(1,1,figsize=(6,6))
    ax.set_aspect("equal")
    ax.pcolormesh(x,y,gwf.npf.k.array[0,:,:])
    hdsarr = hds.get_data()[0,:,:]
    c = ax.contour(x,y,hdsarr,levels=20,colors="w",linestyles="--")
    ax.clabel(c)
    pids = df.particleid.unique()
    pids.sort()
    for pid in pids:
        pdf = df.loc[df.particleid==pid,:].copy()
        pdf.sort_values(by="ptime",inplace=True)
        ax.plot(pdf.x,pdf.y,"k--")

    plt.savefig(os.path.join(ws,"quick_domain_plot.pdf"))
    plt.close(fig)


def load_pathline(pathline_fname,totim):
    dfs = []
    pids = []
    with open(pathline_fname, 'r') as f:
        for _ in range(3):
            f.readline()
        while True:
            line = f.readline()
            if line == "":
                break

            hraw = line.strip().split()
            pid = int(hraw[2])
            nrec = int(hraw[3])
            # print(pid)
            data = {"particleid": [], "ptime": [], "x": [], "y": [], "layer": []}
            for irec in range(nrec):
                line = f.readline()
                if line == "":
                    raise EOFError()
                line = line.lower()
                raw = line.strip().split()
                for i,r in enumerate(raw):
                    if '-' in r[1:] and 'e' not in r:
                        raw[i] = r.replace("-","e-")
                x = float(raw[1])
                y = float(raw[2])
                z = float(raw[3])
                t = float(raw[4])
                layer = int(raw[8])
                data["x"].append(x)
                data["y"].append(y)
                data["particleid"].append(pid)
                data["layer"].append(layer)
                data["ptime"].append(t)
            df1 = pd.DataFrame(data, index=data["ptime"])

            df1.drop_duplicates(inplace=True, subset="ptime")
            # print(df)
            pids.append(pid)
            df = df1#.reindex(totim, fill_value=np.nan)
            # if df.shape[0] != totim.shape[0]:
            #print(pid)

            df.loc[:, "particleid"] = pid
            #df.loc[:, "ptime"] = totim
            dfs.append(df.copy())

    dfmp7 = pd.concat(dfs, axis=0, ignore_index=True)
    #dfmp7.loc[:,"status"] = dfmp7.particleid.apply(lambda x: status[x])
    return dfmp7



if __name__ == "__main__":
    build_flowmodel("base_model")
    build_ptmodel("base_model")
    quick_domain_plot("base_model_pt")