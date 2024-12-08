import os
import sys
import shutil
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import flopy 
import pyemu

is_arm = False
if sys.platform.startswith('win'):
    bin_path = os.path.join("bin","win")
    
elif sys.platform.startswith('linux'):
    bin_path = os.path.join("bin","linux")


elif sys.platform.lower().startswith('dar') or sys.platform.lower().startswith('mac'):
    bin_path = os.path.join("bin","mac")
    is_arm = True
    
else:
    raise Exception('***ERROR: OPERATING SYSTEM UNKOWN***')


def build_flowmodel(new_d):
    if os.path.exists(new_d):
        shutil.rmtree(new_d)
    os.makedirs(new_d)

    sim = flopy.mf6.MFSimulation(sim_name="model", exe_name="mf6", version="mf6", sim_ws=new_d,
                                 memory_print_option="ALL",continue_=True)

    nper = 1
    sp_len = 1000000000
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
    df = load_pathline(os.path.join(ws,'model_mp.mppth'))

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
        #print(pdf.iloc[-1,:])

    plt.savefig(os.path.join(ws,"quick_domain_plot.pdf"))
    plt.close(fig)


def get_final_particle_info(pathline_fname):
    if isinstance(pathline_fname,str):
        df = load_pathline(pathline_fname)
    else:
        df = pathline_fname
    pids = df.particleid.unique()
    pids.sort()
    final_time,final_x = [],[]
    for pid in pids:
        pdf = df.loc[df.particleid==pid,:].copy()
        pdf.sort_values(by="ptime",inplace=True)
        pdf = pdf.iloc[-1,:]
        final_time.append(pdf.ptime)
        final_x.append(pdf.x)
    results = pd.DataFrame({"ptime":final_time,"x":final_x},index=pids)
    results.index.name = "pid"
    return results


def load_pathline(pathline_fname="model_mp.mppth"):
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
            df = df1
            df.loc[:, "particleid"] = pid
            dfs.append(df.copy())
    dfmp7 = pd.concat(dfs, axis=0, ignore_index=True)
    dfmp7["particleid"] = dfmp7.particleid.astype(int)
    #dfmp7.loc[:,"status"] = dfmp7.particleid.apply(lambda x: status[x])
    dfmp7.index.name = "count"
    return dfmp7


def interp_pathline_to_consistent_nobs(pathline_fname):
    if isinstance(pathline_fname,str):
        df = load_pathline(pathline_fname)
    else:
        df = pathline_fname
    results = get_final_particle_info(df)
    pids = df.particleid.unique()
    pids.sort()
    reidx = []
    for pid in pids:
        pdf = df.loc[df.particleid==pid,:].copy()
        maxtime = pdf.loc[pdf.y!=0,'ptime'].max()
        times = np.linspace(0,maxtime,100)
        pdf.index = pdf.ptime
        rdf = pdf.reindex(times,method="nearest")
        rdf["particleid"] = pdf.particleid.values[0]
        reidx.append(rdf)
    reidx = pd.concat(reidx)
    reidx.pop("ptime")
    return reidx


def post_process_run(ws="."):
    hds = flopy.utils.HeadFile(os.path.join(ws,"model.hds"))
    arr = hds.get_data()[0,:,:]
    np.savetxt(os.path.join(ws,"heads.dat"),arr,fmt="%15.6E")
    karr = np.loadtxt(os.path.join(ws,"model.npf_k.txt"))
    ivals = [199,399,599,749]
    jvals = [74,249,424]
    names, hvals, kvals = [],[],[]
    for ival in ivals:
        for jval in jvals:
            name = "result_i:{0}_j:{1}".format(ival,jval)
            names.append(name)
            hvals.append(arr[ival,jval])
            kvals.append(karr[ival,jval])
    hdf = pd.DataFrame({"hname":names,"hval":hvals,"kval":kvals})
    hdf.to_csv(os.path.join(ws,"heads.csv"),index=False)
    pathdf = load_pathline(os.path.join(ws,'model_mp.mppth'))
    resultsdf = get_final_particle_info(pathdf)
    reidx = interp_pathline_to_consistent_nobs(pathdf)
    reidx.to_csv(os.path.join(ws,"pathline_consistent.csv"))
    pathdf.to_csv(os.path.join(ws,"pathline.csv"))
    resultsdf.to_csv(os.path.join(ws,"endpoint.csv"))
    return arr,hdf,pathdf,resultsdf


def setup_pst(org_d,include_all_outputs=False):
    temp_d = 'temp'
    if os.path.exists(os.path.join(temp_d)):
        shutil.rmtree(temp_d)
    shutil.copytree(org_d,temp_d)
    shutil.copytree("pyemu",os.path.join(temp_d,"pyemu"))
    if is_arm:
        ppu_d = "pypestutils"
        if os.path.exists(ppu_d):
            shutil.rmtree(ppu_d)
        shutil.copytree(os.path.join("..",ppu_d,ppu_d),ppu_d)
        shutil.copytree(os.path.join("..",ppu_d,ppu_d),os.path.join(temp_d,ppu_d))
        import sys
        sys.path.insert(0,".")
    import pypestutils as ppu
        
    sim = flopy.mf6.MFSimulation.load(sim_ws=temp_d)
    gwf = sim.get_model()
    karr_fname = "model.npf_k.txt"
    arr = np.loadtxt(os.path.join(temp_d,karr_fname)).reshape(gwf.dis.nrow.data,gwf.dis.ncol.data)
    np.savetxt(os.path.join(temp_d,karr_fname),arr,fmt="%15.6E")
    pyemu.os_utils.run("mf6",cwd=temp_d)
    pyemu.os_utils.run("mp7 model_mp.mpsim",cwd=temp_d)
    arr,hdf,pathdf,resultsdf = post_process_run(ws=temp_d)
    
    pf = pyemu.utils.PstFrom(temp_d,"template",spatial_reference=gwf.modelgrid,remove_existing=True)

    pf.extra_py_imports.append("shutil")
    pf.extra_py_imports.append("os")
    pf.extra_py_imports.append("flopy")

    pf.mod_sys_cmds.append("mf6")
    pf.mod_sys_cmds.append("mp7 model_mp.mpsim")

    pf.add_py_function("workflow.py","interp_pathline_to_consistent_nobs(ws='.')",is_pre_cmd=None)
    pf.add_py_function("workflow.py","get_final_particle_info(pathdf)",is_pre_cmd=None)
    pf.add_py_function("workflow.py","load_pathline(pathdf)",is_pre_cmd=None)
    pf.add_py_function("workflow.py","post_process_run()",is_pre_cmd=False)
    
    pf.add_observations("pathline_consistent.csv",index_cols=["particleid","ptime"],
                        use_cols=["x","y"],prefix="pathline",
                        obsgp=["pathlinex","pathliney"])
    pf.add_observations("endpoint.csv",index_cols=["pid"],
                        use_cols=["ptime","x"],prefix="endpoint",
                        obsgp=["endpointptime","endpointx"])
    if include_all_outputs:
        pf.add_observations("heads.dat",prefix="head",obsgp="head")
    pf.add_observations("heads.csv",index_cols="hname",use_cols=["hval","kval"],prefix="result",obsgp=["head","k"])


    pf.add_observations(karr_fname,prefix="karr",obsgp="karr")

    value_v = pyemu.geostats.ExpVario(contribution=1, a=200, anisotropy=5, bearing=0.0)
    value_gs = pyemu.geostats.GeoStruct(variograms=value_v)
    bearing_v = pyemu.geostats.ExpVario(contribution=1,a=1000,anisotropy=3,bearing=90.0)
    bearing_gs = pyemu.geostats.GeoStruct(variograms=bearing_v)
    aniso_v = pyemu.geostats.ExpVario(contribution=1, a=1000, anisotropy=3, bearing=45.0)
    aniso_gs = pyemu.geostats.GeoStruct(variograms=aniso_v)

    pf.add_parameters(karr_fname,par_type="pilotpoints",geostruct=value_gs,pargp="ppvalue",par_name_base="ppvalue",
                      upper_bound=10.0,lower_bound=0.1,apply_order=2,
                      pp_options={"pp_space":30,"prep_hyperpars":True,"try_use_ppu":True})

    tag = "pp"
    tfiles = [f for f in os.listdir(pf.new_d) if f.startswith(tag)]
    afile = [f for f in tfiles if "aniso" in f][0]
    pf.add_parameters(afile,par_type="constant",par_name_base=tag+"aniso",
                      pargp=tag+"aniso",lower_bound=-1.0,upper_bound=1.0,
                      apply_order=1,
                      par_style="a",transform="none",initial_value=0.0)
    if include_all_outputs:
        pf.add_observations(afile, prefix=tag+"aniso", obsgp=tag+"aniso")
    
    bfile = [f for f in tfiles if "bearing" in f][0]
    pf.add_parameters(bfile, par_type="pilotpoints", par_name_base=tag + "bearing",
                      pargp=tag + "bearing", lower_bound=-45,upper_bound=45,
                      par_style="a",transform="none",
                      pp_options={"try_use_ppu":True,"pp_space":30},
                      apply_order=1,geostruct=bearing_gs)
    if include_all_outputs:
        pf.add_observations(bfile, prefix=tag + "bearing", obsgp=tag + "bearing")

    pst = pf.build_pst(version=None)
    pst.control_data.noptmax = 0
    pst.write(os.path.join(pf.new_d,"pest.pst"),version=2)
    pyemu.os_utils.run("pestpp-ies pest.pst",cwd=pf.new_d)

    pe = pf.draw(num_reals=1000)
    pe.enforce()
    pe.to_binary(os.path.join(pf.new_d,"prior.jcb"))
    pst.pestpp_options["ies_par_en"] = "prior.jcb"
    pst.pestpp_options["ies_save_binary"] = True
    pst.pestpp_options["ies_ordered_binary"] = False
    pst.control_data.noptmax = -2
    obs = pst.observation_data
    obs.loc[:,"weight"] = 0
    obs.loc[obs.oname=="result","weight"] = 1.0
    assert pst.nnz_obs > 0
    pst.write(os.path.join(pf.new_d,"pest.pst"),version=2)
    pyemu.os_utils.run("pestpp-ies pest.pst",cwd=pf.new_d)




if __name__ == "__main__":
    #build_flowmodel("base_model")
    #build_ptmodel("base_model")
    #quick_domain_plot("base_model_pt")
    #post_process_run("base_model_pt")
    setup_pst("base_model_pt")
    #interp_pathline_to_consistent_nobs(os.path.join("temp",'model_mp.mppth'))