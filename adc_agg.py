

import MDAnalysis as mda
import numpy as np
import pandas as pd
from rdkit import Chem
from multiprocessing import cpu_count
from joblib import Parallel, delayed, wrap_non_picklable_objects
import scipy.spatial
import os
import subprocess
import math
from matplotlib import pyplot as plt


import logging
logging.basicConfig(format='[%(levelname)s] %(message)s', \
                    level=logging.INFO)

logging.getLogger('MDAnalysis').setLevel(logging.ERROR)


class grid():
    def __init__(self, **args):
        #self.mol = args["rdmol_obj"]
        self.xyz = args["xyz"]
        self.vdw = args["vdw"]
        self.probe_radius = args["margin"]
        #self.method = args["sample_method"]

        self.nMC = args["nMC"]
        
        self.dimension = np.pi * 4 * self.nMC ** 3

        try:
            self.precision = args["precision"]
        except Exception as e:
            self.precision = 1e-3
        
        try:
            self.n_thread = args["n_thread"]
        except Exception as e:
            self.n_thread = cpu_count()
        
        self.n_layer = max(int(self.nMC * self.probe_radius), 1)

        self.layer_thickness = self.probe_radius / self.n_layer
        self.serial_radius = [i * self.layer_thickness for i in range(1, self.n_layer+1)]

        self.gloden_ratio = (1 + 5 ** 0.5) / 2

        try:
            self.generate_space = args["xyz_idx_list"]
        except Exception as e:
            self.generate_space = [ii for ii in range(self.xyz.shape[0])]

    def dotsphere(self, dimension):
        i = np.arange(0, dimension)
        theta = 2 * np.pi * i /self.gloden_ratio
        phi = np.arccos(1 - 2 * (i+0.5)/ dimension)

        x, y, z = (
            np.cos(theta) * np.sin(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(phi),
        )
        return np.array([x, y, z]).T
    
    def sample_surface(self, probe_radius):
        
        sample_dots = np.array([])
        for i, coor in enumerate(self.xyz):
            if i in self.generate_space:
                anti_coor = np.delete(self.xyz, i, axis=0)
                anti_vdw = np.delete(self.vdw, i, axis=0)
                sample_size = int((self.vdw[i][0] + probe_radius) * self.dimension)
                raw_dots = self.dotsphere(sample_size) * (probe_radius + self.vdw[i][0]) + coor

                if not anti_coor.size:
                    save_dots = raw_dots
                else:
                    dis = scipy.spatial.distance.cdist(raw_dots, anti_coor, metric='euclidean')
                    save_dots = raw_dots[np.min(dis, axis=1) > (anti_vdw[np.argmin(dis, axis=1)].flatten() + probe_radius - self.precision)]

                if not sample_dots.shape[0]:
                    sample_dots = save_dots
                else:
                    sample_dots = np.vstack((sample_dots, save_dots))
                
        return sample_dots
    
    def run(self):
        return self.sample_surface(self.probe_radius)


class adc_agg():
    def __init__(self, **args):
        self.md_traj = args["md_traj"]
        self.md_tpr = args["md_tpr"]

        self.focuse = args["focuse"]

        self.trj_type = self.md_traj.split(".")[-1]

        try:
            self.polar_cutoff = float(args["polar_cutoff"])
        except Exception as e:
            self.polar_cutoff = 0.3

        try:
            self.hit_scope = float(args["hit_scope"])
        except Exception as e:
            self.hit_scope = 4.0
            
        
        try:
            self.terminal_residx = args["ter_res_idx"]
        except Exception as e:
            self.terminal_residx = 1359

        try:
            self.nMC = int(args["nMC"])
        except Exception as e:
            self.nMC = 3
        
        try:
            self.margin = float(args["margin"])
        except Exception as e:
            self.margin = 0
        

        try:
            self.md_frame = int(args["md_frame"])
        except Exception as e:
            self.md_frame = 500
        
        #try:
        #    self.md_length = args["md_length_ns"]
        #except Exception as e:
        #    self.md_length = 10
        
        #self.md_step = int(self.md_length * 1000 / self.md_frame)

        self.n_thread = cpu_count()

        self.md_real_frame = self.md_frame + 1

        if self.n_thread > 1:
            self.n_in_thread = math.ceil(self.md_real_frame / self.n_thread)

            if self.n_in_thread == math.ceil(self.md_real_frame / self.n_in_thread):
                self.n_in_thread += 1
        else:
            self.n_in_thread = self.md_real_frame
            

        #self.u = mda.Universe(self.md_tpr, self.md_traj)

    
    def trj_ana(self, input_traj: str) -> list:
        #idx, input_traj = trj_dic_item
        #index_shift = idx * self.n_in_thread
        # single frame

        #if input_traj == self.md_traj:
        #    index_shift = 0
        #else:
        #    b = int(input_traj.split("/")[-1].split(".")[0].split("-")[-1].split("_")[0])
        #    index_shift = int(b/self.md_step)

        assemble = []

        #u = mda.Universe(self.md_tpr, self.md_traj)
        u = mda.Universe(self.md_tpr, input_traj)

        LP = u.select_atoms("resname L31 or resname L32 or resname L33 or resname L34 or resname L35 or resname L36 or resname L37 or resname L38")

        #self.pre_Ab = self.u.select_atoms(f"resid 1-{self.terminal_residx} and around {self.hit_scope} group LP", LP=self.LP)
        pre_Ab = u.select_atoms(f"resid 1-{self.terminal_residx} and around {self.hit_scope} group LP", LP=LP)
        Ab = pre_Ab.residues

        for ts in u.trajectory:
            xyz_LP = []
            charge_LP = []
            vdw_LP = []

            for aa in LP.atoms:
                xyz_LP.append(list(aa.position))
                vdw_LP.append(Chem.GetPeriodicTable().GetRvdw(Chem.GetPeriodicTable().GetAtomicNumber(aa.name[0])))
                #atomicidx.append(atom.index+1)
                charge_LP.append(aa.charge)

            get_LP_xyz = np.array(xyz_LP)
            get_LP_vdw = np.array(vdw_LP).reshape(-1,1)
            get_LP_charge = np.array(charge_LP).reshape(-1,1)

            #print(get_LP_xyz)

            #assembled_res = []

            #for aa in self.pre_Ab.atoms:
            #    assembled_res.append(aa.resid)
            
            xyz_ab = []
            vdw_ab = []
            #atomicidx = []
            charge_ab = []

            for atom in Ab.atoms:
                xyz_ab.append(list(atom.position))
                vdw_ab.append(Chem.GetPeriodicTable().GetRvdw(Chem.GetPeriodicTable().GetAtomicNumber(atom.name[0])))
                charge_ab.append(atom.charge)
            
            #for atom in self.u.atoms:
            #    if atom.resid in list(set(assembled_res)):
            #        xyz_ab.append(list(atom.position))
            #        vdw_ab.append(Chem.GetPeriodicTable().GetRvdw(Chem.GetPeriodicTable().GetAtomicNumber(atom.name[0])))
            #        #atomicidx.append(atom.index+1)
            #        charge_ab.append(atom.charge)
        

            get_ab_xyz = np.array(xyz_ab)
            get_ab_vdw = np.array(vdw_ab).reshape(-1,1)
            get_ab_charge = np.array(charge_ab).reshape(-1,1)

            region = {"LP": {"xyz": get_LP_xyz,
                        "vdw": get_LP_vdw,
                        "charge": get_LP_charge},
                      "Ab": {"xyz": get_ab_xyz,
                        "vdw": get_ab_vdw,
                        "charge": get_ab_charge}}

            try:
                focuse = region[self.focuse]
            except Exception as e:
                return None
            
            surrounding_key = [kk for kk in region.keys() if kk != self.focuse][0]
            surrounding = region[surrounding_key]

            PolarIdx = np.where((focuse["charge"]>=self.polar_cutoff) | (focuse["charge"] <= self.polar_cutoff * (-0.1)))[0]
            NonPolarIdx = np.where((focuse["charge"] < self.polar_cutoff) & (focuse["charge"] > self.polar_cutoff * (-0.1)))[0]

            sys_xyz = np.vstack((focuse["xyz"], surrounding["xyz"]))
            sys_vdw = np.vstack((focuse["vdw"], surrounding["vdw"]))

            #print(sys_xyz)

            assemble.append({
                "frame_idx": ts.frame,
                "sys_xyz": sys_xyz,
                "sys_vdw": sys_vdw,
                "focuse_xyz": focuse["xyz"],
                "focuse_vdw": focuse["vdw"],
                "PolarIdx": PolarIdx,
                "NonPolarIdx": NonPolarIdx
            })



            #get_df = self.calc_shade(frame_idx=ts.frame + index_shift,
            #                        sys_xyz=sys_xyz,
            #                        sys_vdw=sys_vdw, 
            #                        focuse_xyz=focuse["xyz"],
            #                        focuse_vdw=focuse["vdw"],
            #                        PolarIdx=PolarIdx,
            #                        NonPolarIdx=NonPolarIdx)

            #assemble.append(get_df)

        return assemble
    

    def calc_shade(self,
                   frame_idx: int,
                   sys_xyz: object,
                   sys_vdw: object, 
                   focuse_xyz: object,
                   focuse_vdw: object,
                   PolarIdx: list,
                   NonPolarIdx: list) -> object:
        
        polar_region_in_sys = grid(xyz=sys_xyz, 
                                    vdw=sys_vdw, 
                                    margin=self.margin,
                                    nMC=self.nMC,
                                    xyz_idx_list=list(PolarIdx)).run()

        nonpolar_region_in_sys = grid(xyz=sys_xyz, 
                                        vdw=sys_vdw, 
                                        margin=self.margin,
                                        nMC=self.nMC,
                                        xyz_idx_list=list(NonPolarIdx)).run()
        
        polar_region_in_solo = grid(xyz=focuse_xyz, 
                                    vdw=focuse_vdw, 
                                    margin=self.margin,
                                    nMC=self.nMC,
                                    xyz_idx_list=list(PolarIdx)).run()

        nonpolar_region_in_solo = grid(xyz=focuse_xyz, 
                                    vdw=focuse_vdw, 
                                    margin=self.margin,
                                    nMC=self.nMC,
                                    xyz_idx_list=list(NonPolarIdx)).run()
        
        df = pd.DataFrame({"idx": [frame_idx],
                           "polar_mol_in_sys": [len(polar_region_in_sys)],
                           "nonpolar_mol_in_sys": [len(nonpolar_region_in_sys)],
                           "polar_mol": [len(polar_region_in_solo)],
                           "nonpolar_mol": [len(nonpolar_region_in_solo)]})

        return df

    def split_trj(self) -> list:
        if self.n_thread == 1:
            return [f"{self.md_traj}"]
        
        col = []
            
        for i in range(self.n_thread):
            begin = i * self.n_in_thread * self.md_step
            end = min(((i + 1) * self.n_in_thread - 1) * self.md_step, int(self.md_length * 1000))

            command = f"echo -e 'System\n' | gmx trjconv -f {self.md_traj} -s {self.md_tpr} -o TEMPTRJ_{i}-{begin}_{end}.{self.trj_type} -b {begin} -e {end}"
            
            (status, output) = subprocess.getstatusoutput(command)

            if status == 0 and os.path.getsize(f"TEMPTRJ_{i}-{begin}_{end}.{self.trj_type}"):
                col.append(f"TEMPTRJ_{i}-{begin}_{end}.{self.trj_type}")
            else:
                logging.info(f"Parallel spliting failed at {i}th traj")
            
        return col
    
    def run_serial(self, assembled_info_in_list: list) -> list:
        _col = []

        for item in assembled_info_in_list:
            get_df = self.calc_shade(frame_idx=item["frame_idx"],
                                     sys_xyz=item["sys_xyz"],
                                     sys_vdw=item["sys_vdw"], 
                                     focuse_xyz=item["focuse_xyz"],
                                     focuse_vdw=item["focuse_vdw"],
                                     PolarIdx=item["PolarIdx"],
                                     NonPolarIdx=item["NonPolarIdx"])
            _col.append(get_df)
        
        return _col
    
    def run(self) -> object:
        assembled_traj_info = self.trj_ana(self.md_traj)

        if len(assembled_traj_info) != self.md_real_frame:
            logging.info(f"Readout info size: {len(assembled_traj_info)} not equal to md_frame: {self.md_real_frame}")
            logging.info("Terminated")
            return pd.DataFrame()
        
        sheet = []
        
        if self.n_thread == 1:
            sheet = self.run_serial(assembled_traj_info)
        
        else:
            _col = Parallel(n_jobs=self.n_thread)(delayed(self.run_serial)(assembled_traj_info[ii*self.n_in_thread: (ii+1)*self.n_in_thread])\
                                           for ii in range(self.n_thread))
            
            for each in _col:
                sheet += each
        
        df = pd.concat(sheet)

        df["delta_np"] = df.apply(lambda x: x["nonpolar_mol_in_sys"] - x["nonpolar_mol"], axis=1)
        df["delta_p"] = df.apply(lambda x: x["polar_mol_in_sys"] - x["polar_mol"], axis=1)
        df["shade:sas"] = df.apply(lambda x: (x["polar_mol"] + x["nonpolar_mol"]) - (x["nonpolar_mol_in_sys"] + x["polar_mol_in_sys"]) , \
                                    axis=1)

        df["ratio:delta_np"] = df.apply(lambda x: abs(x["delta_np"]) / (abs(x["delta_p"]) + abs(x["delta_np"])), axis=1)
        df["ratio:delta_p"] = df.apply(lambda x: abs(x["delta_p"]) / (abs(x["delta_p"]) + abs(x["delta_np"])), axis=1)
        df["ratio-delta_p:ratio-delta_np"] = df.apply(lambda x: x["ratio:delta_p"] / x["ratio:delta_np"], axis=1)

        return df
    
def output(data_dic: dict, save_dir: str, if_direct_save: bool) -> object:
    
    df_saver = {}

    show_types = ["shade:sas", "ratio-delta_p:ratio-delta_np", "delta_np", "delta_p", "ratio:delta_np", "ratio:delta_p"]

    cc_map = ["orange", "blue", "green", "green", "green", "green"]

    ss_idx = [321, 322, 323, 324, 325, 326]

    titles = ["Shield surface in total",
            r"Ratio of $\Delta_{Polar Surface}$ vs $\Delta_{Non Polar Surface}$",
            r"$\Delta_{Non Polar Surface}$",
            r"$\Delta_{Polar Surface}$",
            r"Ratio of $\Delta_{Non Polar Surface}$",
            r"Ratio of $\Delta_{Polar Surface}$ "]
    ylabels = ["Sheilded surface sas",
            r"$\Delta_{Polar Surface}$ / $\Delta_{Non Polar Surface}$",
            r"$\Delta_{Non Polar Surface}$",
            r"$\Delta_{Polar Surface}$",
            r"$\Delta_{Non Polar Surface}$ / ($\Delta_{Non Polar Surface}$ +  $\Delta_{Polar Surface}$)",
            r"$\Delta_{Polar Surface}$ / ($\Delta_{Non Polar Surface}$ +  $\Delta_{Polar Surface}$)"]

    #f, axes = plt.subplots(3, 2, figsize=(12,8),tight_layout=True)
    f = plt.figure(figsize=(15, 9), tight_layout=True)

    for idx, tt in enumerate(show_types):
        _df = pd.DataFrame({"sys": [kk for kk in data_dic.keys()],
                f"{tt}:avg": [vv[tt].mean() for vv in data_dic.values()],
                f"{tt}:std": [vv[tt].std() for vv in data_dic.values()]})

        df_saver.setdefault(tt, _df)

        ax = f.add_subplot(ss_idx[idx])

        _df.plot.bar(x="sys", 
                    y=f"{tt}:avg",
                    rot=30, 
                    legend=False, 
                    color=cc_map[idx],
                    ylabel=f"{ylabels[idx]}",
                    ax=ax)
        
        ax.title.set_text(f"{titles[idx]}")

        for p in ax.patches:
            ax.annotate(f"{p.get_height():.3f}",
                                (p.get_x() * 1.005, p.get_height() * 1.005))
    
    if if_direct_save:
        for kk, df in df_saver.items():
            df.to_csv(os.path.join(save_dir, f"RESULT_Stat_{kk}.csv"), index=None)
        
        f.savefig(os.path.join(save_dir, "RESULT_show.png"))
    else:
        return f, df_saver


def checker(xtc_list: list, 
            tpr_list: list,
            ) -> object:
    
    if len(xtc_list) != len(tpr_list):
        logging.info("Input sample size for trj and tpr not met")
        logging.info("Abort")
        return None
    
    paired = {}

    #for prefix, trj in xtc_dict.items():
    #    try:
    #        get_topo = tpr_dict[prefix]
    #    except Exception as e:
    #        logging.info(f"Can not find topo file for system {prefix}")
    #        logging.info("Abort")
        
    #    paired.setdefault(prefix, [trj, get_topo])  

    for ii, trj in enumerate(xtc_list):
        prefix = trj.split("/")[-1].split(".")[0]
        #_path = "/".join(trj.split("/")[:-1])
        try:
            get_tpr = [tt for tt in tpr_list if tt.split("/")[-1].split(".")[0] == prefix]
        except Exception as e:
            logging.info(f"Missing {prefix}.tpr")
            logging.info("Terminated")
            return None
        
        if not get_tpr:
            logging.info(f"Missing {prefix}.tpr")
            logging.info("Terminated")
            return None
        
        paired.setdefault(prefix, [trj, get_tpr[0]])   
    
    return paired
        
def executor(
             xtc_list: list, 
             tpr_list: list,
             #paired: dict,
             focuse: str,
             polar_cutoff: float,
             hit_scope: float,
             ter_res_idx: int,
             nMC: int,
             margin: float,
             md_frame: int):
    
    work_dir = os.getcwd()
    
    ## collect trj and tpr name list
    #xtc_list = [xtc for xtc in os.listdir() if xtc.endswith(".xtc")]
    #tpr_list = [tpr for tpr in os.listdir() if tpr.endswith(".tpr")]

    focuse_dic = {
        "Antibody": "Ab",
        "Linker_Payload": "LP"
    }

    try:
        get_focuse = focuse_dic[focuse]
    except Exception as e:
        logging.info("No available region with defined focuse, switch to default as [Antibody]")
        get_focuse = "Ab"

    logging.info("----> STEP 0: Collecting trj")

    paired = checker(xtc_list, tpr_list)
    if not paired:
        logging.info("No input trj and/or topology")
        logging.info("Abort")
        return 
    
    logging.info("----> STEP 1: Analyze trj")

    col = {}
    for prefix, md_input in paired.items():
        #prefix = trj.split("/")[-1].split(".")[0]
        logging.info(f"-- Working with {prefix}")

        df = adc_agg(md_traj=md_input[0],
                     md_tpr=md_input[1],
                     focuse=get_focuse,
                     polar_cutoff=polar_cutoff,
                     hit_scope=hit_scope,
                     ter_res_idx=ter_res_idx,
                     nMC=nMC,
                     margin=margin,
                     md_frame=md_frame).run()
        
        if df.size:
            df.to_csv(os.path.join(work_dir, f"TEMP_{prefix}.csv"), index=None)
            col.setdefault(prefix, os.path.join(work_dir, f"TEMP_{prefix}.csv"))
            logging.info(f"-- Finish with {prefix}")

        else:
            logging.info(f"-- Failed with {prefix}")
        
    if not col:
        logging.info(f"-- Nothing to process")
        logging.info(f"-- Abort")
        return 
    
    logging.info("----> STEP 2: Processing result")

    dic = {}

    for kk, vv in col.items():
        df = pd.read_csv(vv)
        os.remove(vv)
        dic.setdefault(kk, df)
    
    logging.info("----> Done")
    
    return dic
    
    #output(data_dic=dic,
    #       save_dir=output_dir, 
    #       if_direct_save=True)
    
    #logging.info("----> Done")
    #return 




            

    
    
        



