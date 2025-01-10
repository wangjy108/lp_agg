import MDAnalysis as mda
import numpy as np
import pandas as pd
from rdkit import Chem
from multiprocessing import cpu_count
from joblib import Parallel, delayed

import logging
logging.basicConfig(format='[%(levelname)s] %(message)s', \
                    level=logging.INFO)



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

        try:
            self.polar_cutoff = float(args["polar_cutoff"])
        except Exception as e:
            self.polar_cutoff = 0.3

        try:
            self.hit_scope = float(args["hit_scope"])
        except Exception as e:
            self.hit_scope = 3.5
            
        
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
    

    
    def region_split(self, frame):
        # single frame
        LP = frame.select_atoms("resname L31 or resname L32 or resname L33 or resname L34 or resname L35 or resname L36 or resname L37 or resname L38")

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

        pre_Ab = frame.select_atoms(f"resid 1-{self.terminal_residx} and around {self.hit_scope} group LP", LP=LP)

        assembled_res = []

        for aa in pre_Ab:
            assembled_res.append(aa.resid)
        
        xyz_ab = []
        vdw_ab = []
        #atomicidx = []
        charge_ab = []
        
        for atom in frame.atoms:
            if atom.resid in assembled_res:
                xyz_ab.append(list(atom.position))
                vdw_ab.append(Chem.GetPeriodicTable().GetRvdw(Chem.GetPeriodicTable().GetAtomicNumber(atom.name[0])))
                #atomicidx.append(atom.index+1)
                charge_ab.append(atom.charge)

        get_ab_xyz = np.array(xyz_ab)
        get_ab_vdw = np.array(vdw_ab).reshape(-1,1)
        get_ab_charge = np.array(charge_ab).reshape(-1,1)

        return {"LP": {"xyz": get_LP_xyz,
                        "vdw": get_LP_vdw,
                        "charge": get_LP_charge},
                "Ab": {"xyz": get_ab_xyz,
                       "vdw": get_ab_vdw,
                       "charge": get_ab_charge}}
    
    def region_shade(self, frame, frame_idx):
        get_splited = self.region_split(frame)
        try:
            focuse = get_splited[self.focuse]
        except Exception as e:
            return None
        
        surrounding_key = [kk for kk in get_splited.keys() if kk != self.focuse][0]
        surrounding = get_splited[surrounding_key]

        PolarIdx = np.where((focuse["charge"]>=self.polar_cutoff) | (focuse["charge"] <= self.polar_cutoff * (-0.1)))[0]
        NonPolarIdx = np.where((focuse["charge"] < self.polar_cutoff) & (focuse["charge"] > self.polar_cutoff * (-0.1)))[0]

        sys_xyz = np.vstack((focuse["xyz"], surrounding["xyz"]))
        sys_vdw = np.vstack((focuse["vdw"], surrounding["vdw"]))

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
        
        polar_region_in_solo = grid(xyz=focuse["xyz"], 
                                    vdw=focuse["vdw"], 
                                    margin=self.margin,
                                    nMC=self.nMC,
                                    xyz_idx_list=list(PolarIdx)).run()

        nonpolar_region_in_solo = grid(xyz=focuse["xyz"], 
                                    vdw=focuse["vdw"], 
                                    margin=self.margin,
                                    nMC=self.nMC,
                                    xyz_idx_list=list(NonPolarIdx)).run()
        
        df = pd.DataFrame({"idx": [frame_idx],
                           "polar_mol_in_sys": [len(polar_region_in_sys)],
                           "nonpolar_mol_in_sys": [len(nonpolar_region_in_sys)],
                           "polar_mol": [len(polar_region_in_solo)],
                           "polar_mol": [len(nonpolar_region_in_solo)],
                           "delta_polar_sas": [len(polar_region_in_solo) - len(polar_region_in_sys)],
                           "delta_nonpolar_sas": [len(nonpolar_region_in_solo) - len(nonpolar_region_in_sys)]})
        return df
    
    def sequential(self, frame_list):
        _list = []
        for frame in frame_list:
            df = self.region_shade(frame, frame.frame)
            _list.append(df)
        
        return _list


    def run(self):
        u = mda.Universe(self.md_tpr, self.md_traj)

        # frame list -> u.trajectory

        sheet = []

        n_thread = cpu_count()

        if n_thread > 1:
            while True:
                n_in_thread = math.ceil(len(u.trajectory) / n_thread)

                if math.ceil(len(u.trajectory) / n_in_thread) == n_thread:
                    break
                
                n_thread -= 1
            
            #logging.info(f"Use {self.n_thread} thread(s) for simulation")
        
        else:
            n_in_thread = math.ceil(len(u.trajectory) / n_thread)

        _collect = Parallel(n_jobs=n_thread)(\
                           delayed(sequential)(u.trajectory[i*n_in_thread:(i+1)*n_in_thread]) \
                           for i in range(n_thread))


        for each in _collect:
            sheet += each
        
        df_all = np.concat(sheet)

        return df_all

