from rdkit import Chem
from rdkit.Chem import Draw, AllChem, rdMolAlign, rdFMCS, rdchem,rdMolAlign, rdShapeHelpers, rdMolDescriptors
import os
from matplotlib import pyplot as plt
import copy

from util.ConfRelaxbySQM import System as optimizer
from util.Cluster import cluster
from util.OptbySQM import System as sysopt

from util.Align import Align as align
import scipy.spatial
import subprocess
from multiprocessing import cpu_count

import pandas as pd
import numpy as np
import logging
logging.basicConfig(format='[%(levelname)s] %(message)s', level=logging.INFO)


#from grid_sample import sasa as grid
## unipf grid sample

##

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

def matched_region(input_rdmol_obj: object, apo_rdmol_obj:object) -> dict:

    hit_region = list(input_rdmol_obj.GetSubstructMatch(apo_rdmol_obj))

    HA_idx = [aa.GetIdx() for aa in input_rdmol_obj.GetAtoms() if aa.GetSymbol() != 'H']    

    _dic = {}

    for each in hit_region:
        try:
            HH = [(bb.GetEndAtomIdx(),bb.GetBeginAtomIdx()) for bb in input_rdmol_obj.GetBonds() if bb.GetBeginAtomIdx() == each or bb.GetEndAtomIdx() == each]
        except Exception as e:
            HH = None
        
        if HH:
            get_HH = []
            for every in HH:
                reduced = [hh for hh in list(every) if hh not in HA_idx]
                get_HH += reduced

            if set(get_HH):
                _dic.setdefault(each, list(set(get_HH)))
    
    return _dic

def get_region_distribution(mol_xyz: object, mol_vdw: object, grid_space: object) -> dict:
    _dot = {}

    for i, coor in enumerate(mol_xyz):

        #anti_coor = np.delete(mol_xyz, i, axis=0)
        #anti_vdw = np.delete(mol_vdw, i, axis=0)
        
        #sample_size = int(mol_vdw[i][0] * dimension)
        
        #raw_dots = dotsphere(sample_size) * mol_vdw[i][0] + coor
        
        dis = scipy.spatial.distance.cdist(grid_space, coor.reshape(1, -1), metric='euclidean')
        save_dots = grid_space[np.min(scipy.spatial.distance.cdist(grid_space,coor.reshape(1, -1), metric='euclidean'), axis=1) <= mol_vdw[i]]

        _size = save_dots.shape[0]

        _dot.setdefault(i, _size)
                
    return _dot

def calc_np_p(dots, _charge, cutoff):

    df_mol = pd.DataFrame({"idx": [kk for kk in dots.keys()],
                            "dot_shape": [vv for vv in dots.values()]}).sort_values(by="idx", ascending=True)

    input_charge = pd.DataFrame({"idx":[i for i in range(len(_charge))], "charge": _charge})

    df = pd.merge(df_mol, input_charge, on="idx")

    sas_p = df[abs(df["charge"]) >= cutoff]["dot_shape"].sum() 
    sas_np = df[abs(df["charge"]) < cutoff]["dot_shape"].sum() 

    return {"sas_p": sas_p, "sas_np": sas_np}


def get_np_set(mol: object,
               mol_reduce: object,
               charge_cutoff: float,
               define_margin: float,
               define_nMC: int) -> object:

    #apo_rdmol_obj = Chem.MolFromSmiles(apo_smi)

    #if not os.path.exists(_path):
    #    logging.info("Wrong calculation path, abort")
    #    return None

    #os.chdir(_path)

    #if not (os.path.isfile("mol.sdf") and os.path.isfile("reduce.sdf")):
    #    logging.info("Neccessary sdf file(s) missing, nothing to do, abort")
    #    return None

    
    #try:
    #    mol = Chem.SDMolSupplier("./mol.sdf", removeHs=False)[0]
    #except Exception as e:
    #    logging.info("Neccessary mol file(s) missing, nothing to do, abort")
    #    return None
    
    #try:
    #    mol_reduce = Chem.SDMolSupplier("./reduce.sdf", removeHs=False)[0]
    #except Exception as e:
    #    logging.info("Neccessary reduce file(s) missing, nothing to do, abort")
    #    return None
    
    if not (mol and mol_reduce):
        logging.info("No mol to process, abort")
        return None
    
    apo_rdmol_obj = copy.deepcopy(mol_reduce)
    apo_rdmol_obj = Chem.RemoveHs(apo_rdmol_obj)

    _ = AllChem.Compute2DCoords(apo_rdmol_obj)
    _dic = matched_region(mol, apo_rdmol_obj)

    if not _dic:
        logging.info("Reduced form matching error, abort")
        return None
        
    reduce_in_mol_idx = [kk for kk in _dic.keys()]

    for vv in _dic.values():
        reduce_in_mol_idx += vv
        

    get_xyz = mol.GetConformer().GetPositions()
    get_vdw = np.array([Chem.GetPeriodicTable().GetRvdw(a.GetAtomicNum()) \
                                for a in mol.GetAtoms()]).reshape(-1,1)

    get_xyz_reduce = mol_reduce.GetConformer().GetPositions()
    get_vdw_reduce = np.array([Chem.GetPeriodicTable().GetRvdw(a.GetAtomicNum()) \
                                for a in mol_reduce.GetAtoms()]).reshape(-1,1)
    

    ## get reduce in mol
    sample_mol = grid(xyz=get_xyz,
                  vdw=get_vdw,
                  margin=define_margin,
                  nMC=define_nMC).run()
    sample_reduce = grid(xyz=get_xyz_reduce,
                    vdw=get_vdw_reduce,
                    margin=define_margin,
                    nMC=define_nMC).run()
    sample_reduce_in_mol =grid(xyz=get_xyz,
                               vdw=get_vdw,
                               margin=define_margin,
                               nMC=define_nMC,
                               xyz_idx_list=reduce_in_mol_idx).run()
    
    #idx_array = np.where(np.min(scipy.spatial.distance.cdist(sample_reduce, sample_mol, metric='euclidean'), axis=1) ==0)
    #print(np.argmin(scipy.spatial.distance.cdist(sample_reduce, sample_mol, metric='euclidean'), axis=1))
    #idx_array = np.argmin(scipy.spatial.distance.cdist(sample_reduce, sample_mol, metric='euclidean'), axis=1)

    #sample_reduce_in_mol = sample_mol[list(idx_array)]

    #matched_xyz = get_xyz[list(np.where(np.min(scipy.spatial.distance.cdist(get_xyz, get_xyz_reduce, metric='euclidean'), axis=1) ==0)[0])]
    #matched_vdw = get_vdw[list(np.where(np.min(scipy.spatial.distance.cdist(get_xyz, get_xyz_reduce, metric='euclidean'), axis=1) ==0)[0])]

    AllChem.ComputeGasteigerCharges(mol)
    get_charge = [float(atom.GetProp("_GasteigerCharge")) for atom in mol.GetAtoms()]

    AllChem.ComputeGasteigerCharges(mol_reduce)
    get_charge_reduce = [float(atom.GetProp("_GasteigerCharge")) for atom in mol_reduce.GetAtoms()]

    matched_xyz = get_xyz[reduce_in_mol_idx]
    matched_vdw = get_vdw[reduce_in_mol_idx]

    matched_charge = np.array(get_charge)[reduce_in_mol_idx]

    #input_charge = pd.DataFrame({"idx":[i for i in range(len(get_charge))], "charge": get_charge})

    ##cal dots
    mol_dots = get_region_distribution(get_xyz,  get_vdw, sample_mol)
    reduce_dots = get_region_distribution(get_xyz_reduce, get_vdw_reduce, sample_reduce)
    reduce_dots_in_mol = get_region_distribution(matched_xyz, matched_vdw, sample_reduce_in_mol)

    dic_mol = calc_np_p(mol_dots, get_charge, charge_cutoff)
    dic_reduce = calc_np_p(reduce_dots, get_charge_reduce, charge_cutoff)
    dic_reduce_in_mol = calc_np_p(reduce_dots_in_mol, matched_charge, charge_cutoff)

    ## shading effects
    shades = (sample_reduce.shape[0] - sample_reduce_in_mol.shape[0]) / sample_reduce.shape[0]

    df = pd.DataFrame({"shade:sas": [shades],
                       "mol:sas_p": [dic_mol["sas_p"]],
                       "mol:sas_np": [dic_mol["sas_np"]],
                       "reduce:sas_p": [dic_reduce["sas_p"]],
                       "reduce:sas_np": [dic_reduce["sas_np"]],
                       "reduce_in_mol:sas_p": [dic_reduce_in_mol["sas_p"]],
                       "reduce_in_mol:sas_np": [dic_reduce_in_mol["sas_np"]]})
    #df.rename(index={0: int(_path)})
    
    return df

def get_fragment_each(BeginAtomIdx, BondPair, HeavyAtomIdx, RingAtomSet, **args):
    col = []
    try:
        EndAtomIdx = args["EndAtomIdx"]
    except Exception as e:
        EndAtomIdx = False
    else:
        EndAtomIdx = str(EndAtomIdx)

    for tracker in BeginAtomIdx:
        hit_pair = [pp for pp in BondPair if tracker in pp]
        #print(hit_pair)
        for each in hit_pair:
            if EndAtomIdx:
                _col = [aa for aa in each if aa in HeavyAtomIdx and aa != int(EndAtomIdx) and aa not in RingAtomSet]
            else:
                _col = [aa for aa in each if aa in HeavyAtomIdx and aa not in RingAtomSet]
            if _col:
                col += _col
    
    if set(col) == set(BeginAtomIdx):
        return BeginAtomIdx
    else:
        BeginAtomIdx = list(set(col))
        return get_fragment_each(BeginAtomIdx, BondPair, HeavyAtomIdx, RingAtomSet, EndAtomIdx=EndAtomIdx)

def get_fragment(rdmol_obj_ref):
    HA_atomIdx = [aa.GetIdx() for aa in rdmol_obj_ref.GetAtoms() if aa.GetSymbol() != 'H']
    

    single_bond_pair = [(bb.GetBeginAtomIdx(), bb.GetEndAtomIdx()) for bb in rdmol_obj_ref.GetBonds() \
                                if (bb.GetBondTypeAsDouble() == 1.0) and (not bb.IsInRing()) \
                                and max(bb.GetBeginAtomIdx(), bb.GetEndAtomIdx()) in HA_atomIdx]
    bond_pair = [(bb.GetBeginAtomIdx(), bb.GetEndAtomIdx()) for bb in rdmol_obj_ref.GetBonds()]
    non_single_bond_pair = [bp for bp in bond_pair if bp not in single_bond_pair]
    aliphatic_non_single_bond_pair = [[bb.GetBeginAtomIdx(), bb.GetEndAtomIdx()] for bb in rdmol_obj_ref.GetBonds() \
                            if (bb.GetBondTypeAsDouble() != 1.0) and (not bb.IsInRing()) \
                            and max(bb.GetBeginAtomIdx(), bb.GetEndAtomIdx()) in HA_atomIdx]

    ring_atom_set = []
    ring_size = []
    _dic = {}
    _dic.setdefault("ring",[])
    _dic.setdefault("decoration_on_ring", {})
    _dic.setdefault("aliphatic_NonSingle_group", aliphatic_non_single_bond_pair)
    _dic.setdefault("aliphatic_Single_atom", [])

    for idx, ring_set in enumerate(list(rdmol_obj_ref.GetRingInfo().AtomRings())):
        _dic["ring"].append(list(ring_set))
        ring_atom_set += list(ring_set)
        ring_size.append(len(ring_set))

    ring_atom_set = set(ring_atom_set)

    for pair in single_bond_pair:
        flag = [aa for aa in pair if aa in ring_atom_set]
        if not flag:
            forward = get_fragment_each([pair[0]], non_single_bond_pair, HA_atomIdx, ring_atom_set, EndAtomIdx=pair[-1])
            if len(forward) > 1 and forward not in aliphatic_non_single_bond_pair:
                _dic["aliphatic_NonSingle_group"].append(forward)
            #backward = get_fragment_each([pair[-1]], non_single_bond_pair, HA_atomIdx, ring_atom_set, EndAtomIdx=pair[0])
            #if backward:
            #    _dic.setdefault((pair[-1], pair[0]), backward)
        #   if len(forward) > 1 and len(forward) <= max(ring_size):
        #    if len(backward) > 1 and len(backward) <= max(ring_size):
        elif len(flag) == 1:
            BeginAtom = [aa for aa in pair if aa != flag[0]][0]
            EndAtom = [aa for aa in pair if aa != BeginAtom][0]
            get_frag = get_fragment_each([BeginAtom], bond_pair, HA_atomIdx, ring_atom_set, EndAtomIdx=EndAtom)
            if len(get_frag) >= 1 and len(get_frag) <= max(ring_size):
                get_frag = list(set(get_frag + [BeginAtom]))
                if get_frag not in _dic["decoration_on_ring"].values():
                    try:
                        _dic["decoration_on_ring"][str(EndAtom)]
                    except Exception as e:
                        _dic["decoration_on_ring"].setdefault(str(EndAtom), get_frag)
                    else:
                        idx = len([cc for cc in _dic["decoration_on_ring"].keys() if str(EndAtom) == cc])
                        #idx = len([cc for cc in _dic["decoration_on_ring"][str(EndAtom)]])
                        _dic["decoration_on_ring"].setdefault(f"{EndAtom}_{idx}", get_frag)
        else:
            pass
        
    for atom in HA_atomIdx:
        flag = [aa for aa in _dic["ring"] if atom in aa] \
        + [aa for aa in _dic["decoration_on_ring"].values() if atom in aa] \
        + [aa for aa in _dic["aliphatic_NonSingle_group"] if atom in aa]
        if not flag:
            _dic["aliphatic_Single_atom"].append([atom])
    
    if _dic["aliphatic_NonSingle_group"]:
        get_aliphatic_NonSingle_group_atom = []
    for each in _dic["aliphatic_NonSingle_group"]:
        get_aliphatic_NonSingle_group_atom += each

    for root_atom, decoration_atom in _dic["decoration_on_ring"].items():
        if get_aliphatic_NonSingle_group_atom:
            update_decoration_atom = [cc for cc in decoration_atom if cc not in get_aliphatic_NonSingle_group_atom]
            _dic["decoration_on_ring"][root_atom] = update_decoration_atom

    return _dic

def readin(input_sdf: str, input_apo: str):
    try:
        mol = [cc for cc in Chem.SDMolSupplier(input_sdf, removeHs=False) if cc]
    except Exception as e:
        mol = None
    
    try:
        apo = Chem.MolFromSmiles(input_apo)
    except Exception as e:
        apo = None
    
    if not (mol and apo):
        return 

    return mol, apo

def pairHeavyHydrogen(input_rdmol_obj: object, **args) -> dict:

    HA_idx = [aa.GetIdx() for aa in input_rdmol_obj.GetAtoms() if aa.GetSymbol() != 'H']

    try:
        hit_region = args["hit_region"]
    except Exception as e:
        hit_region = HA_idx
    
    else:
        if not hit_region:
            hit_region = HA_idx

    ## get Heavy Atom list
    

    _dic = {}

    for each in hit_region:
        try:
            HH = [(bb.GetEndAtomIdx(),bb.GetBeginAtomIdx()) for bb in input_rdmol_obj.GetBonds() if bb.GetBeginAtomIdx() == each or bb.GetEndAtomIdx() == each]
        except Exception as e:
            HH = None
        
        if HH:
            get_HH = []
            for every in HH:
                reduced = [hh for hh in list(every) if hh not in HA_idx]
                get_HH += reduced

            if set(get_HH):
                _dic.setdefault(each, list(set(get_HH)))
    
    return _dic

def getsdf(atom: list, xyz: object, prefix: str) -> object:
    df = pd.DataFrame({"atom": atom, \
                        "x": xyz[:, 0], \
                        "y": xyz[:, 1], \
                        "z": xyz[:, 2]})
    
    with open(f"_TEMP_{prefix}.xyz", "w+") as ff:
        ff.write(f"{xyz.shape[0]}\n")
        ff.write(f"{prefix}\n")
        for idx, row in df.iterrows():
            ff.write(f"{row['atom']:<3}{row['x']:>15.3f}{row['y']:>15.3f}{row['z']:>15.3f}\n")
    
    cmd = f"obabel -ixyz _TEMP_{prefix}.xyz -O _TEMP_{prefix}.sdf"

    try:
        p = subprocess.run(cmd.split(), timeout=20, check=True, stdout=subprocess.PIPE)
    except subprocess.TimeoutExpired:
        logging.info("Timeout with sdf gen")
    
    if os.path.isfile(f"_TEMP_{prefix}.sdf") and os.path.getsize(f"_TEMP_{prefix}.sdf"):
        try:
            mol = [cc for cc in Chem.SDMolSupplier(f"_TEMP_{prefix}.sdf", removeHs=False) if cc]
        except Exception as e:
            return None
        if mol:
            return mol[0]
        else:
            return None
        
    else:
        return None

def get_segmantation_group(input_rdmol_obj: object, segmantation: int):
    fragment_group = get_fragment(input_rdmol_obj)

    flag = None

    types = ["ring", "aliphatic_NonSingle_group", "aliphatic_Single_atom"]

    i = 0

    while i < len(types):
        if flag:
            break
        for groups in fragment_group[types[i]]:
            flag = [cc for cc in groups if cc == segmantation]
            if flag:
                flag = groups
                break
        
        i += 1
    

    return  flag


def get_segmantation(each_rdmol_obj: object, apo_rdmol_obj: object):
    assign_idx = list(each_rdmol_obj.GetSubstructMatch(apo_rdmol_obj))

    ha_idx = [aa.GetIdx() for aa in each_rdmol_obj.GetAtoms() if aa.GetSymbol() != 'H']

    segmantation = {}

    ## define the cutting point and it's resemblance group
    for idx, ha in enumerate(assign_idx):
        try:
            connects = [(bb.GetEndAtomIdx(),bb.GetBeginAtomIdx()) for bb in each_rdmol_obj.GetBonds() if bb.GetBeginAtomIdx() == ha or bb.GetEndAtomIdx() == ha]
        except Exception as e:
            connects = None
        
        if connects:
            filter_ = []
            for every in connects:
                reduced = [hh for hh in list(every) if (hh in ha_idx and hh != ha)]
                filter_ += reduced

            if set(filter_):
                flag = [aa for aa in list(set(filter_)) if aa not in assign_idx]

                if flag:
                    segmantation.setdefault("idx_apo", idx)
                    segmantation.setdefault("idx_mol", ha)
    
    return segmantation

def assemble_saturated_fragment(rdmol_obj: object, apo_rdmol_obj: object):
    assign_idx = list(rdmol_obj.GetSubstructMatch(apo_rdmol_obj))
    segmentation_idx = get_segmantation(rdmol_obj, apo_rdmol_obj)

    ## apo match
    fragment_apo = get_fragment(apo_rdmol_obj)
    hit_group_apo = get_segmantation_group(apo_rdmol_obj, segmentation_idx["idx_apo"])

    expand_hit_group = [(bb.GetBeginAtomIdx(), bb.GetEndAtomIdx()) for bb in apo_rdmol_obj.GetBonds() \
        if bb.GetEndAtomIdx() in hit_group_apo or bb.GetBeginAtomIdx() in hit_group_apo]

    _list = []

    for each in expand_hit_group:
        _list += list(set(each))

    apo_frag_smi = None

    for aa in list(set(_list)):
        if aa not in hit_group_apo:
            end_connects = [(bb.GetBeginAtomIdx(), bb.GetEndAtomIdx()) for bb in apo_rdmol_obj.GetBonds() \
                if (bb.GetEndAtomIdx()==aa and bb.GetBeginAtomIdx() in hit_group_apo) \
                        or (bb.GetBeginAtomIdx()==aa and bb.GetEndAtomIdx() in hit_group_apo)][0]
            
            get_bond_idx = apo_rdmol_obj.GetBondBetweenAtoms(end_connects[0], end_connects[1]).GetIdx()
            get_split = Chem.MolToSmiles(Chem.FragmentOnBonds(apo_rdmol_obj, [get_bond_idx], addDummies=False)).split(".")

            if len(get_split[0]) < len(get_split[1]):
                apo_frag_smi = get_split[0]
            else:
                apo_frag_smi = get_split[1]
            
    if apo_frag_smi:
        appending = apo_frag_smi
        m2d = Chem.AddHs(Chem.MolFromSmiles(appending))
        nGenConfs = AllChem.EmbedMultipleConfs(m2d,numConfs=1, numThreads=0)
        res = AllChem.MMFFOptimizeMoleculeConfs(m2d, numThreads=0)

        cc = Chem.SDWriter("_TEMP_apo_frag.sdf")
        cc.write(m2d, confId=0)
        cc.close()

        try:
            apo_frag_3D = [cc for cc in Chem.SDMolSupplier("_TEMP_apo_frag.sdf", removeHs=False) if cc]
        except Exception as e:
            apo_frag_3D = None
        
        if apo_frag_3D:
            apo_frag_3D = apo_frag_3D[0]
    else:
        apo_frag_3D = None


    frgament_mol = get_fragment(rdmol_obj)
    hit_group_mol = get_segmantation_group(rdmol_obj, segmentation_idx["idx_mol"])

    atom = [atom.GetSymbol() for atom in rdmol_obj.GetAtoms()]
    xyz = rdmol_obj.GetConformer().GetPositions()

    hh_pair = pairHeavyHydrogen(rdmol_obj, hit_region=hit_group_mol)

    HH_idx = []

    for vv in hh_pair.values():
        HH_idx += vv

    hit_atom = []

    for each in hit_group_mol+HH_idx:
        hit_atom.append(atom[each])

    hit_xyz = xyz[hit_group_mol+HH_idx]

    hit_frag_mol = getsdf(hit_atom, hit_xyz, "hit_group")

    connects_in_mol = xyz[[segmentation_idx["idx_mol"]]]

    # do align 
    if not (apo_frag_3D and hit_frag_mol):
        return None
    

    aligned = align(SearchMolObj=apo_frag_3D, RefMolObj=hit_frag_mol, method="crippen3D").run()

    aligned_xyz = aligned.GetConformer().GetPositions()
    dis_aligned = scipy.spatial.distance.cdist(aligned_xyz, connects_in_mol, metric="euclidean")

    get_aligned_connect_id = np.argmin(dis_aligned)

    col_h = pairHeavyHydrogen(aligned, hit_region=[get_aligned_connect_id])

    get_append_H_idx = []

    for rr in col_h.values():
        get_append_H_idx += rr

    mol_pair = pairHeavyHydrogen(rdmol_obj)

    HH_append = []

    for mha in assign_idx:
        try:
            get_matched_hh = mol_pair[mha]
        except Exception as e:
            continue

        if get_matched_hh:
            HH_append += get_matched_hh

    reduced_xyz = xyz[assign_idx + HH_append]

    reduced_atom = []

    for each in assign_idx + HH_append:
        reduced_atom.append(atom[each])

    complete_xyz = np.vstack((reduced_xyz, aligned_xyz[get_append_H_idx]))
    complete_atom = reduced_atom + [atom.GetSymbol() for atom in aligned.GetAtoms() if atom.GetIdx() in get_append_H_idx]

    complete_hit_mol = getsdf(complete_atom, complete_xyz, "Mol:reduced")

    if complete_hit_mol:
        os.system("rm -f _TEMP*")

    return complete_hit_mol


class adc_lp_agg():
    def __init__(self, **args):
        #self.Apo = args["apo_smi"]
        #self.mol = args["mol_initial_conformer"]
        
        try:
            self.nMC = int(args["set_nMC"])
        except Exception as e:
            self.nMC = 1
        
        try:
            self.margin = float(args["set_margin"])
        except Exception as e:
            self.margin = 0
        
        try:
            self.partial_charge_cutoff = float(args["polarity_cutoff"])
        except Exception as e:
            self.partial_charge_cutoff = 0.3
        
        

        try:
            self.mol = [cc for cc in Chem.SDMolSupplier(args["mol_initial_conformer"], removeHs=False) if cc]
        except Exception as e:
            self.mol = None

        
        try:
            self.apo = Chem.MolFromSmiles(args["apo_smi"])
        except Exception as e:
            self.apo = None
        
        try:
            self.sample_frame = int(args["sample_n_frame"])
        except Exception as e:
            self.sample_frame = None
        
        try:
            self.energy_window = float(args["energy_window"])
        except Exception as e:
            self.energy_window = 20.0
        
        try:
            self.duplicate_rmsd_cutoff = float(args["duplicate_rmsd_cutoff"])
        except Exception as e:
            self.duplicate_rmsd_cutoff = None
        

        try:
            self.charge = args["define_charge"]
        except Exception as e:
            self.charge = None
        
    
    def sample_from_initial_conformer(self, each_mol):
        _dic_rotableBond = {0: 120, 1: 300} 
        _dic_duplicate = {0: 1.0, 1: 2.0}

        if not self.sample_frame:
            rotable_bond = rdMolDescriptors.CalcNumRotatableBonds(each_mol)
            _def_func = lambda x: 0 if max(8, x) == 8 else 1
            rotable_bond_index = _def_func(rotable_bond)
            n_sample_frame = _dic_rotableBond[rotable_bond_index]
        else:
            n_sample_frame = self.sample_frame
        
        if not self.duplicate_rmsd_cutoff:
            rotable_bond = rdMolDescriptors.CalcNumRotatableBonds(each_mol)
            _def_func = lambda x: 0 if max(8, x) == 8 else 1
            rotable_bond_index = _def_func(rotable_bond)
            duplicate_rmsd_cutoff = _dic_duplicate[rotable_bond_index]
        else:
            duplicate_rmsd_cutoff = self.duplicate_rmsd_cutoff

        
        ## perform initial_ha_opt

        _initial_opt = sysopt(input_rdmol_obj=[each_mol],
                                HA_constrain=True).run()
        #self.charge = _initial_opt[0].GetProp("charge")
        
        #if not opt:
        if not _initial_opt:
            logging.info("Failed at initial opt")
            return None
        
        cc = Chem.SDWriter("initial_opt.sdf")
        cc.write(_initial_opt[0])
        cc.close()

        ## perform sampling

        optimizer(input_sdf="initial_opt.sdf",
                  save_frame=n_sample_frame).run()
        
        if not (os.path.isfile("SAVE.sdf") and os.path.getsize("SAVE.sdf")):
            logging.info("Failed at sampling")
            return 
            #logging.info("Sampled Conformation(s) were saved in [SAVE.sdf]")
            
        ## reduce duplicate
        MOL = cluster(input_sdf="SAVE.sdf", 
                      rmsd_cutoff_cluster=duplicate_rmsd_cutoff,
                      only_reduce_duplicate=True).run()
        
        if not MOL:
            logging.info("Failed at decrease redundancy")
            return None

        optimized = sysopt(input_rdmol_obj=MOL,
                            cpu_core=cpu_count() ,
                            HA_constrain=True).run() 

        if not optimized:
            logging.info("Failed at post sampling optimization")
            return None
        

        ## reduce outside window conformer

        min_energy = min([float(x.GetProp("Energy_xtb")) for x in optimized])

        saved_optimized = [cc for cc in optimized if (float(cc.GetProp("Energy_xtb")) - min_energy)*627.51 < self.energy_window]

        return saved_optimized
    
    def get_reduce(self, each_mol: object) -> object:
        
        try:
            _reduce = assemble_saturated_fragment(each_mol, self.apo)
        except Exception as e:
            logging.info("Failed at get matched apo form from specific conformer, abort")
            return None
        
        if not _reduce:
            logging.info("No matched apo form from specific conformer, abort")
            return None
        
        opt_reduced = sysopt(input_rdmol_obj=[_reduce],
                        HA_constrain=True,
                        if_write_sdf=False).run()
        
        if not opt_reduced:
            logging.info("Failed at optimize matched apo form from specific conformer, abort")
            return None
        

        return opt_reduced[0]
    
    def calc_surface_property(self, mol: object, _reduce: object) -> object:

        try:
            df = get_np_set(mol, _reduce, self.partial_charge_cutoff, self.margin, self.nMC)
        except Exception as e:
            df = None
        
        return df
    
    def run(self):
        logging.info("START linker-payload aggregation tendency evluation ---> ")

        if not self.mol:
            logging.info("Failed at picking up input conformers, abort")
            return 
        
        logging.info("----> Processing")

        _dic = {}
            
        for idx, each_mol in enumerate(self.mol):
            mol_assemble = []
            get_prefix = each_mol.GetProp("_Name")
            logging.info(f"-> Working with {get_prefix}")
            sampled_conformer = self.sample_from_initial_conformer(each_mol)
            if not sampled_conformer:
                logging.info(f"Failed with {get_prefix}, continue")
                continue

            for ii, opt_mol in enumerate(sampled_conformer):
                get_reduced = self.get_reduce(opt_mol)
                if not get_reduced:
                    logging.info(f"Failed when get reduced form of {ii}th conformer for {get_prefix}, continue")
                    continue

                df = self.calc_surface_property(opt_mol, get_reduced)
                if (not df) or df.shape[0]:
                    mol_assemble.append(df)
            
            if mol_assemble:
                df_all = pd.concat(mol_assemble)
                df_all["delta_np"] = df_all.apply(lambda x: x["reduce_in_mol:sas_np"] - x["reduce:sas_np"], axis=1)
                df_all["delta_p"] = df_all.apply(lambda x: x["reduce_in_mol:sas_p"] - x["reduce:sas_p"], axis=1)
                #df["delta_np:delta_p"] = df.apply(lambda x: abs(x["delta_np"] / x["delta_p"]) if x['delta_p'] else (abs(x["delta_np"])), axis=1)
                #df["np:p"] = df.apply(lambda x: x["mol:sas_np"] / x["mol:sas_p"], axis=1)
                #df["reduce:np:p"] = df.apply(lambda x: x["reduce:sas_np"] / x["reduce:sas_p"], axis=1)
                df_all["ratio:delta_np"] = df_all.apply(lambda x: abs(x["delta_np"]) / (abs(x["delta_p"]) + abs(x["delta_np"])), axis=1)
                df_all["ratio:delta_p"] = df_all.apply(lambda x: abs(x["delta_p"]) / (abs(x["delta_p"]) + abs(x["delta_np"])), axis=1)
                df_all["ratio-delta_p:ratio-delta_np"] = df_all.apply(lambda x: x["ratio:delta_p"] / x["ratio:delta_np"], axis=1)
                _dic.setdefault(get_prefix, df_all)
                logging.info("-> Success")
        
        logging.info("----> Finish")

        return _dic
    
    def output(self, data_dic: dict) -> object:

        df_saver = {}

        show_types = ["shade:sas", "ratio-delta_p:ratio-delta_np", "delta_np", "delta_p", "ratio:delta_np", "ratio:delta_p"]

        cc_map = ["orange", "blue", "green", "green", "green", "green"]

        ss_idx = [321, 322, 323, 324, 325, 326]

        titles = ["Ratio of shield surface in total",
                "Ratio of $\Delta_{Polar Surface}$ vs $\Delta_{Non Polar Surface}$",
                "$\Delta_{Non Polar Surface}$",
                "$\Delta_{Polar Surface}$",
                "Ratio of $\Delta_{Non Polar Surface}$",
                "Ratio of $\Delta_{Polar Surface}$ "]
        ylabels = ["Sheilded surface / total surface",
                "$\Delta_{Polar Surface}$ / $\Delta_{Non Polar Surface}$",
                "$\Delta_{Non Polar Surface}$",
                "$\Delta_{Polar Surface}$",
                "$\Delta_{Non Polar Surface}$ / ($\Delta_{Non Polar Surface}$ +  $\Delta_{Polar Surface}$)",
                    "$\Delta_{Polar Surface}$ / ($\Delta_{Non Polar Surface}$ +  $\Delta_{Polar Surface}$)"]

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
                        ylabel="ratio $\Delta_{Polar Surface}$",
                        ax=ax)
            
            ax.title.set_text(f"{titles[idx]}")

            for p in ax.patches:
                ax.annotate(f"{p.get_height():.3f}",
                                    (p.get_x() * 1.005, p.get_height() * 1.005))
            
        return f, df_saver

