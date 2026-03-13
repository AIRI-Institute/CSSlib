"""Module with functions for parsing VASP output xml files."""

__all__ = []

import os
import pickle
import re
import tqdm
from pymatgen.io.vasp import Vasprun


def getStructuresAsList(workdir):
    '''
        Input:
            - workdir - directory, where vasp xml files are located
        Description:
            Parse xml files in work directory, reads .pkl file if it exists or reads all xml files and writes data to .pkl: 
            extractes structure formula, natoms, atomic_symbols, atomic_numbers, nelements, initial_cell (optional), 
            initial_structure, final_cell (optional), final_structure, last_step_pressure, last_step_forces and energy.
        ```
        Return - list of data with information about structures
        ```
    '''
    xml_list = [xml for xml in os.listdir(workdir) if xml.endswith('xml')]
    pkl_file_name = re.sub('3res_structures/', '3res_list.pkl', workdir) # replaces 3res_structures/ in workdir path into 3res_list.pkl
    print('\ngetStructuresAsList out:', pkl_file_name)
    if os.path.exists(pkl_file_name) :
        with open(pkl_file_name, 'rb') as f:
            result = pickle.load(f)
        if len(xml_list) == len(result) :
            print(f'result file for {workdir} has been read from pkl')
            return result
    print(f'result file for {workdir} is being updated')
    result =[]
    for xml_file in tqdm(xml_list):
        tag = re.sub('vasprun_', '', re.sub('.xml', '', xml_file))
        #print(tag)
        try:
            run = Vasprun(f'{workdir}/{xml_file}', parse_potcar_file=False)
        except:
            print(f'vasprun.xml is not OK for {tag}')

        if not run.converged:
            print(f'vasprun.xml IS NOT CONVERGED for {tag}')

        initial_structure = run.initial_structure
        final_structure = run.final_structure
        #initial_cell = initial_structure.lattice.matrix
        #final_cell = final_structure.lattice.matrix
        #initial_structure = np.array([x.coords for x in initial_structure])
        #final_structure = np.array([x.coords for x in final_structure])
        last_step_forces = np.array(run.ionic_steps[-1]['forces'])
        last_step_pressure = np.trace(run.ionic_steps[-1]['stress']) / 3
        energy = run.final_energy
        atomic_symbols = run.atomic_symbols
        atomic_numbers = np.array(list(map(lambda x: Element(x).number, atomic_symbols)))
        formula = OrderedDict(sorted(Counter(atomic_symbols).items()))
        natoms = len(atomic_symbols)
        nelements = len(formula)
        result += [[tag, formula, natoms, atomic_symbols, atomic_numbers, nelements, 
                    initial_structure, #initial_cell, initial_structure, 
                    final_structure, #final_cell, final_structure, 
                    last_step_pressure, last_step_forces, energy]]
    with open(pkl_file_name, 'wb') as f:
        pickle.dump(result, f)
    return result