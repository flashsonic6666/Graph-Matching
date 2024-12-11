from rdkit import Chem

def are_graphs_isomorphic(smiles1, smiles2):
    """
    Checks if the molecular graphs represented by two SMILES strings are isomorphic.
    
    Parameters:
        smiles1 (str): First SMILES string.
        smiles2 (str): Second SMILES string.

    Returns:
        bool: True if the graphs are isomorphic, False otherwise.
    """

    try:
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)

        if mol1 is None or mol2 is None:
            return False

        # Check isomorphism using RDKit's HasSubstructMatch
        return mol1.HasSubstructMatch(mol2) and mol2.HasSubstructMatch(mol1)

    except:
        return False

def preprocess_smiles(smiles, handle_r_groups=True):
    """
    Preprocesses SMILES strings to handle stereochemistry and R-groups.

    Parameters:
        smiles (str): The SMILES string to preprocess.
        handle_r_groups (bool): Whether to handle R-groups.

    Returns:
        str: The preprocessed canonical SMILES string.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            return None
            #raise ValueError(f"Invalid SMILES string: {smiles}")
        
        # Handle R-groups by replacing with wildcards or custom patterns
        if handle_r_groups:
            for atom in mol.GetAtoms():
                # Replace generic R-groups with "*"
                if atom.GetSymbol() == "R":
                    atom.SetAtomicNum(0)  # Wildcard (*)
                # Replace numbered R-groups (e.g., "R1") with "[1*]"
                if atom.GetIsotope() > 0:  # R-group with number
                    atom.SetIsotope(0)
                    atom.SetAtomicNum(0)  # Wildcard (*)

        # Convert to canonical SMILES with stereochemistry
        return Chem.MolToSmiles(mol, canonical=True)
    except:
        return smiles
    
def molfile_to_bonds_manual(molfile_string):
    """
    Extracts bonds manually from a MOL file string.

    Parameters:
        molfile_string (str): The MOL file content as a string.

    Returns:
        list of tuples: Each tuple contains:
            - Atom indices of the bond (int, int)
            - Bond type (int)
    """
    try:
        lines = molfile_string.splitlines()
        
        # The number of atoms and bonds are in the 4th line of the MOL file
        header_line = lines[3]
        num_atoms = int(header_line[:3].strip())
        num_bonds = int(header_line[3:6].strip())
        
        # Bonds start after the atom block
        bond_start_index = 4 + num_atoms  # 4 header lines + atom lines
        bonds = []

        # Parse the bond lines
        for i in range(bond_start_index, bond_start_index + num_bonds):
            bond_line = lines[i]
            
            # Extract atom indices and bond type from bond line
            atom1 = int(bond_line[:3].strip()) - 1  # Convert to 0-based index
            atom2 = int(bond_line[3:6].strip()) - 1  # Convert to 0-based index
            bond_type = int(bond_line[6:9].strip())
            
            bonds.append((atom1, atom2, bond_type))

        return bonds
    except:
        return None