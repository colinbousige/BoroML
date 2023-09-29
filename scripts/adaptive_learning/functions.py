import numpy as np
from ase import Atoms
import datetime

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

def flatten(l):
    """
    Flatten nested list
    """
    return [item for sublist in l for item in sublist]

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

def read_inputdata(filename: str):
    """
    Read input.data file and return a list of Atoms objects and the comments
    """
    def read_data_block(lines, begin: int, end: int):
        """
        Read block of structure in input.data that have been read in to `lines`
        """
        lattice = np.array([[i for i in lines[j].split(' ') if i][1:4] for j in range(begin+2, begin+5)], dtype=float)
        pos = np.array([[i for i in lines[j].split(' ') if i][1:4] for j in range(begin+5, end-2)], dtype=float)
        symb = np.array([[i for i in lines[j].split(' ') if i][4] for j in range(begin+5, end-2)])
        return(Atoms(symbols   = symb,
                     positions = pos*0.529177, #convert to Å
                     cell      = lattice*0.529177, pbc=True))
    # # # # # # # # # # # # 
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    begin = np.array([i for i, x in enumerate(lines) if x == 'begin'])
    end = np.array([i for i, x in enumerate(lines) if x == 'end'])
    comments = np.array([x for x in lines if 'comment' in x])
    atoms = [read_data_block(lines, b, e) for b,e in zip(begin, end)]
    return(atoms, comments)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

def write_inputdata(filename: str, atoms, comments):
    """
    From a list of Atoms object and comments, write an input.data file to `filename`.
    """
    with open(filename, "w") as f:
        for atom,comment in zip(atoms, comments):
            a,b,c = atom.get_cell()/0.529177 #convert to Bohr
            f.write("begin\n")
            f.write(f"comment {comment}\n")
            f.write(f"lattice {a[0]:22.14e} {a[1]:22.14e} {a[2]:22.14e}\n")
            f.write(f"lattice {b[0]:22.14e} {b[1]:22.14e} {b[2]:22.14e}\n")
            f.write(f"lattice {c[0]:22.14e} {c[1]:22.14e} {c[2]:22.14e}\n")
            for at,pos in zip(atom.get_chemical_symbols(), atom.positions):
                x,y,z = pos/0.529177 #convert to Bohr
                f.write(f"atom {x:22.14e} {y:22.14e} {z:22.14e} {at:s} {'0.0':s} {'0.0':s} {0:22.14e} {0:22.14e} {0:22.14e}\n")
            f.write(f"energy {0:22.14e}\n")
            f.write(f"charge 0\n")
            f.write("end\n")

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

def print_success_message():
    """
    Prints a super cool success message. Absolutely *not geeky*.
    """
    now = datetime.datetime.now()
    print(f"""
    
****************************************************************
* OMFG!!! The job finished without any problem !!!             *
* This is a godlike achievement, you get a beer on my account! *
* Time of ending: {now.strftime('%Y-%m-%d %H:%M:%S')}                          *
****************************************************************

                       ¶¶¶¶¶¶¶¶¶
                    ¶¶           ¶¶
         ¶¶¶¶     ¶¶                ¶¶
        ¶     ¶  ¶¶      ¶¶     ¶¶     ¶¶
        ¶     ¶ ¶¶      ¶¶¶¶   ¶¶¶¶      ¶¶
        ¶    ¶  ¶¶       ¶¶     ¶¶        ¶¶
        ¶   ¶   ¶                          ¶¶
     ¶¶¶¶¶¶¶¶¶¶              ¶              ¶¶
    ¶            ¶           ¶¶             ¶¶
    ¶¶            ¶    ¶            ¶       ¶¶
    ¶¶   ¶¶¶¶¶¶¶¶¶¶¶    ¶¶        ¶¶       ¶¶
    ¶               ¶     ¶¶¶¶¶¶¶¶        ¶¶
    ¶¶              ¶                    ¶¶
    ¶   ¶¶¶¶¶¶¶¶¶¶¶¶                   ¶¶
    ¶¶           ¶  ¶¶                ¶¶
     ¶¶¶¶¶¶¶¶¶¶      ¶¶            ¶¶
                        ¶¶¶¶¶¶¶¶¶¶¶
""", flush=True)


