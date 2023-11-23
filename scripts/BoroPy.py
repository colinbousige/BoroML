# run with "streamlit run BoroPy.py"

from generatorfunctions import *
from fractions import Fraction
import streamlit as st
import numpy as np
import itertools
from ase import *
from ase.io import write
from ase.visualize import view
import matplotlib.pyplot as plt
from scipy.spatial import distance
import os.path
import os
import pandas as pd
from io import StringIO 
import sys
from pathlib import Path, PurePath


class Capturing(list):
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self
    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio    # free up some memory
        sys.stdout = self._stdout


dico_predef = {'\u03b1'  :(3,3,[0,10]), #alpha
               '\u03b11' :(2,4,[0,10]), #alpha1
               '\u03b12' :(16,16,[0,18,37,55,74,92,111,7,25,44,62,65,83,102,120,139,157,
                                  113,132,150,169,187,206,160,178,197,215,234,252,208,227,
                                  245,264,282,301,319,273,292,310,329,347,366,322,368,340,
                                  359,377,396,414,387,405,424,442,461,479,417,435,482,500,
                                  454,472,491,509,271]), #alpha2
               '\u03b14' :(9,9,[0,11,23,34,37,48,60,71,97,85,74,99,111,136,122,148,134,159]), #alpha4
               '\u03b15' :(2,6,[0,15]), #alpha5
               '\u03b43' :(1,3,[0,4]),  #delta3
               '\u03b44' :(1,2,[0]),    #delta4
               '\u03b45' :(7,7,[0,16,11,32,52,73,93,36,57,77,48,27,68,89]),    #delta5
               '\u03b46' :(2, 2, []),   #delta6
               '\u03c72' :(3,6,[0,8,16,18,26,34]),  #chi2
               '\u03c73' :(1,5,[0,7]),  #chi3
               '\u03c74' :(6,6,[0,8,17,19,30,28,39,50,61,69,58,47]),  #chi4
               '\u03b24' :(5,3,[0,6,16]),    #beta4
               '\u03b25' :(5,3,[0,6,16,22]),    #beta5
               '\u03b28' :(1,9,[0,13]), #beta8
               '\u03b210':(1,4,[0]),    #beta10
               '\u03b211':(12,8,[0,4,19,28,43,53,68,77,92,102,117,126,141,151,166,175,190,
                                 24,23,47,49,64,73,88,98,113,122,137,147,162,171,186]),    #beta11
               '\u03b212':(1,3,[0]),    #beta12
               '\u03b213':(2,3,[0,4])   #beta13
               }

st.set_page_config(
    page_title="Borophene builder",
    page_icon=":hammer_and_pick:",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://lmi.cnrs.fr/author/colin-bousige/',
        'Report a bug': "https://lmi.cnrs.fr/author/colin-bousige/",
        'About': """
        ## Borophene builder
        Version date 2023-06-26.

        This app was made by [Colin Bousige](https://lmi.cnrs.fr/author/colin-bousige/). Contact me for support, requests, or to signal a bug.
        """
    }
)
padding_top = 0

st.markdown(
    f"""
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {{
        width: 500px;
    }}
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {{
        width: 500px;
        margin-left: -500px;
    }}
    .block-container{{
            padding-top: {padding_top}rem;
        }}
    thead tr th:first-child {{display:none}}
        tbody th {{display:none}}
    </style>
    """,
    unsafe_allow_html=True
)


st.write("""
# Borophene builder
""")


# # # # # # # # # # # # # # # # # # 
# Borophene Structure
# # # # # # # # # # # # # # # # # # 
c1, c2, c3 = st.columns((3, 1, 1))

st.sidebar.write("# Borophene Lattice")

col1,col2 = st.sidebar.columns((1, 1))

predef = col1.selectbox("Predefined structures", ['']+[k for k in dico_predef.keys()], key='predef')
Nboro = col2.number_input("Number of borophene layers", min_value=1, value=1, key="Nboro")

if predef != '':
    st.session_state.Nx = dico_predef[predef][0]
    st.session_state.Ny = dico_predef[predef][1]
    st.session_state.holes = dico_predef[predef][2]
nx = col1.number_input("Nx", min_value=1, max_value=None, key="Nx")
ny = col2.number_input("Ny", min_value=1, max_value=None, key="Ny")
listholes = st.sidebar.multiselect("Holes", range(2*nx*ny), key="holes")

col1,col2,col3 = st.sidebar.columns((2, 2, 3))
repeatx = col1.number_input("Repeat x", min_value=1, max_value=None, value=1, key="repeatx")
repeaty = col2.number_input("Repeat y", min_value=1, max_value=None, value=1, key="repeaty")
size_min = col3.number_input("Min size of the box [Å]", min_value=0., max_value=None, value=0.)

# # # # # # # # # # # # # # # # # # 
# Add metal slab
# # # # # # # # # # # # # # # # # # 

st.sidebar.write("# Metal Slab")
left, mid, right = st.sidebar.columns((1, 1, 1))
metalchoice = left.selectbox("Metal", ('', 'Ag', 'Au', 'Cu', 'Pt', 'Ni', 'Ir', 'Si'))
surfchoice = mid.selectbox("Surface", ('fcc111', 'fcc110', 'fcc100', 'fcc211', 'diamond100'))
NZ = right.number_input("Number of metal layers", min_value=0, max_value=None, value=3)
left, right = st.sidebar.columns((1, 1))
vac = left.number_input("Add vaccum layer on top [Å]", min_value=0., max_value=None, value=25.)
vdwdist = right.number_input("Van der Waals distance [Å]", min_value=0., max_value=None, value=2.5)
angle = left.selectbox( "Rotate borophene [˚]", (0, 90), index=0, key="angle")
random = right.number_input("Add random motion of max value [Å]", min_value=0., max_value=None, value=0.)
shiftX = left.number_input("Shift Borophene X [Å]", min_value=None, max_value=None, value=0., step=.1)
shiftY = right.number_input("Shift Borophene Y [Å]", min_value=None, max_value=None, value=0., step=.1)

# # # # # # # # # # # # # # # # # # 
st.sidebar.write("# Borophene island")
island = st.sidebar.number_input("Borophene island diameter [Å]", min_value=0., max_value=None, value=0., step=5.)



st.sidebar.markdown("<br><br>", unsafe_allow_html=True)

# # # # # # # # # # # # # # # # # # 
# Plotting
# # # # # # # # # # # # # # # # # # 

c2.write("## Plotting options")
show_bonds = c2.checkbox("Show bonds", value=True)
show_numbers = c2.checkbox("Show numbers")
Nrepx = c2.number_input("Repeat x", min_value=1, max_value=None, value= 1)
Nrepy = c2.number_input("Repeat y", min_value=1, max_value=None, value= 1)
size = c2.slider("Point size", min_value=0, max_value=100, value=100, step=1, key="size")
fsize = c2.slider("Font size", min_value=0, max_value=10, value=5, step=1, key="fsize")
view3d = c2.button("3D View")

if ny==1:
    metalchoice=''

struct = create_structure(
    nx=nx,
    ny=ny,
    listholes=listholes,
    repeatx=repeatx,
    repeaty=repeaty,
    metalchoice=metalchoice,
    surfchoice=surfchoice,
    vdwdist=vdwdist,
    vac=vac,
    angle=angle,
    NZ=NZ,
    random=random,
    size_min=size_min,
    glimpse=False,
    dmin=1,
    dmax=0,
    shiftX=shiftX,
    shiftY=shiftY,
    Nboro=Nboro,
    island=island
)

base = create_structure(
    nx=nx,
    ny=ny,
    listholes=listholes,
    repeatx=repeatx,
    repeaty=repeaty,
    metalchoice='None',
    surfchoice=surfchoice,
    vdwdist=vdwdist,
    vac=vac,
    angle=angle,
    NZ=NZ,
    random=random,
    size_min=size_min,
    glimpse=False,
    dmin=1,
    dmax=0,
    shiftX=shiftX,
    shiftY=shiftY,
    Nboro=Nboro
)

structfull = create_structure(
    nx=nx,
    ny=ny,
    listholes=[],
    repeatx=repeatx,
    repeaty=repeaty,
    metalchoice=metalchoice,
    surfchoice=surfchoice,
    vdwdist=vdwdist,
    vac=vac,
    angle=angle,
    NZ=NZ,
    random=random,
    size_min=size_min,
    glimpse=False,
    dmin=1,
    dmax=0,
    shiftX=shiftX,
    shiftY=shiftY,
    Nboro=Nboro
)

structfull = structfull[np.array(structfull.get_chemical_symbols())=="B"]
sortedpos = np.lexsort((structfull.positions[:, 1], structfull.positions[:, 0]))
structfull.positions = structfull.positions[sortedpos]

if view3d:
    view(struct)

fig = plt.figure()
# define plot organization
ax = fig.add_subplot(1,1,1)
# retrieve lattice parameters and define new repeated structure with the original unit cell
a = struct.cell[0,0]
b = struct.cell[1,1]
structrep = struct.repeat((Nrepx,Nrepy,1))
structrep.set_cell(struct.cell)
boron = structrep[structrep.get_atomic_numbers()==5] #get boron atoms
metal = structrep[structrep.get_atomic_numbers()!=5] #get metal atoms
ax.scatter(metal.positions[:, 0], metal.positions[:, 1], c='darkgray',
           zorder=1, s=200*size/100, edgecolor='black', linewidth=0.5)
if show_bonds:
    # compute all possible pairs of boron
    pairs = np.array(list(itertools.combinations(
        range(boron.get_global_number_of_atoms()), 2)))
    # compute all distances between boron atoms
    dr = distance.pdist(boron.positions)
    for i, j in pairs[dr<2.1]:
        # for each bond, define the segment of the bond to plot
        x = [boron.positions[i,0], boron.positions[j,0]]
        y = [boron.positions[i,1], boron.positions[j,1]]
        # plot the segment (bond)
        ax.plot(x, y, c='pink',linewidth=1.5, zorder=1)
# plot the atoms
ax.scatter(boron.positions[:,0],boron.positions[:,1], c='pink', zorder=2, s=120*size/100, edgecolor='black', linewidth=0.5)
# plot the unit cell
ax.plot([0,a,a,0,0], [0,0,b,b,0], c='darkgray',linewidth=1, zorder=3, linestyle='--')
if show_numbers:
    for i, pos in enumerate(structfull.positions[structfull.get_atomic_numbers()==5]):
        ax.text(pos[0],pos[1],str(i),horizontalalignment='center',
        verticalalignment='center',fontsize=fsize, color="black")
# remove black border
ax.axis('off')
ax.set_aspect('equal', 'datalim')
# define plot range
c1.pyplot(fig)

# Show table with info
a = np.round(struct.cell[0,0],4)
b = np.round(struct.cell[1,1],4)
c = np.round(struct.cell[2,2],4)
mx = f"{(struct.get_cell()[0][0]-base.get_cell()[0][0])/base.get_cell()[0][0]*100:.2f} %"
my = f"{(struct.get_cell()[1][1]-base.get_cell()[1][1])/base.get_cell()[1][1]*100:.2f} %"
hd = Fraction(len(listholes), (2*nx*ny))
df = pd.DataFrame(
    [[a,b,c, len(struct), len(boron), len(metal), f"{hd.numerator}/{hd.denominator}"]],
    columns=('a [Å]', 'b [Å]', 'c [Å]', 'Ntotal', 'Nboro', 'Nmetal','Hole density'))
c1.table(df)
df = pd.DataFrame(
    [[mx, my]],
    columns=('Borophene deformation along x', 'Borophene deformation along y'))
c1.table(df)

# # # # # # # # # # # # # # # # # # 
# OUTPUT
# # # # # # # # # # # # # # # # # # 

def writeout(struct, extension_out):
    """
    Write output file in the selected format in the current directory
    """
    # Define name with the wanted parameters
    name = 'Boro' + str(nx) + ',' + str(ny)
    if ','.join(map(str,listholes))!='':
        name += '_' + ','.join(map(str,listholes))
    if metalchoice!='':
        name += f"-{NZ}{metalchoice}_{surfchoice}"
    # define file extension...
    extlist = {'VASP':'.POSCAR', 'xyz':'.xyz', 'LAMMPS':'.data', 'PDB':'.pdb'}
    # and the corresponding ASE writing function
    formlist = {'VASP':'vasp', 'xyz':'xyz', 'LAMMPS':'lammps-data', 'PDB':'proteindatabank'}
    name += extlist[extension_out]
    if extension_out=='VASP':
        with Capturing() as outfile:
            write(sys.stdout, struct, format=formlist[extension_out], vasp5=True)
    elif extension_out=='LAMMPS':
        with Capturing() as outfile:
            write_lammps(None, struct, atom_style="charge", units="real")
    elif extension_out=='PDB': 
        with Capturing() as outfile:
            a, b, c, alpha, beta, gamma = struct.get_cell_lengths_and_angles()
            sys.stdout.write("TITLE       " + name + "\n")
            sys.stdout.write("CRYST1{:9.3f}{:9.3f}{:9.3f}{:7.2f}{:7.2f}{:7.2f} P 1           1\n".format(a,b,c,alpha,beta,gamma));
            boron = struct[struct.get_atomic_numbers() == 5]
            metal = struct[struct.get_atomic_numbers() != 5]
            for i, (at, x, y, z) in enumerate(zip(
                boron.get_chemical_symbols(), boron.positions[:,0], boron.positions[:,1], boron.positions[:,2])):
                sys.stdout.write("ATOM  {:5d}  {:<4s}             {:8.3f}{:8.3f}{:8.3f}  1.00  1.00           {:s}\n".format(i+1,at,x,y,z,at));
            for i, (at, x, y, z) in enumerate(zip(
                metal.get_chemical_symbols(), metal.positions[:,0], metal.positions[:,1], metal.positions[:,2])):
                sys.stdout.write("ATOM  {:5d}  {:<4s}             {:8.3f}{:8.3f}{:8.3f}  1.00  1.00           {:s}\n".format(i+1+len(boron),at,x,y,z,at));
            pairs = np.array(list(itertools.combinations(range(len(boron)),2)))
            dr = distance.pdist(boron.positions)
            connected = pairs[dr<1.9]
            for i, j in connected:
                sys.stdout.write( "CONECT{:5d}{:5d}\n".format(i+1,j+1) )
                sys.stdout.write("ENDMDL\n")
    else:
        with Capturing() as outfile:
            write(sys.stdout, struct, format=formlist[extension_out])
    return("\n".join(outfile),name)

c3.write("""
## Output options
""")
extension_out = c3.selectbox("File format", ('VASP', 'xyz', 'LAMMPS', 'PDB'))

outfile,name = writeout(struct, extension_out)

c3.download_button(
    label     = "Download Output file",
    data      = outfile,
    file_name = name,
    mime      = 'text/csv',
)
