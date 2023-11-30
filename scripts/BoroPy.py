# run with "streamlit run BoroPy.py"

from generatorfunctions import *
from fractions import Fraction
import streamlit as st
import numpy as np
import itertools
from ase import *
from ase.io import write
from ase.visualize import view
from ase.constraints import FixAtoms
import matplotlib.pyplot as plt
from scipy.spatial import distance
import os.path
import os
import pandas as pd
from io import StringIO 
import sys
from pathlib import Path, PurePath
import py3Dmol
from stmol import showmol

class Capturing(list):
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self
    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio    # free up some memory
        sys.stdout = self._stdout

def write_pdb(struct, name=""):
    a, b, c, alpha, beta, gamma = struct.cell.cellpar() 
    out  = "TITLE       " + name + "\n"
    out += "CRYST1{:9.3f}{:9.3f}{:9.3f}{:7.2f}{:7.2f}{:7.2f} P 1           1\n".format(a,b,c,alpha,beta,gamma)
    boron = struct[struct.get_atomic_numbers() == 5]
    metal = struct[struct.get_atomic_numbers() != 5]
    for i, (at, x, y, z) in enumerate(zip(
        boron.get_chemical_symbols(), boron.positions[:,0], boron.positions[:,1], boron.positions[:,2])):
        out += "ATOM  {:5d}  {:<4s}             {:8.3f}{:8.3f}{:8.3f}  1.00  1.00           {:s}\n".format(i+1,at,x,y,z,at)
    for i, (at, x, y, z) in enumerate(zip(
        metal.get_chemical_symbols(), metal.positions[:,0], metal.positions[:,1], metal.positions[:,2])):
        out += "ATOM  {:5d}  {:<4s}             {:8.3f}{:8.3f}{:8.3f}  1.00  1.00           {:s}\n".format(i+1+len(boron),at,x,y,z,at)
    pairs = np.array(list(itertools.combinations(range(len(boron)),2)))
    dr = distance.pdist(boron.positions)
    connected = pairs[dr<1.9]
    for i, j in connected:
        out +=  "CONECT{:5d}{:5d}\n".format(i+1,j+1)
        out += "ENDMDL\n"
    return out


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
css = '''
<style>
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
    font-size:2rem;
    }
</style>
'''

st.markdown(
    f"""
    <style>
    .stTabs [data-baseweb="tab-list"] {{
		gap: 10px;
    }}

	.stTabs [data-baseweb="tab"] {{
		height: 50px;
        white-space: pre-wrap;
		background-color: #F0F2F6;
		border-radius: 5px 5px 0px 0px;
		gap: 1px;
		padding: 10px;
    }}

	.stTabs [aria-selected="false"] {{
  		background-color: #FFFFFF;
	}}
 
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {{
        font-size:1.7rem;
        font-weight: bold;
    }}
    
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
c1, outcol = st.columns((4, 1))

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
metalchoice = left.selectbox("Metal", ('', 'Ag', 'Al', 'Au', 'Cu', 'Pt', 'Ni', 'Ir', 'Si'))
surfchoice = mid.selectbox("Surface", ('fcc111', 'fcc110', 'fcc100', 'fcc211', 'diamond100'))
NZ = right.number_input("Number of metal layers", min_value=0, max_value=None, value=3)
left, right = st.sidebar.columns((1, 1))
vac = left.number_input("Add vaccum layer on top [Å]", min_value=0., max_value=None, value=25.)
vdwdist = right.number_input("Van der Waals distance [Å]", min_value=0., max_value=None, value=2.5)
angle = left.selectbox( "Rotate borophene [˚]", (0, 90), index=0, key="angle")
random = right.number_input("Add random motion of max value [Å]", min_value=0., max_value=None, value=0.)
shiftX = left.number_input("Shift Borophene X [Å]", min_value=None, max_value=None, value=0., step=5.)
shiftY = right.number_input("Shift Borophene Y [Å]", min_value=None, max_value=None, value=0., step=5.)

# # # # # # # # # # # # # # # # # # 
st.sidebar.write("# Borophene island")
left, right = st.sidebar.columns((1, 1))
island_size = left.number_input("Borophene island size [Å]", min_value=0., max_value=None, value=0., step=5.)
island_shape = right.selectbox("Borophene island shape", ('Circle','Square','Triangle'))
island_angle = st.sidebar.number_input( "Rotate island [˚]", min_value=0., max_value=90., value=0., step=5., key="island_angle")

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
    island_size =island_size,
    island_shape=island_shape.lower(),
    island_angle=island_angle
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

tab1, tab2, tab3 = c1.tabs(["2D view", "3D view", "Output"])
collist = {'Ag':'silver', 'Al':'lightgrey', 'Au':'gold', 'Cu':'#fdb07d', 
           'Pt':'lightgrey', 'Ni':'#fed776', 'Ir':'lightgrey', 'Si':'lightgrey'}


with tab1:
    # # # # # # # # # # # # # # # # # # 
    # Plotting
    # # # # # # # # # # # # # # # # # # 
    left1, right1 = st.columns((3, 1))
    show_bonds = right1.checkbox("Show bonds", value=True)
    show_numbers = right1.checkbox("Show numbers")
    Nrepx = right1.number_input("Repeat x", min_value=1, max_value=None, value= 1)
    Nrepy = right1.number_input("Repeat y", min_value=1, max_value=None, value= 1)
    size = right1.slider("Point size", min_value=0, max_value=100, value=100, step=1, key="size")
    fsize = right1.slider("Font size", min_value=0, max_value=10, value=5, step=1, key="fsize")
    
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
    if metalchoice!='':
        ax.scatter(metal.positions[:, 0], metal.positions[:, 1], c=collist[metalchoice],
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
    left1.pyplot(fig)

with tab2:
    # # # # # # # # # # # # # # # # # # 
    # Plotting
    # # # # # # # # # # # # # # # # # # 
    t2l, t2r = st.columns((3, 1))
    width3d = t2r.slider("Width", min_value=10, max_value=1200, value=450, step=10, key="width3d")
    height3d = t2r.slider("Height", min_value=10, max_value=1200, value=400, step=10, key="height3d")
    
    system = write_pdb(struct.repeat((Nrepx,Nrepy,1)))
    atB = {'atom':'B'}
    xyzview = py3Dmol.view(width=width3d, height=height3d)
    xyzview.addModelsAsFrames(str(system))
    if metalchoice!='':
        xyzview.setStyle({'sphere':{'color':collist[metalchoice]}})
    xyzview.setStyle(atB,{'stick': {'color':'red', 'radius':0.2, 'opacity':0.9}, 
                          'sphere':{'color':'red', 'radius':1, 
                                    'opacity':0.9, 'scale':0.5}})
    xyzview.setBackgroundColor('white')
    xyzview.addUnitCell({'box':{'color':'purple'}})
    xyzview.replicateUnitCell(Nrepx, Nrepy, 1, addBonds=True)
    xyzview.spin(False)
    xyzview.zoomTo(atB)
    with t2l:
        showmol(xyzview, width=width3d, height=height3d)

# Show table with info
a = str(np.round(struct.cell[0,0],4))
b = str(np.round(struct.cell[1,1],4))
c = str(np.round(struct.cell[2,2],4))
mx = f"{(struct.get_cell()[0][0]-base.get_cell()[0][0])/base.get_cell()[0][0]*100:.2f} %"
my = f"{(struct.get_cell()[1][1]-base.get_cell()[1][1])/base.get_cell()[1][1]*100:.2f} %"
hd = Fraction(len(listholes), (2*nx*ny))
outcol.write("##### Cell Info")
if metalchoice!='':
    df = pd.DataFrame({'':['a [Å]', 'b [Å]', 'c [Å]', 'N<sub>total</sub>', 'N<sub>B</sub>', f'N<sub>{metalchoice}</sub>','Hole density'],
                    'b':[a,b,c, len(struct), len(boron), len(metal), f"{hd.numerator}/{hd.denominator}"]})
else:
    df = pd.DataFrame({'':['a [Å]', 'b [Å]', 'c [Å]', 'N<sub>total</sub>', 'N<sub>B</sub>', 'Hole density'],
                       'b':[a,b,c, len(struct), len(boron), f"{hd.numerator}/{hd.denominator}"]})
outcol.write(df.style.hide(axis=0).hide(axis=1).to_html(), unsafe_allow_html=True)
outcol.write("")
outcol.write("##### Borophene deformation")
df = pd.DataFrame({'':['x', 'y'],
                   'Borophene deformation': [mx, my]})
outcol.write(df.style.hide(axis=0).hide(axis=1).to_html(), unsafe_allow_html=True)


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
            sys.stdout.write(write_pdb(struct, name=name))
    else:
        with Capturing() as outfile:
            write(sys.stdout, struct, format=formlist[extension_out])
    return("\n".join(outfile),name)

with tab3:
    st.write("""
    ## Output options
    """)
    extension_out = st.selectbox("File format", ('VASP', 'xyz', 'LAMMPS', 'PDB'))

    if extension_out=='VASP':
        st.write("### For VASP output")
        fixed = st.number_input("Fixed number of layers", 
                                min_value=0, max_value=NZ+1, 
                                value=NZ-1, step=1)
        if fixed > 0:
            c = FixAtoms(mask=struct.positions[:,2] <= 2.35*(fixed-1)+1.2)
            struct.set_constraint(c)

    st.markdown("<br><br>", unsafe_allow_html=True)

    outfile,name = writeout(struct, extension_out)

    st.download_button(
        label     = "Download Output file",
        data      = outfile,
        file_name = name,
        mime      = 'text/csv',
    )
