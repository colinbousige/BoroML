#!/usr/local/bin/python3

"""
Collection of functions to read and write files in different formats.

Copyright (c) 2026 Colin Bousige
Licensed under the MIT License
"""

import sys
import os
from os.path import isfile
import numpy as np
import pandas as pd
from ase import Atoms
from ase.io.lammpsdata import read_lammps_data
from ase.io.lammpsrun import read_lammps_dump
from ase.io.vasp import read_vasp_out, read_vasp
from ase.io import read as ase_read
from ase.cell import Cell
from ase.calculators.singlepoint import SinglePointCalculator
import re
from tqdm import tqdm
from io import StringIO
from ase.io.lammpsdata import Prism, convert

numeric_const_pattern = r"[-+]?(?:(?: \d*\.\d+)|(?: \d+\.?))(?:[Ee][+-]?\d+)?"
rx = re.compile(numeric_const_pattern, re.VERBOSE)

orange = "\033[93m"
bold = "\033[1m"
normal = "\033[0m"

cdist = 1.0 / 0.52917721
cener = 1.0 / 27.21138469
cforce = cener / cdist

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

PT = pd.read_csv(
    StringIO("""1,Hydrogen,H,1.007,0,1,1,1,1,gas,,yes,,yes,,Nonmetal,0.79,2.2,13.5984,8.99E-05,14.175,20.28,3,Cavendish,1766,14.304,1,1
2,Helium,He,4.002,2,2,2,1,18,gas,,yes,,yes,,Noble Gas,0.49,,24.5874,1.79E-04,,4.22,5,Janssen,1868,5.193,1,
3,Lithium,Li,6.941,4,3,3,2,1,solid,,yes,yes,,,Alkali Metal,2.1,0.98,5.3917,5.34E-01,453.85,1615,5,Arfvedson,1817,3.582,2,1
4,Beryllium,Be,9.012,5,4,4,2,2,solid,,yes,yes,,,Alkaline Earth Metal,1.4,1.57,9.3227,1.85E+00,1560.15,2742,6,Vaulquelin,1798,1.825,2,2
5,Boron,B,10.811,6,5,5,2,13,solid,,yes,,,yes,Metalloid,1.2,2.04,8.298,2.34E+00,2573.15,4200,6,Gay-Lussac,1808,1.026,2,3
6,Carbon,C,12.011,6,6,6,2,14,solid,,yes,,yes,,Nonmetal,0.91,2.55,11.2603,2.27E+00,3948.15,4300,7,Prehistoric,,0.709,2,4
7,Nitrogen,N,14.007,7,7,7,2,15,gas,,yes,,yes,,Nonmetal,0.75,3.04,14.5341,1.25E-03,63.29,77.36,8,Rutherford,1772,1.04,2,5
8,Oxygen,O,15.999,8,8,8,2,16,gas,,yes,,yes,,Nonmetal,0.65,3.44,13.6181,1.43E-03,50.5,90.2,8,Priestley/Scheele,1774,0.918,2,6
9,Fluorine,F,18.998,10,9,9,2,17,gas,,yes,,yes,,Halogen,0.57,3.98,17.4228,1.70E-03,53.63,85.03,6,Moissan,1886,0.824,2,7
10,Neon,Ne,20.18,10,10,10,2,18,gas,,yes,,yes,,Noble Gas,0.51,,21.5645,9.00E-04,24.703,27.07,8,Ramsay and Travers,1898,1.03,2,8
11,Sodium,Na,22.99,12,11,11,3,1,solid,,yes,yes,,,Alkali Metal,2.2,0.93,5.1391,9.71E-01,371.15,1156,7,Davy,1807,1.228,3,1
12,Magnesium,Mg,24.305,12,12,12,3,2,solid,,yes,yes,,,Alkaline Earth Metal,1.7,1.31,7.6462,1.74E+00,923.15,1363,8,Black,1755,1.023,3,2
13,Aluminum,Al,26.982,14,13,13,3,13,solid,,yes,yes,,,Metal,1.8,1.61,5.9858,2.70E+00,933.4,2792,8,Wshler,1827,0.897,3,3
14,Silicon,Si,28.086,14,14,14,3,14,solid,,yes,,,yes,Metalloid,1.5,1.9,8.1517,2.33E+00,1683.15,3538,8,Berzelius,1824,0.705,3,4
15,Phosphorus,P,30.974,16,15,15,3,15,solid,,yes,,yes,,Nonmetal,1.2,2.19,10.4867,1.82E+00,317.25,553,7,BranBrand,1669,0.769,3,5
16,Sulfur,S,32.065,16,16,16,3,16,solid,,yes,,yes,,Nonmetal,1.1,2.58,10.36,2.07E+00,388.51,717.8,10,Prehistoric,,0.71,3,6
17,Chlorine,Cl,35.453,18,17,17,3,17,gas,,yes,,yes,,Halogen,0.97,3.16,12.9676,3.21E-03,172.31,239.11,11,Scheele,1774,0.479,3,7
18,Argon,Ar,39.948,22,18,18,3,18,gas,,yes,,yes,,Noble Gas,0.88,,15.7596,1.78E-03,83.96,87.3,8,Rayleigh and Ramsay,1894,0.52,3,8
19,Potassium,K,39.098,20,19,19,4,1,solid,,yes,yes,,,Alkali Metal,2.8,0.82,4.3407,8.62E-01,336.5,1032,10,Davy,1807,0.757,4,1
20,Calcium,Ca,40.078,20,20,20,4,2,solid,,yes,yes,,,Alkaline Earth Metal,2.2,1,6.1132,1.54E+00,1112.15,1757,14,Davy,1808,0.647,4,2
21,Scandium,Sc,44.956,24,21,21,4,3,solid,,yes,yes,,,Transition Metal,2.1,1.36,6.5615,2.99E+00,1812.15,3109,15,Nilson,1878,0.568,4,
22,Titanium,Ti,47.867,26,22,22,4,4,solid,,yes,yes,,,Transition Metal,2,1.54,6.8281,4.54E+00,1933.15,3560,9,Gregor,1791,0.523,4,
23,Vanadium,V,50.942,28,23,23,4,5,solid,,yes,yes,,,Transition Metal,1.9,1.63,6.7462,6.11E+00,2175.15,3680,9,   del Rio,1801,0.489,4,
24,Chromium,Cr,51.996,28,24,24,4,6,solid,,yes,yes,,,Transition Metal,1.9,1.66,6.7665,7.15E+00,2130.15,2944,9,Vauquelin,1797,0.449,4,
25,Manganese,Mn,54.938,30,25,25,4,7,solid,,yes,yes,,,Transition Metal,1.8,1.55,7.434,7.44E+00,1519.15,2334,11,"Gahn, Scheele",1774,0.479,4,
26,Iron,Fe,55.845,30,26,26,4,8,solid,,yes,yes,,,Transition Metal,1.7,1.83,7.9024,7.87E+00,1808.15,3134,10,Prehistoric,,0.449,4,
27,Cobalt,Co,58.933,32,27,27,4,9,solid,,yes,yes,,,Transition Metal,1.7,1.88,7.881,8.86E+00,1768.15,3200,14,Brandt,1735,0.421,4,
28,Nickel,Ni,58.693,31,28,28,4,10,solid,,yes,yes,,,Transition Metal,1.6,1.91,7.6398,8.91E+00,1726.15,3186,11,Cronstedt,1751,0.444,4,
29,Copper,Cu,63.546,35,29,29,4,11,solid,,yes,yes,,,Transition Metal,1.6,1.9,7.7264,8.96E+00,1357.75,2835,11,Prehistoric,,0.385,4,
30,Zinc,Zn,65.38,35,30,30,4,12,solid,,yes,yes,,,Transition Metal,1.5,1.65,9.3942,7.13E+00,692.88,1180,15,Prehistoric,,0.388,4,
31,Gallium,Ga,69.723,39,31,31,4,13,solid,,yes,yes,,,Metal,1.8,1.81,5.9993,5.91E+00,302.91,2477,14,de Boisbaudran,1875,0.371,4,3
32,Germanium,Ge,72.64,41,32,32,4,14,solid,,yes,,,yes,Metalloid,1.5,2.01,7.8994,5.32E+00,1211.45,3106,17,Winkler,1886,0.32,4,4
33,Arsenic,As,74.922,42,33,33,4,15,solid,,yes,,,yes,Metalloid,1.3,2.18,9.7886,5.78E+00,1090.15,887,14,Albertus Magnus,1250,0.329,4,5
34,Selenium,Se,78.96,45,34,34,4,16,solid,,yes,,yes,,Nonmetal,1.2,2.55,9.7524,4.81E+00,494.15,958,20,Berzelius,1817,0.321,4,6
35,Bromine,Br,79.904,45,35,35,4,17,liq,,yes,,yes,,Halogen,1.1,2.96,11.8138,3.12E+00,266.05,332,19,Balard,1826,0.474,4,7
36,Krypton,Kr,83.798,48,36,36,4,18,gas,,yes,,yes,,Noble Gas,1,,13.9996,3.73E-03,115.93,119.93,23,Ramsay and Travers,1898,0.248,4,8
37,Rubidium,Rb,85.468,48,37,37,5,1,solid,,yes,yes,,,Alkali Metal,3,0.82,4.1771,1.53E+00,312.79,961,20,Bunsen and Kirchoff,1861,0.363,5,1
38,Strontium,Sr,87.62,50,38,38,5,2,solid,,yes,yes,,,Alkaline Earth Metal,2.5,0.95,5.6949,2.64E+00,1042.15,1655,18,Davy,1808,0.301,5,2
39,Yttrium,Y,88.906,50,39,39,5,3,solid,,yes,yes,,,Transition Metal,2.3,1.22,6.2173,4.47E+00,1799.15,3609,21,Gadolin,1794,0.298,5,
40,Zirconium,Zr,91.224,51,40,40,5,4,solid,,yes,yes,,,Transition Metal,2.2,1.33,6.6339,6.51E+00,2125.15,4682,20,Klaproth,1789,0.278,5,
41,Niobium,Nb,92.906,52,41,41,5,5,solid,,yes,yes,,,Transition Metal,2.1,1.6,6.7589,8.57E+00,2741.15,5017,24,Hatchett,1801,0.265,5,
42,Molybdenum,Mo,95.96,54,42,42,5,6,solid,,yes,yes,,,Transition Metal,2,2.16,7.0924,1.02E+01,2890.15,4912,20,Scheele,1778,0.251,5,
43,Technetium,Tc,98,55,43,43,5,7,artificial,yes,,yes,,,Transition Metal,2,1.9,7.28,1.15E+01,2473.15,5150,23,Perrier and Segr�,1937,,5,
44,Ruthenium,Ru,101.07,57,44,44,5,8,solid,,yes,yes,,,Transition Metal,1.9,2.2,7.3605,1.24E+01,2523.15,4423,16,Klaus,1844,0.238,5,
45,Rhodium,Rh,102.906,58,45,45,5,9,solid,,yes,yes,,,Transition Metal,1.8,2.28,7.4589,1.24E+01,2239.15,3968,20,Wollaston,1803,0.243,5,
46,Palladium,Pd,106.42,60,46,46,5,10,solid,,yes,yes,,,Transition Metal,1.8,2.2,8.3369,1.20E+01,1825.15,3236,21,Wollaston,1803,0.244,5,
47,Silver,Ag,107.868,61,47,47,5,11,solid,,yes,yes,,,Transition Metal,1.8,1.93,7.5762,1.05E+01,1234.15,2435,27,Prehistoric,,0.235,5,
48,Cadmium,Cd,112.411,64,48,48,5,12,solid,,yes,yes,,,Transition Metal,1.7,1.69,8.9938,8.69E+00,594.33,1040,22,Stromeyer,1817,0.232,5,
49,Indium,In,114.818,66,49,49,5,13,solid,,yes,yes,,,Metal,2,1.78,5.7864,7.31E+00,429.91,2345,34,Reich and Richter,1863,0.233,5,3
50,Tin,Sn,118.71,69,50,50,5,14,solid,,yes,yes,,,Metal,1.7,1.96,7.3439,7.29E+00,505.21,2875,28,Prehistoric,,0.228,5,4
51,Antimony,Sb,121.76,71,51,51,5,15,solid,,yes,,,yes,Metalloid,1.5,2.05,8.6084,6.69E+00,904.05,1860,29,Early historic times,,0.207,5,5
52,Tellurium,Te,127.6,76,52,52,5,16,solid,,yes,,,yes,Metalloid,1.4,2.1,9.0096,6.23E+00,722.8,1261,29,von Reichenstein,1782,0.202,5,6
53,Iodine,I,126.904,74,53,53,5,17,solid,,yes,,yes,,Halogen,1.3,2.66,10.4513,4.93E+00,386.65,457.4,24,Courtois,1811,0.214,5,7
54,Xenon,Xe,131.293,77,54,54,5,18,gas,,yes,,yes,,Noble Gas,1.2,,12.1298,5.89E-03,161.45,165.03,31,Ramsay and Travers,1898,0.158,5,8
55,Cesium,Cs,132.905,78,55,55,6,1,solid,,yes,yes,,,Alkali Metal,3.3,0.79,3.8939,1.87E+00,301.7,944,22,Bunsen and Kirchoff,1860,0.242,6,1
56,Barium,Ba,137.327,81,56,56,6,2,solid,,yes,yes,,,Alkaline Earth Metal,2.8,0.89,5.2117,3.59E+00,1002.15,2170,25,Davy,1808,0.204,6,2
57,Lanthanum,La,138.905,82,57,57,6,3,solid,,yes,yes,,,Lanthanide,2.7,1.1,5.5769,6.15E+00,1193.15,3737,19,Mosander,1839,0.195,6,
58,Cerium,Ce,140.116,82,58,58,6,,solid,,yes,yes,,,Lanthanide,2.7,1.12,5.5387,6.77E+00,1071.15,3716,19,Berzelius,1803,0.192,6,
59,Praseodymium,Pr,140.908,82,59,59,6,,solid,,yes,yes,,,Lanthanide,2.7,1.13,5.473,6.77E+00,1204.15,3793,15,von Welsbach,1885,0.193,6,
60,Neodymium,Nd,144.242,84,60,60,6,,solid,,yes,yes,,,Lanthanide,2.6,1.14,5.525,7.01E+00,1289.15,3347,16,von Welsbach,1885,0.19,6,
61,Promethium,Pm,145,84,61,61,6,,artificial,yes,,yes,,,Lanthanide,2.6,1.13,5.582,7.26E+00,1204.15,3273,14,Marinsky et al.,1945,,6,
62,Samarium,Sm,150.36,88,62,62,6,,solid,,yes,yes,,,Lanthanide,2.6,1.17,5.6437,7.52E+00,1345.15,2067,17,Boisbaudran,1879,0.197,6,
63,Europium,Eu,151.964,89,63,63,6,,solid,,yes,yes,,,Lanthanide,2.6,1.2,5.6704,5.24E+00,1095.15,1802,21,Demarcay,1901,0.182,6,
64,Gadolinium,Gd,157.25,93,64,64,6,,solid,,yes,yes,,,Lanthanide,2.5,1.2,6.1501,7.90E+00,1585.15,3546,17,de Marignac,1880,0.236,6,
65,Terbium,Tb,158.925,94,65,65,6,,solid,,yes,yes,,,Lanthanide,2.5,1.2,5.8638,8.23E+00,1630.15,3503,24,Mosander,1843,0.182,6,
66,Dysprosium,Dy,162.5,97,66,66,6,,solid,,yes,yes,,,Lanthanide,2.5,1.22,5.9389,8.55E+00,1680.15,2840,21,de Boisbaudran,1886,0.17,6,
67,Holmium,Ho,164.93,98,67,67,6,,solid,,yes,yes,,,Lanthanide,2.5,1.23,6.0215,8.80E+00,1743.15,2993,29,Delafontaine and Soret,1878,0.165,6,
68,Erbium,Er,167.259,99,68,68,6,,solid,,yes,yes,,,Lanthanide,2.5,1.24,6.1077,9.07E+00,1795.15,3503,16,Mosander,1843,0.168,6,
69,Thulium,Tm,168.934,100,69,69,6,,solid,,yes,yes,,,Lanthanide,2.4,1.25,6.1843,9.32E+00,1818.15,2223,18,Cleve,1879,0.16,6,
70,Ytterbium,Yb,173.054,103,70,70,6,,solid,,yes,yes,,,Lanthanide,2.4,1.1,6.2542,6.97E+00,1097.15,1469,16,Marignac,1878,0.155,6,
71,Lutetium,Lu,174.967,104,71,71,6,,solid,,yes,yes,,,Lanthanide,2.3,1.27,5.4259,9.84E+00,1936.15,3675,22,Urbain/ von Welsbach,1907,0.154,6,
72,Hafnium,Hf,178.49,106,72,72,6,4,solid,,yes,yes,,,Transition Metal,2.2,1.3,6.8251,1.33E+01,2500.15,4876,17,Coster and von Hevesy,1923,0.144,6,
73,Tantalum,Ta,180.948,108,73,73,6,5,solid,,yes,yes,,,Transition Metal,2.1,1.5,7.5496,1.67E+01,3269.15,5731,19,Ekeberg,1801,0.14,6,
74,Wolfram,W,183.84,110,74,74,6,6,solid,,yes,yes,,,Transition Metal,2,2.36,7.864,1.93E+01,3680.15,5828,22,J. and F. d'Elhuyar,1783,0.132,6,
75,Rhenium,Re,186.207,111,75,75,6,7,solid,,yes,yes,,,Transition Metal,2,1.9,7.8335,2.10E+01,3453.15,5869,21,"Noddack, Berg, and Tacke",1925,0.137,6,
76,Osmium,Os,190.23,114,76,76,6,8,solid,,yes,yes,,,Transition Metal,1.9,2.2,8.4382,2.26E+01,3300.15,5285,19,Tennant,1803,0.13,6,
77,Iridium,Ir,192.217,115,77,77,6,9,solid,,yes,yes,,,Transition Metal,1.9,2.2,8.967,2.26E+01,2716.15,4701,25,Tennant,1804,0.131,6,
78,Platinum,Pt,195.084,117,78,78,6,10,solid,,yes,yes,,,Transition Metal,1.8,2.28,8.9587,2.15E+01,2045.15,4098,32,Ulloa/Wood,1735,0.133,6,
79,Gold,Au,196.967,118,79,79,6,11,solid,,yes,yes,,,Transition Metal,1.8,2.54,9.2255,1.93E+01,1337.73,3129,21,Prehistoric,,0.129,6,
80,Mercury,Hg,200.59,121,80,80,6,12,liq,,yes,yes,,,Transition Metal,1.8,2,10.4375,1.35E+01,234.43,630,26,Prehistoric,,0.14,6,
81,Thallium,Tl,204.383,123,81,81,6,13,solid,,yes,yes,,,Metal,2.1,2.04,6.1082,1.19E+01,577.15,1746,28,Crookes,1861,0.129,6,3
82,Lead,Pb,207.2,125,82,82,6,14,solid,,yes,yes,,,Metal,1.8,2.33,7.4167,1.13E+01,600.75,2022,29,Prehistoric,,0.129,6,4
83,Bismuth,Bi,208.98,126,83,83,6,15,solid,,yes,yes,,,Metal,1.6,2.02,7.2856,9.81E+00,544.67,1837,19,Geoffroy the Younger,1753,0.122,6,5
84,Polonium,Po,210,126,84,84,6,16,solid,yes,yes,,,yes,Metalloid,1.5,2,8.417,9.32E+00,527.15,1235,34,Curie,1898,,6,6
85,Astatine,At,210,125,85,85,6,17,solid,yes,yes,,yes,,Noble Gas,1.4,2.2,9.3,7.00E+00,575.15,610,21,Corson et al.,1940,,6,7
86,Radon,Rn,222,136,86,86,6,18,gas,yes,yes,yes,,,Alkali Metal,1.3,,10.7485,9.73E-03,202.15,211.3,20,Dorn,1900,0.094,6,8
87,Francium,Fr,223,136,87,87,7,1,solid,yes,yes,yes,,,Alkaline Earth Metal,,0.7,4.0727,1.87E+00,300.15,950,21,Perey,1939,,7,1
88,Radium,Ra,226,138,88,88,7,2,solid,yes,yes,yes,,,Actinide,,0.9,5.2784,5.50E+00,973.15,2010,15,Pierre and Marie Curie,1898,,7,2
89,Actinium,Ac,227,138,89,89,7,3,solid,yes,yes,yes,,,Actinide,,1.1,5.17,1.01E+01,1323.15,3471,11,Debierne/Giesel,1899,0.12,7,
90,Thorium,Th,232.038,142,90,90,7,,solid,yes,yes,yes,,,Actinide,,1.3,6.3067,1.17E+01,2028.15,5061,12,Berzelius,1828,0.113,7,
91,Protactinium,Pa,231.036,140,91,91,7,,solid,yes,yes,yes,,,Actinide,,1.5,5.89,1.54E+01,1873.15,4300,14,Hahn and Meitner,1917,,7,
92,Uranium,U,238.029,146,92,92,7,,solid,yes,yes,yes,,,Actinide,,1.38,6.1941,1.90E+01,1405.15,4404,15,Peligot,1841,0.116,7,
93,Neptunium,Np,237,144,93,93,7,,artificial,yes,,yes,,,Actinide,,1.36,6.2657,2.05E+01,913.15,4273,153,McMillan and Abelson,1940,,7,
94,Plutonium,Pu,244,150,94,94,7,,artificial,yes,,yes,,,Actinide,,1.28,6.0262,1.98E+01,913.15,3501,163,Seaborg et al.,1940,,7,
95,Americium,Am,243,148,95,95,7,,artificial,yes,,yes,,,Actinide,,1.3,5.9738,1.37E+01,1267.15,2880,133,Seaborg et al.,1944,,7,
96,Curium,Cm,247,151,96,96,7,,artificial,yes,,yes,,,Actinide,,1.3,5.9915,1.35E+01,1340.15,3383,133,Seaborg et al.,1944,,7,
97,Berkelium,Bk,247,150,97,97,7,,artificial,yes,,yes,,,Actinide,,1.3,6.1979,1.48E+01,1259.15,983,83,Seaborg et al.,1949,,7,
98,Californium,Cf,251,153,98,98,7,,artificial,yes,,yes,,,Actinide,,1.3,6.2817,1.51E+01,1925.15,1173,123,Seaborg et al.,1950,,7,
99,Einsteinium,Es,252,153,99,99,7,,artificial,yes,,yes,,,Actinide,,1.3,6.42,1.35E+01,1133.15,,123,Ghiorso et al.,1952,,7,
100,Fermium,Fm,257,157,100,100,7,,artificial,yes,,yes,,,Actinide,,1.3,6.5,,,,103,Ghiorso et al.,1953,,7,
101,Mendelevium,Md,258,157,101,101,7,,artificial,yes,,yes,,,Actinide,,1.3,6.58,,,,33,Ghiorso et al.,1955,,7,
102,Nobelium,No,259,157,102,102,7,,artificial,yes,,yes,,,Actinide,,1.3,6.65,,,,73,Ghiorso et al.,1958,,7,
103,Lawrencium,Lr,262,159,103,103,7,,artificial,yes,,yes,,,Actinide,,,,,,,203,Ghiorso et al.,1961,,7,
104,Rutherfordium,Rf,261,157,104,104,7,4,artificial,yes,,yes,,,Transactinide,,,,1.81E+01,,,,Ghiorso et al.,1969,,7,
105,Dubnium,Db,262,157,105,105,7,5,artificial,yes,,yes,,,Transactinide,,,,3.90E+01,,,,Ghiorso et al.,1970,,7,
106,Seaborgium,Sg,266,160,106,106,7,6,artificial,yes,,yes,,,Transactinide,,,,3.50E+01,,,,Ghiorso et al.,1974,,7,
107,Bohrium,Bh,264,157,107,107,7,7,artificial,yes,,yes,,,Transactinide,,,,3.70E+01,,,,Armbruster and M�nzenberg,1981,,7,
108,Hassium,Hs,267,159,108,108,7,8,artificial,yes,,yes,,,Transactinide,,,,4.10E+01,,,,Armbruster and M�nzenberg,1983,,7,
109,Meitnerium,Mt,268,159,109,109,7,9,artificial,yes,,yes,,,Transactinide,,,,3.50E+01,,,,"GSI, Darmstadt, West Germany",1982,,7,
110,Darmstadtium ,Ds ,271,161,110,110,7,10,artificial,yes,,yes,,,Transactinide,,,,,,,,,1994,,7,
111,Roentgenium ,Rg ,272,161,111,111,7,11,artificial,yes,,yes,,,Transactinide,,,,,,,,,1994,,7,
112,Copernicium ,Cn ,285,173,112,112,7,12,artificial,yes,,yes,,,Transactinide,,,,,,,,,1996,,7,
113,Nihonium,Nh,284,171,113,113,7,13,artificial,yes,,yes,,,,,,,,,,,,2004,,7,3
114,Flerovium,Fl,289,175,114,114,7,14,artificial,yes,,yes,,,Transactinide,,,,,,,,,1999,,7,4
115,Moscovium,Mc,288,173,115,115,7,15,artificial,yes,,yes,,,,,,,,,,,,2010,,7,5
116,Livermorium,Lv,292,176,116,116,7,16,artificial,yes,,yes,,,Transactinide,,,,,,,,,2000,,7,6
117,Tennessine,Ts,295,178,117,117,7,17,artificial,yes,,,yes,,,,,,,,,,,2010,,7,7
118,Oganesson,Og,294,176,118,118,7,18,artificial,yes,,,yes,,Noble Gas,,,,,,,,,2006,,7,8
"""),
    names=[
        "AtomicNumber",
        "Element",
        "Symbol",
        "AtomicMass",
        "NumberofNeutrons",
        "NumberofProtons",
        "NumberofElectrons",
        "Period",
        "Group",
        "Phase",
        "Radioactive",
        "Natural",
        "Metal",
        "Nonmetal",
        "Metalloid",
        "Type",
        "AtomicRadius",
        "Electronegativity",
        "FirstIonization",
        "Density",
        "MeltingPoint",
        "BoilingPoint",
        "NumberOfIsotopes",
        "Discoverer",
        "Year",
        "SpecificHeat",
        "NumberofShells",
        "NumberofValence",
    ],
)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def get_symbol_from_mass(m):
    return PT.loc[np.abs(PT["AtomicMass"] - m) < 0.1, "Symbol"].values[0]

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def get_vel_CONTCAR(file: str = "CONTCAR") -> np.ndarray:
    """
    Read velocities from a VASP CONTCAR file.
    Velocities units: Å/fs
    """
    # velocities are in CONTCAR file after the first blank line afterward the positions
    with open(file, "r") as f:
        lines = f.readlines()
    Natoms = sum([int(n) for n in lines[6].split()])
    Vel = np.zeros(shape=(Natoms, 3))
    if len(lines) - Natoms >= 9 + Natoms:
        # velocities are in the last block of lines
        lines = lines[10 + Natoms:]
        for i, line in enumerate(lines):
            Vel[i, :] = [float(x) * 1e3 for x in line.split()]
    return Vel

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def read_xyz(filename, index=slice(None)):
    """
    Read a file in xyz format and return structures and comments for the specified slice.
    Comment lines contain the cell parameters.
    """
    images = []
    comments = []
    current_block = 0

    # Handle slice parameters
    start = index.start or 0
    stop = index.stop or float('inf')
    step = index.step or 1

    # Get file size for progress bar
    import os
    file_size = os.path.getsize(filename)

    with open(filename, "r") as f:
        with tqdm(total=file_size, unit='B', unit_scale=True, desc=f"Reading {filename}", file=sys.stderr) as pbar:
            while True:
                pos_before = f.tell()

                # Read number of atoms line
                line = f.readline()
                if not line:  # End of file
                    break

                line = line.strip()
                if not line:
                    pbar.update(f.tell() - pos_before)
                    continue

                try:
                    natoms = int(line)
                except ValueError:
                    pbar.update(f.tell() - pos_before)
                    continue

                # Read comment/cell parameters line
                comment_line = f.readline().strip()

                # Check if this block is in the desired slice
                should_process = current_block in range(start, int(
                    stop) if stop != float('inf') else current_block + 1)[::step]

                if should_process:
                    cellparams = [float(i) for i in comment_line.split()]
                    if len(cellparams) == 6:
                        cell = Cell.fromcellpar(cellparams)
                    else:
                        cell = None

                    symbols = []
                    positions = []

                # Read atom lines
                for _ in range(natoms):
                    atom_line = f.readline()
                    if should_process:
                        parts = atom_line.split()[:4]
                        symbol, x, y, z = parts
                        symbol = symbol.lower().capitalize()
                        symbols.append(symbol)
                        positions.append([float(x), float(y), float(z)])

                # Create Atoms object only for processed blocks
                if should_process:
                    images.append(
                        Atoms(
                            symbols=symbols,
                            positions=positions,
                            cell=cell,
                            pbc=[True, True, True]
                        )
                    )
                    comments.append(comment_line)

                current_block += 1

                # Update progress bar
                pbar.update(f.tell() - pos_before)

                # Early exit if we've passed the stop point
                if stop != float('inf') and current_block >= stop:
                    break

    return images, comments

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def write_xyz(fileobj, images, fmt="%22.15f"):
    """
    Write a file in xyz format and add the cell parameters in the comment line
    """
    for atoms in tqdm(enumerate(images), total=len(images), desc="Writing xyz file", file=sys.stderr):
        natoms = len(atoms)
        cell = atoms.cell.cellpar()
        cell = "    ".join([str(i) for i in cell])
        fileobj.write(f"{natoms}\n{cell}\n")
        for s, (x, y, z) in zip(atoms.symbols, atoms.positions):
            fileobj.write("%-2s %s %s %s\n" % (s, fmt % x, fmt % y, fmt % z))

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def parse_ACEpickle(file_path, index=slice(None)):
    try:
        df = pd.read_pickle(file_path, compression='gzip')
    except:
        df = pd.read_pickle(file_path)
    df = df.iloc[index]
    atoms = []
    comments = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Parsing ACE pickle file", unit="struct", file=sys.stderr):
        struct = row['ase_atoms']
        energy = row['energy']
        forces = row['forces']
        comment = row['name']
        struct.calc = SinglePointCalculator(
            atoms=struct, forces=forces, energy=energy
        )
        atoms.append(struct)
        comments.append(comment)
    return atoms, comments

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def atomsList2DataFrame(atoms, names):
    data = {'name': [],
            'energy': [],
            'forces': [],
            'ase_atoms': [],
            'energy_corrected': [],
            'energy_corrected_per_atom': []}
    for struct, comment in zip(atoms, names):
        data['name'].append(comment)
        energy = struct.get_potential_energy()
        data['energy'].append(energy)
        data['forces'].append(struct.get_forces())
        data['energy_corrected'].append(energy)
        data['energy_corrected_per_atom'].append(energy / len(struct))
        struct.calc = None
        data['ase_atoms'].append(struct)
    df = pd.DataFrame(data)
    return df

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def save_pckl_gzip(atoms, comments, file_path):
    data = atomsList2DataFrame(atoms, comments)
    data.to_pickle(file_path, compression='gzip', protocol=4)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def write_MACExyz(atoms, comments, file_path):
    data = atomsList2DataFrame(atoms, comments)
    with open(file_path, 'w') as f:
        for _, row in tqdm(data.iterrows(), total=len(data), desc="Writing MACE xyz file", file=sys.stderr):
            atoms = row['ase_atoms']
            num_atoms = len(atoms)
            pbcs = ' '.join(['T' if p else 'F' for p in atoms.get_pbc()])
            lat = ' '.join([f"{x}" for x in atoms.get_cell().flatten()])
            f.write(f"{num_atoms}\n")
            f.write(
                f'Lattice="{lat}" Properties=species:S:1:pos:R:3:REF_forces:R:3 name={row["name"]} REF_energy={row["energy"]} pbc="{pbcs}"\n')
            forces = row['forces']
            for symbol, position, force in zip(atoms.get_chemical_symbols(), atoms.get_positions(), forces):
                f.write(
                    f"{symbol}{position[0]:17.8f}{position[1]:17.8f}{position[2]:17.8f}{force[0]:17.8f}{force[1]:17.8f}{force[2]:17.8f}\n")

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def parse_MACExyz(file_path, index=slice(None)):
    atoms = []
    comments = []
    current_block = 0

    # Handle slice parameters
    start = index.start or 0
    stop = index.stop or float('inf')
    step = index.step or 1

    # Get file size for progress bar
    file_size = os.path.getsize(file_path)

    with open(file_path, 'r') as f:
        with tqdm(total=file_size, unit='B', unit_scale=True, desc=f"Reading {file_path}", file=sys.stderr) as pbar:
            while True:
                pos_before = f.tell()

                # Read number of atoms line
                line = f.readline()
                if not line:  # End of file
                    break

                line = line.strip()
                if not line:
                    pbar.update(f.tell() - pos_before)
                    continue

                # Check if this block is in the desired slice
                should_process = current_block in range(start, int(
                    stop) if stop != float('inf') else current_block + 1)[::step]

                try:
                    num_atoms = int(line)
                except ValueError:
                    pbar.update(f.tell() - pos_before)
                    continue

                # Read lattice/properties line
                lattice_line = f.readline().strip()

                if should_process:
                    # Parse metadata
                    lattice_str = lattice_line.split(
                        'Lattice="')[1].split('"')[0]
                    name = lattice_line.split('name=')[1].split(
                        ' category=')[0].strip('"')
                    category_split = lattice_line.split('category=')
                    if len(category_split) > 1:
                        category = category_split[1].split(' ')[0].strip('"')
                    else:
                        category = 'unknown_category'
                    energy = float(lattice_line.split(
                        'energy=')[1].split(' ')[0])
                    pbc = [s.strip() == 'T' for s in lattice_line.split(
                        'pbc="')[1].strip('"').split()]

                    # Parse lattice
                    lattice = np.array(lattice_str.split()).reshape(
                        3, 3).astype(float)

                    # Initialize arrays
                    forces = []
                    positions = []
                    symbols = []

                # Read atom lines
                for _ in range(num_atoms):
                    atom_line = f.readline().strip()
                    if should_process:
                        atom_data = atom_line.split()
                        symbol = atom_data[0]
                        x, y, z = map(float, atom_data[1:4])
                        fx, fy, fz = map(float, atom_data[4:7])
                        symbols.append(symbol)
                        positions.append([x, y, z])
                        forces.append([fx, fy, fz])

                # Create ASE Atoms object only for processed blocks
                if should_process:
                    struct = Atoms(symbols=symbols,
                                   positions=positions,
                                   cell=lattice,
                                   pbc=pbc)
                    struct.calc = SinglePointCalculator(
                        atoms=struct, forces=forces, energy=energy
                    )
                    comment = f"{category}, {name}"
                    atoms.append(struct)
                    comments.append(comment)

                current_block += 1

                # Update progress bar
                pbar.update(f.tell() - pos_before)

                # Early exit if we've passed the stop point
                if stop != float('inf') and current_block >= stop:
                    pbar.update(file_size - f.tell())  # Complete the bar
                    break

    return atoms, comments

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def write_lammps_traj(structure, outfile=sys.stdout, energies=False, forces=False, allotrope=False):
    """Write a list of ASE Atoms objects to a LAMMPS trajectory file.

    Parameters
    ----------
        structure (list of Atoms): List of ASE Atoms objects to write.
        outfile (str or file-like, optional): Output file path or file-like object. Default is sys.stdout.
        energies (list or None, optional): List of energies corresponding to each Atoms object. Default is None.
        forces (list or None, optional): List of forces corresponding to each Atoms object. Default is None.
        allotrope (bool, optional): Whether to include allotrope information. Default is False.
    """
    if not structure:
        sys.stderr.write("Warning: Empty structure provided.\n")
        return

    if outfile != sys.stdout:
        outfile = open(outfile, "w")

    for it, struc in tqdm(enumerate(structure), total=len(structure),
                          desc="Writing lammpstrj file", file=sys.stderr):

        # Get cell vectors (3x3 matrix where each row is a vector)
        cell_matrix = struc.cell.array

        allotrope_info = ['']*len(struc)
        if allotrope:
            allotrope_data = struc.arrays.get('allotrope', None)
            if allotrope_data is not None:
                if isinstance(allotrope_data, (list, np.ndarray)) and len(allotrope_data) == len(struc):
                    allotrope_info = [al if al !=
                                      '0' else '' for al in allotrope_data]
                else:
                    sys.stderr.write(
                        f"Warning: 'allotrope' array is missing or malformed for structure {it}. Using empty strings.\n")

        outfile.write("ITEM: TIMESTEP\n")
        outfile.write(f"{it}\n")
        outfile.write("ITEM: NUMBER OF ATOMS\n")
        outfile.write(f"{len(struc)}\n")

        # Check if orthogonal
        a, b, c, alpha, beta, gamma = struc.cell.cellpar()

        if not (np.isclose(alpha, 90) and np.isclose(beta, 90) and np.isclose(gamma, 90)):
            # Convert to LAMMPS lower-triangular format
            # LAMMPS expects:
            #   a = (lx, 0, 0)
            #   b = (xy, ly, 0)
            #   c = (xz, yz, lz)

            a_vec = cell_matrix[0]
            b_vec = cell_matrix[1]
            c_vec = cell_matrix[2]

            # Calculate LAMMPS box parameters
            lx = np.linalg.norm(a_vec)
            xy = np.dot(b_vec, a_vec) / lx
            ly = np.sqrt(np.dot(b_vec, b_vec) - xy**2)
            xz = np.dot(c_vec, a_vec) / lx
            yz = (np.dot(b_vec, c_vec) - xy * xz) / ly
            lz = np.sqrt(np.dot(c_vec, c_vec) - xz**2 - yz**2)

            # LAMMPS bounds format: xlo xhi xy, ylo yhi xz, zlo zhi yz
            outfile.write("ITEM: BOX BOUNDS xy xz yz pp pp pp\n")
            outfile.write(f"{0.0:15.9e} {lx:15.9e} {xy:15.9e}\n")
            outfile.write(f"{0.0:15.9e} {ly:15.9e} {xz:15.9e}\n")
            outfile.write(f"{0.0:15.9e} {lz:15.9e} {yz:15.9e}\n")

            # Build LAMMPS cell in lower triangular form
            lammps_cell = np.array([
                [lx, 0.0, 0.0],
                [xy, ly, 0.0],
                [xz, yz, lz]
            ])

            # Transform positions: new_pos = old_pos @ (old_cell^-1 @ new_cell)^T
            # This maps fractional coordinates through the new cell
            transform = lammps_cell @ np.linalg.inv(cell_matrix)
            positions = struc.positions @ transform.T

        else:
            # Orthogonal box
            outfile.write("ITEM: BOX BOUNDS pp pp pp\n")
            outfile.write(f"{0.0:15.9e} {a:15.9e}\n")
            outfile.write(f"{0.0:15.9e} {b:15.9e}\n")
            outfile.write(f"{0.0:15.9e} {c:15.9e}\n")
            positions = struc.positions

        strvel = "" if np.sum(np.abs(struc.get_velocities())
                              ) < 1e-10 else " vx vy vz"
        stre = ""
        if energies:
            stre = " c_pe" if energies else ""
            Es = struc.get_potential_energies()
        strf = ""
        if forces:
            strf = " fx fy fz"
            Fs = struc.forces()
        velstr = ''
        c_pestr = ''
        f_str = ''
        # Write atomic coordinates
        outfile.write(f"ITEM: ATOMS id element x y z{strvel}{stre}{strf}\n")
        if np.sum(np.abs(struc.get_velocities())) > 1e-10:
            velocities = struc.get_velocities()
            if not (np.isclose(alpha, 90) and np.isclose(beta, 90) and np.isclose(gamma, 90)):
                velocities = velocities @ transform.T
        for i, (at, pos, info) in enumerate(
            zip(struc.get_chemical_symbols(), positions, allotrope_info)
        ):
            x, y, z = pos
            if np.sum(np.abs(struc.get_velocities())) > 1e-10:
                vx, vy, vz = velocities[i]
                velstr = f" {vx:12.6e} {vy:12.6e} {vz:12.6e}"
            if energies:
                try:
                    energy_value = float(Es[i])
                    c_pestr = f" {energy_value:12.6e}"
                except (ValueError, TypeError):
                    c_pestr = ""
            else:
                c_pestr = ""
            if forces:
                try:
                    fx, fy, fz = Fs[i]
                    f_str = f" {fx:12.6e} {fy:12.6e} {fz:12.6e}"
                except (ValueError, TypeError, IndexError):
                    f_str = ""
            outfile.write(
                f"{i + 1:>4d} {at:>2s}{info} {x:12.6e} {y:12.6e} {z:12.6e}{velstr}{c_pestr}{f_str}\n"
            )

    if outfile != sys.stdout:
        outfile.close()

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def write_inputdata(atoms, comments, rmFE=False):
    """
    From a list of Atoms object and comments, write an input.data file to `filename`.
    """
    for atom, comment in tqdm(zip(atoms, comments), total=len(atoms), desc="Writing n2p2 file", file=sys.stderr):
        a, b, c = atom.get_cell() * cdist  # convert Å to Bohr
        if atom.calc is not None and not rmFE:
            forces = np.array(atom.get_forces())
            energy = atom.get_potential_energy()
            energy *= cener  # convert eV to Hartree
            forces *= cforce  # convert eV/Å to Hartree/Bohr
        else:
            forces = np.zeros_like(atom.positions)
            energy = 0.0
        sys.stdout.write("begin\n")
        sys.stdout.write(f"comment {comment}\n")
        sys.stdout.write(
            f"lattice {a[0]:22.14e} {a[1]:22.14e} {a[2]:22.14e}\n")
        sys.stdout.write(
            f"lattice {b[0]:22.14e} {b[1]:22.14e} {b[2]:22.14e}\n")
        sys.stdout.write(
            f"lattice {c[0]:22.14e} {c[1]:22.14e} {c[2]:22.14e}\n")
        for at, pos, (fx, fy, fz) in zip(
            atom.get_chemical_symbols(), atom.positions, forces
        ):
            x, y, z = pos * cdist  # convert Å to Bohr
            sys.stdout.write(
                f"atom {x:22.14e} {y:22.14e} {z:22.14e} {at:s} {'0.0':s} {'0.0':s} {fx:22.14e} {fy:22.14e} {fz:22.14e}\n"
            )
        sys.stdout.write(f"energy {energy:22.14e}\n")
        sys.stdout.write("charge 0\n")
        sys.stdout.write("end\n")

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def read_inputdata(filename: str, index=slice(None)):
    """
    Read input.data file and return a list of Atoms objects and the comments for the specified slice.
    """
    atoms = []
    comments = []
    current_block = 0
    block_lines = []
    in_block = False
    file_size = os.path.getsize(filename)

    with open(filename, 'r') as f:
        with tqdm(total=file_size, unit='B', unit_scale=True, desc=f"Reading {filename}", file=sys.stderr) as pbar:
            while True:
                pos_before = f.tell()
                line = f.readline()

                if not line:  # End of file
                    break

                line = line.strip()

                if line == 'begin':
                    in_block = True
                    block_lines = [line]
                elif line == 'end':
                    in_block = False
                    block_lines.append(line)
                    # Only process blocks in the desired slice
                    start = index.start or 0
                    stop = index.stop or float('inf')
                    step = index.step or 1

                    if current_block in range(start, int(stop) if stop != float('inf') else current_block + 1)[::step]:
                        # Parse the block
                        lattice = []
                        pos = []
                        forces = []
                        symb = []
                        comment = None
                        for block_line in block_lines:
                            if 'lattice' in block_line:
                                lattice.append(
                                    [float(x)/cdist for x in block_line.split()[1:4]])
                            elif 'comment' in block_line:
                                comment = block_line.split(
                                    'comment ')[1].strip()
                            elif 'atom' in block_line:
                                _, x, y, z, symbol, _, _, fx, fy, fz = block_line.split()
                                symbol = symbol.lower().capitalize()
                                symb.append(symbol)
                                pos.append([float(x) / cdist,
                                            float(y) / cdist,
                                            float(z) / cdist])
                                forces.append(
                                    [float(fx) / cforce,
                                     float(fy) / cforce,
                                     float(fz) / cforce]
                                )
                            elif 'energy' in block_line:
                                energy = float(block_line.split()[1]) / cener
                        # Create Atoms object
                        at = Atoms(symbols=symb,
                                   positions=np.array(pos),
                                   cell=np.array(lattice),
                                   pbc=True)
                        at.calc = SinglePointCalculator(
                            atoms=at,
                            forces=forces,
                            energy=energy
                        )
                        atoms.append(at)
                        comments.append(comment)
                    block_lines = []
                    current_block += 1

                    # Early exit if we've passed the stop point
                    if stop != float('inf') and current_block >= stop:
                        pbar.update(file_size - pos_before)  # Complete the bar
                        break
                elif in_block:
                    block_lines.append(line)

                # Update progress bar
                pbar.update(f.tell() - pos_before)

    return atoms, comments

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def compute_velocities(atoms, dt=1.0):
    """
    Compute velocities from the positions of the atoms at two different times.
    dt is the time difference between the two snapshots, in fs.
    Velocities units: Å/ps

    Parameters
    ----------
        atoms (list): List of ASE Atoms objects
        dt (float): Time difference between snapshots in fs

    Returns
    -------
        atoms (list): List of ASE Atoms objects with velocities set
    """
    if len(atoms) < 2:
        sys.stderr.write(
            f"{bold}{orange}\nVelocities could not be computed!\nAt least two images are necessary to compute them...{normal}\n"
        )
        return atoms
    if len(atoms) >= 3:
        sys.stderr.write(
            "Computing the velocities using the Verlet algorithm.\n")
        # works in the general triclinic case
        for i in range(1, len(atoms) - 1):
            atoms_minus_t = atoms[i - 1]
            atoms_plus_t = atoms[i + 1]
            cell = atoms[i].cell
            inv_cell = np.linalg.inv(cell)

            dx = atoms_plus_t.positions[:, 0] - atoms_minus_t.positions[:, 0]
            dy = atoms_plus_t.positions[:, 1] - atoms_minus_t.positions[:, 1]
            dz = atoms_plus_t.positions[:, 2] - atoms_minus_t.positions[:, 2]

            # Combine displacements into a single array
            dxyz = np.vstack((dx, dy, dz)).T

            # Apply periodic boundary conditions
            dxyz = np.dot(dxyz, inv_cell)  # Convert to fractional coordinates
            dxyz -= np.round(dxyz)  # Apply PBC in fractional coordinates
            dxyz = np.dot(dxyz, cell)  # Convert back to Cartesian coordinates

            vel = dxyz * 1e3 / 2 / dt  # Å/fs to Å/ps
            atoms[i].set_velocities(vel)
        return atoms[1:-1]
    else:
        sys.stderr.write(
            "Computing the velocities using the finite difference method between the two images.\n"
        )
        cell = atoms[0].cell
        inv_cell = np.linalg.inv(cell)

        dx = atoms[1].positions[:, 0] - atoms[0].positions[:, 0]
        dy = atoms[1].positions[:, 1] - atoms[0].positions[:, 1]
        dz = atoms[1].positions[:, 2] - atoms[0].positions[:, 2]

        # Combine displacements into a single array
        dxyz = np.vstack((dx, dy, dz)).T

        # Apply periodic boundary conditions
        dxyz = np.dot(dxyz, inv_cell)  # Convert to fractional coordinates
        dxyz -= np.round(dxyz)  # Apply PBC in fractional coordinates
        dxyz = np.dot(dxyz, cell)  # Convert back to Cartesian coordinates

        vel = dxyz * 1e3 / 2 / dt  # Å/fs to Å/ps
        atoms[0].set_velocities(vel)
        return [atoms[0]]

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def parse_slice(value):
    """
    Parses a `slice()` from string, like `start:stop:step`.
    """
    if value:
        if value.lstrip("+-").isnumeric():
            if value == "-1":
                return slice(-1, None)
            else:
                return slice(int(value), int(value) + 1)
        parts = value.split(":")
        if len(parts) == 1:
            # slice(stop)
            parts = [None, parts[0]]
        # else: slice(start, stop[, step])
    else:
        # slice()
        parts = []
    return slice(*[int(p) if p else None for p in parts])


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def write_lammps(
    name,
    atoms,
    units="metal",
    atom_style="atomic",
    comments="",
    velocities: np.ndarray = None,
):
    """Write atomic structure data to a LAMMPS data file.

    Parameters
    ----------
        name (str): Name of the output file
        atoms (Atom object): ASE Atom object
        units (str): LAMMPS units style
        atom_style (str): LAMMPS atom style
        velocities (array): In case we want to add a initial velocities to the structure.

    Returns
    -------
        None
    """
    if name is None:
        fd = sys.stdout
    else:
        if os.path.dirname(name) != "":
            os.makedirs(os.path.dirname(name), exist_ok=True)
        fd = open(name, "w")
        comments = name

    if isinstance(atoms, list):
        if len(atoms) > 1:
            raise ValueError(
                "Can only write one configuration to a lammps data file!")
        atoms = atoms[0]

    if hasattr(fd, "name"):
        fd.write(f"{comments}\n\n")
    else:
        fd.write("\n\n")

    symbols = atoms.get_chemical_symbols()
    n_atoms = len(symbols)
    fd.write("{0} \t atoms \n".format(n_atoms))

    species = set(symbols)
    if "B" in species:
        species.remove("B")
        species = ["B"] + sorted(species)
    else:
        species = sorted(species)
    n_atom_types = len(species)
    fd.write("{0}  atom types\n".format(n_atom_types))

    p = Prism(atoms.get_cell())

    # Get cell parameters and convert from ASE units to LAMMPS units
    xhi, yhi, zhi, xy, xz, yz = convert(
        p.get_lammps_prism(), "distance", "ASE", units)

    fd.write("0.0 {0:23.17g}  xlo xhi\n".format(xhi))
    fd.write("0.0 {0:23.17g}  ylo yhi\n".format(yhi))
    fd.write("0.0 {0:23.17g}  zlo zhi\n".format(zhi))

    if p.is_skewed():
        fd.write("{0:23.17g} {1:23.17g} {2:23.17g}  xy xz yz\n".format(xy, xz, yz))
    fd.write("\n\n")

    # Write (unwrapped) atomic positions.  If wrapping of atoms back into the
    # cell along periodic directions is desired, this should be done manually
    # on the Atoms object itself beforehand.
    fd.write("Masses \n\n")
    for i in range(n_atom_types):
        m = Atoms(species[i]).get_masses()[0]
        fd.write(str(i + 1) + "   " + str(m) + "\n")

    fd.write("\n\nAtoms \n\n")
    pos = p.vector_to_lammps(atoms.get_positions(), wrap=False)

    if atom_style == "atomic":
        for i, r in enumerate(pos):
            # Convert position from ASE units to LAMMPS units
            r = convert(r, "distance", "ASE", units)
            s = species.index(symbols[i]) + 1
            fd.write(
                "{0:>6} {1:>3} {2:23.17g} {3:23.17g} {4:23.17g}\n".format(
                    *(i + 1, s) + tuple(r)
                )
            )
    elif atom_style == "charge":
        charges = atoms.get_initial_charges()
        for i, (q, r) in enumerate(zip(charges, pos)):
            # Convert position and charge from ASE units to LAMMPS units
            r = convert(r, "distance", "ASE", units)
            q = convert(q, "charge", "ASE", units)
            s = species.index(symbols[i]) + 1
            fd.write(
                "{0:>6} {1:>3} {2:>5} {3:23.17g} {4:23.17g} {5:23.17g}\n".format(
                    *(i + 1, s, q) + tuple(r)
                )
            )
    elif atom_style == "full":
        charges = atoms.get_initial_charges()
        # The label 'mol-id' has apparenlty been introduced in read earlier,
        # but so far not implemented here. Wouldn't a 'underscored' label
        # be better, i.e. 'mol_id' or 'molecule_id'?
        if atoms.has("mol-id"):
            molecules = atoms.get_array("mol-id")
            if not np.issubdtype(molecules.dtype, np.integer):
                raise TypeError(
                    (
                        "If 'atoms' object has 'mol-id' array, then"
                        " mol-id dtype must be subtype of np.integer, and"
                        " not {:s}."
                    ).format(str(molecules.dtype))
                )
            if (len(molecules) != len(atoms)) or (molecules.ndim != 1):
                raise TypeError(
                    (
                        "If 'atoms' object has 'mol-id' array, then"
                        " each atom must have exactly one mol-id."
                    )
                )
        else:
            # Assigning each atom to a distinct molecule id would seem
            # preferableabove assigning all atoms to a single molecule id per
            # default, as done within ase <= v 3.19.1. I.e.,
            # molecules = np.arange(start=1, stop=len(atoms)+1, step=1, dtype=int)
            # However, according to LAMMPS default behavior,
            molecules = np.zeros(len(atoms), dtype=int)
            # which is what happens if one creates new atoms within LAMMPS
            # without explicitly taking care of the molecule id.
            # Quote from docs at https://lammps.sandia.gov/doc/read_data.html:
            #    The molecule ID is a 2nd identifier attached to an atom.
            #    Normally, it is a number from 1 to N, identifying which
            #    molecule the atom belongs to. It can be 0 if it is a
            #    non-bonded atom or if you don't care to keep track of molecule
            #    assignments.

        for i, (m, q, r) in enumerate(zip(molecules, charges, pos)):
            # Convert position and charge from ASE units to LAMMPS units
            r = convert(r, "distance", "ASE", units)
            q = convert(q, "charge", "ASE", units)
            s = species.index(symbols[i]) + 1
            fd.write(
                "{0:>6} {1:>3} {2:>3} {3:>5} {4:23.17g} {5:23.17g} {6:23.17g}\n".format(
                    *(i + 1, m, s, q) + tuple(r)
                )
            )
    else:
        raise NotImplementedError

    if isinstance(velocities, np.ndarray):
        if velocities.shape[0] == len(atoms):
            # atom_style is atomic by default
            # velocity in metal unit in lammps : Angstroms/picosecond
            # see: https://docs.lammps.org/units.html
            # in vasp : Angstroms/femtosecond

            fd.write("\n\nVelocities \n\n")

            for i, (vx, vy, vz) in enumerate(velocities):
                # element = list(species).index(atoms[i].symbol) + 1

                fd.write(
                    "{0:>6} {1:23.17g} {2:23.17g} {3:23.17g}\n".format(
                        i + 1, vx, vy, vz
                    )
                )

        else:
            raise ValueError(
                f"Dimenssion missmatch. velocities must have shape of {len(atoms)} by 3."
            )

    fd.flush()
    if fd is not sys.stdout:
        fd.close()

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def read(filename: str, slice: str, cv=False, dt=1.0) -> list:
    """
    Wrapper function to read a file and return the structure at index `index`.
    Returns a list of Atoms objects and the type of structure read.

    Parameters
    ----------
        filename (str): Name of the input file
        slice (str): Slice of the structures to read, in the form `start:stop:step` or a single integer. If empty, all structures are read.
        cv (bool): If True, compute velocities from the positions of the atoms at two different times.
        dt (float): Time difference between the two snapshots, in fs.

    Returns
    -------
        struc (list): List of Atoms objects
        structype (str): Type of structure read (POSCAR, OUTCAR, lammps, lammpstrj, xyz, traj, n2p2)
        comments (list): List of comments for each structure

    Examples
    --------
    ```python
        struc, structype, comments = read("POSCAR", ":")
        struc, structype, comments = read("OUTCAR", "0:10:2")
        struc, structype, comments = read("data.lammps", "5")
    ```
    """
    slice = parse_slice(slice)

    n2p2 = None
    if ".data" in filename:
        n2p2 = 0
        with open(filename, "r") as f:
            for line in f:
                if "begin" in line:
                    n2p2 = 1
                    break

    # Read the input file
    if not isfile(filename):
        sys.stderr.write(
            f"\n{bold}{orange}{filename} file does not exist!{normal}\n\n")
        sys.exit()
    sys.stderr.write(f"Reading {filename}...")
    sys.stderr.flush()
    if "POSCAR" in filename or "CONTCAR" in filename:
        struc = [read_vasp(filename)]
        comments = [f"{filename} {i}" for i in range(len(struc))]
        structype = "POSCAR"
    elif n2p2 == 0:
        struc = [read_lammps_data(filename, atom_style="atomic")]
        comments = [f"{filename} {i}" for i in range(len(struc))]
        structype = "lammps"
        # get correct symbols from atomic masses
        for i, atom in enumerate(struc):
            masses = atom.get_masses()
            symbols = [get_symbol_from_mass(m) for m in masses]
            atom.set_chemical_symbols(symbols)
    elif "OUTCAR" in filename:
        struc = read_vasp_out(filename, index=slice)
        comments = [f"{filename} {i}" for i in range(len(struc))]
        structype = "OUTCAR"
    elif ".lammpstrj" in filename:
        struc = read_lammps_dump(filename, index=slice)
        comments = [f"{filename} {i}" for i in range(len(struc))]
        structype = "lammpstrj"
    elif ".xyz" in filename:
        with open(filename, "r") as f:
            first_line = f.readline()
            second_line = f.readline()
        if "Lattice=" in second_line:
            struc, comments = parse_MACExyz(filename, index=slice)
            structype = "MACExyz"
        else:
            struc = read_xyz(filename, index=slice)
            structype = "xyz"
    elif ".traj" in filename:
        struc = ase_read(filename, index=slice)
        comments = [f"{filename} {i}" for i in range(len(struc))]
        structype = "traj"
    elif n2p2 == 1 or ".inp" in filename:
        struc, comments = read_inputdata(filename, index=slice)
        structype = "n2p2"
    elif ".pckl.gzip" in filename or ".pckl" in filename:
        struc, comments = parse_ACEpickle(filename, index=slice)
        structype = "ACEpckl"
    else:
        raise ValueError(f"""{bold}{orange}
Unknown input file format!
Name must contain either:
- 'POSCAR', 'CONTCAR' or 'OUTCAR' for vasp files
- '.data' or '.inp' for NNP input files
- '.lammpstrj' for lammps trajectory files
- '.xyz' for xyz trajectory files
- '.traj' for ASE trajectory files
- '.xyz' for MACE datasets
- '.pckl.gzip' for ACE datasets
- '.pckl' for ACE datasets
{normal}
""")
    sys.stderr.write(f"""
Format  : {orange}{bold}{structype}{normal}
N images: {orange}{bold}{len(struc)}{normal}

""")
    sys.stderr.flush()

    if not isinstance(struc, list):
        struc = [struc]

    if cv:
        struc = compute_velocities(struc, dt=dt)

    if "CONTCAR" in filename:
        vel = get_vel_CONTCAR(filename)
        struc[0].set_velocities(vel)

    return struc, structype, comments
