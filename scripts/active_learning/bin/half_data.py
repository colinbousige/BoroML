#!/usr/bin/env python

###############################################################################
# File converter from VASP OUTCAR to input.data format.
# Works also if OUTCAR contains trajectories.
# Tested with VASP 5.2.12
###############################################################################

import numpy as np
import sys

def print_usage():
    sys.stderr.write("USAGE: {0:s} <in_file> <<out_file>>\n".format(sys.argv[0]))
    sys.stderr.write("       <in_file> .... data.inp file name.\n")
    sys.stderr.write("       <out_file> ... data_half.inp file name (optional).\n")
    return

if len(sys.argv) < 2 or sys.argv[1] in ["-?", "-h", "--help"]:
    print_usage()
    sys.exit(1)

file_name = sys.argv[1]
if len(sys.argv) > 2:
    outfile_name = sys.argv[2]
else:
    outfile_name = None

# Read in the whole file first.
f = open(file_name, "r")
lines = [line for line in f]
f.close()

# Open output file or write to stdout.
if outfile_name is not None:
    f = open(outfile_name, "w")
else:
    f = sys.stdout
n=0
m=0
for i in range(len(lines)):
	line = lines[i]
	if "begin" in line:
		n = n+1
	if (n-1)%2==0 :
		f.write(line)
		if "begin" in line:
			m = m + 1
		
print("Number of geometries in the old {:20s} file = {:d}".format(file_name,n))
print("Number of geometries in the new {:20s} file = {:d}".format(outfile_name,m))
f.close()

