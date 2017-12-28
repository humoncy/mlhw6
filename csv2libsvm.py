#!/usr/bin/env python

"""
Convert CSV file to libsvm format. Works only with numeric variables.
Put -1 as label index (argv[4]) if there are no labels in your file.
Expecting no headers. If present, headers can be skipped with argv[5] == 1.

"""

import sys
import csv

def construct_line( label, line ):
	new_line = []
	if float( label ) == 0.0:
		label = "0"
	new_line.append( label )

	for i, item in enumerate( line ):
		if item == '' or float( item ) == 0.0:
			continue
		new_item = "%s:%s" % ( i + 1, item )
		new_line.append( new_item )
	new_line = " ".join( new_line )
	new_line += "\n"
	return new_line

# ---

input_file = sys.argv[1]
output_file = sys.argv[2]
label_file = sys.argv[3]

try:
	label_index = int( sys.argv[4] )
except IndexError:
	label_index = -1

try:
	skip_headers = int( sys.argv[5] )
except IndexError:
	skip_headers = 0

i = open( input_file, 'rb' )
o = open( output_file, 'wb' )
import numpy as np
l = np.loadtxt(label_file, dtype=int)

reader = csv.reader( i )

if skip_headers:
	print("Skip headers")
	headers = reader.next()

for index, line in enumerate(reader):
	if label_index == -1:
		label = '-1'
		label = l[index].__str__()
	else:
		label = line.pop( label_index )

	new_line = construct_line( label, line )
	o.write( new_line )

