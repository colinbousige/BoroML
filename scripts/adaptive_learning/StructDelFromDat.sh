#!/bin/bash

# Usage: ./delete_paragraph.sh stock.data N
# stock.data is the input file, and N is the paragraph number to delete.

file=$1
n=$2

# Check if both arguments are provided
if [[ -z "$file" || -z "$n" ]]; then
  echo "Usage: $0 filename paragraph_number"
  exit 1
fi

# Validate if the file exists
if [[ ! -f "$file" ]]; then
  echo "Error: File '$file' not found."
  exit 1
fi

# Delete the nth paragraph delimited by "begin" and "end"
awk -v para="$n" '
  BEGIN { count = 0; delete_this = 0 }
  /begin/ {
    delete_this = (count == para) ? 1 : 0
    count++;
  }
  !delete_this { print }
  /end/ { delete_this = 0 }
' "$file" > tmpfile && mv tmpfile "$file"

#awk -v para="$n" '                            # pass $n bash variable to "para" variable of awk
#  BEGIN { count = 0; delete_this = 0 }        # initialize awk variables
#  /begin/ {                                   # This block executes whenever a line matches the pattern /begin/ (i.e., it contains the word "begin"). 
#    count++;                                   
#    delete_this = (count == para) ? 1 : 0     # attribute 0 or 1 to delete_this variable wheter count == para condition is met
#  }
#  !delete_this { print }                      # if not delete_this print line
#  /end/ { delete_this = 0 }                   # put delete_this to 0 is end is in line
#' "$file" > tmpfile && mv tmpfile "$file"     # put results in tmpfile and this mv tmpfile to $file