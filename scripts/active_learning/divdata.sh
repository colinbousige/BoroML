#!/bin/bash

inpfile=$1
NStruct=$2
OutFile=$3

# supprime les lignes vide
#sed -i '/^$/d' $inpfile
# supprime les caractèes d'espacement après end, le caractère $ inclus
# [[:space:]] : Équivalent à : [ \t\r\n\v\f] (espace, tabulation, 
#                             retour chariot, nouvelle ligne, 
#                             tabulation verticale, saut de page)
#sed -i 's/^end[[:space:]]*$/end/' $inpfile

awk -v target=$NStruct '/^end/ { count++ } count % target == 0 ' $inpfile > $OutFile
