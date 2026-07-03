#!/bin/bash

cp $1 temp_input
NStruct=$2
OutFile=$3

# supprime les lignes vide
sed -i '/^$/d' temp_input
# supprime les caractèes d'espacement après end, le caractère $ inclus
# [[:space:]] : Équivalent à : [ \t\r\n\v\f] (espace, tabulation, 
#                             retour chariot, nouvelle ligne, 
#                             tabulation verticale, saut de page)
sed -i 's/^end[[:space:]]*$/end/' temp_input

awk -v target=$NStruct '$0 == "end" { count++ } count <= target; $0 == "end" && count == target { exit }' temp_input > $OutFile

rm -f temp_input