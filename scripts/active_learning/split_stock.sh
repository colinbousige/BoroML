#!/bin/bash

input_file=$1
NStruct=$2
output_prefix="stock"

# Supprimer les lignes vides (optionnel, si nécessaire)
#sed -i '/^$/d' "$input_file"
# supprime les caractèes d'espacement après end, le caractère $ inclus
# [[:space:]] : Équivalent à : [ \t\r\n\v\f] (espace, tabulation, 
#                             retour chariot, nouvelle ligne, 
#                             tabulation verticale, saut de page)
#sed -i 's/^end[[:space:]]*$/end/' temp_input
#sed -i 's/^begin[[:space:]]*$/begin/' temp_input

# Utiliser awk pour découper le fichier en une seule passe
awk -v NStruct="$NStruct" '
    BEGIN {
        part = 1
        output_file = "'"$output_prefix"'1.data"
    }
    /^begin/ {
        if (count == NStruct) {
            close(output_file)
            print "Created " output_file " with " count " structures"
            part++
            output_file = "'"$output_prefix"'" part ".data"
            count = 0
        }
        count++
    }
    { print > output_file }
    END {
        output_file = "'"$output_prefix"'" part ".data"
        print "Created " output_file  " with " count " structures"
    }
' $input_file

