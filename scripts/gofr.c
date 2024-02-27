// Compute G(r) or z-densities time-averaged on a LAMMPS trajectory dump file.
// Outputs in csv format to stdout.
// 
// Compile with: 
// gcc -o GofR gofr.c -lm
// 
// Usage: GofR -in inputfile.lammpstrj [-dr 0.01 -Rmax 10 -tidy 1 -z 0 -skip 0 -max 0] > outfile.txt
// Where:
//   -in   : *.lammpstrj LAMMPS dump file
//   -dr   : step size (Å) (optional, defaults to 0.01)
//   -Rmax : maximum R (Å) for G(r) (optional, defaults to 10)
//   -tidy : tidy output (1) or not (0) (optional, defaults to 1)
//   -z    : project positions along z (1) or not (0) (optional, defaults to 0)
//   -max  : if max > 0, maximum number of images to read (optional, defaults to 0)
//   -skip : if skip > 0, skip this amount of images before reading (optional, defaults to 0)
//   -h    : show the help
//
// Note: probably not the most efficient or clean code, but hey, it works -- at least I get the same values as with VMD.
// 
// Author: Colin BOUSIGE, colin.bousige@cnrs.fr
// Version: 1.0
// Date: 2023/09/29

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define MIN(a,b) (a<b ? a : b)
#define MAX(a,b) (a>b ? a : b)
#define PI 3.14159265359
#define SIGN(x)  ((x > 0) ? 1 : ((x < 0) ? -1 : 0))

char * read_line (char const * const s_filename, unsigned int i_line_num, char ligne[500]);
int nb_lines (char const * const s_filename);
double Dist(double xlo,double xhi,double ylo,double yhi,double zlo,double zhi, 
            double point1, double point2, double point3, double pos1, double pos2, double pos3);
void progressbar(const char *message, int i, int N, int Nchar);
int in(char **arr, int len, char *target);
int which(char **arr, int len, char *target);

int main(int Narg, char **argv){
    // Default values
    double dr=.01;
    double Rmax=10+dr;
    int tidy = 1, Zproj = 0, maxImages=0, skip=0;
    // Other variables
    int i, j, k, l, m, n, N, Nbin, N1 = 0, N2 = 0, id;
    int Nimages, conf, head=9, Ntype=1, exists=0, ti, tj;
    double a, x, y, z,r, charge;
    double xlo,xhi,ylo,yhi,zlo,zhi;
    char text[100], input[100], ligne[500];
    char toprint[30], attyp[3], typex[3], trash[10];
    char **typelist = malloc(100 * sizeof(char*));
    double **atom;
    double Vtot, Vmean=0;
    double *count, *integ, ***countij, ***integij;
    int *Ni, *type;
    FILE *fp;
        
    // Read parameters
    if (Narg > 1) {
        for (i = 1; i < Narg; i++){
            if (!strcmp(argv[i], "-in"))
                sprintf(input, "%s", argv[++i]);
            else if (!strcmp(argv[i], "-dr"))
                dr = atof(argv[++i]);
            else if (!strcmp(argv[i], "-Rmax"))
                Rmax = atof(argv[++i]);
            else if (!strcmp(argv[i], "-tidy"))
                tidy = atoi(argv[++i]);
            else if (!strcmp(argv[i], "-z"))
                Zproj = atoi(argv[++i]);
            else if (!strcmp(argv[i], "-max"))
                maxImages = atoi(argv[++i]);
            else if (!strcmp(argv[i], "-skip"))
                skip = atoi(argv[++i]);
            else if (!strcmp(argv[i], "-h"))
                goto help;
        }
        if (fopen(input, "r") == NULL) {
            printf("No \"%s\" file in the folder dude !\n", input);
            return -1;
        }
    } else {
        help:
        fprintf(stderr, "\n\nGofR:\n\n   Compute G(r) or z-densities time-averaged on a LAMMPS trajectory dump file.\n   Outputs in csv format to stdout.\n");
        fflush(stderr);
        fprintf(stderr, "\nUsage:\n\n   GofR -in inputfile.lammpstrj [-dr %.2lf -Rmax %.2lf -tidy %d -z %d -skip %d -max %d] > outfile.txt\n", dr, Rmax, tidy, Zproj, skip, maxImages);
        fflush(stderr);
        fprintf(stderr, "\nWhere:\n");fflush(stderr);
        fprintf(stderr, "   -in   : *.lammpstrj LAMMPS dump file\n");fflush(stderr);
        fprintf(stderr, "   -dr   : step size (Å) (optional, defaults to %.2lf)\n", dr);fflush(stderr);
        fprintf(stderr, "   -Rmax : maximum R (Å) for G(r) (optional, defaults to %.2lf)\n", Rmax);fflush(stderr);
        fprintf(stderr, "   -tidy : tidy output (1) or not (0) (optional, defaults to %d)\n", tidy);fflush(stderr);
        fprintf(stderr, "   -z    : project positions along z (1) or not (0) (optional, defaults to %d)\n", Zproj);
        fprintf(stderr, "   -max  : if max > 0, maximum number of images to read (optional, defaults to %d)\n", maxImages);
        fprintf(stderr, "   -skip : if skip > 0, skip this amount of images before reading (optional, defaults to %d)\n", skip);
        fprintf(stderr, "   -h    : show the help\n\n\n");
        fflush(stderr);
        return (-1);
    }
   
    Nbin = (int)((Rmax)/dr)+1;
    double invstep=1.0/dr;

    sscanf(read_line(input,4,ligne) ,"%d",&N);
    sscanf(read_line(input,6,ligne) ,"%lf %lf",&xlo,&xhi);
    sscanf(read_line(input,7,ligne) ,"%lf %lf",&ylo,&yhi);
    sscanf(read_line(input,8,ligne) ,"%lf %lf",&zlo,&zhi);

    Nimages = (int)(nb_lines(input) / (N + 9));
    for(i = 0; i < 100; i++) {
        typelist[i] = malloc((4) * sizeof(char));
    }
    
    /* Get atoms types */
    fp = fopen(input, "r");
    /* Header */
    i = 0; while(i<head){ fgets(ligne, sizeof(ligne), fp);i++; }
    fgets(ligne, sizeof(ligne), fp);
    sscanf(ligne, "%d %s %lf %lf %lf", &id, attyp, &x, &y, &z);
    strcpy(typelist[0], attyp);
    for (i = 1; i < N; i++){
        fgets(ligne, sizeof(ligne), fp);
        sscanf(ligne,"%d %s %lf %lf %lf",&id,attyp,&x,&y,&z);
        if (in(typelist, Ntype, attyp) == 0){
            strcpy(typelist[Ntype], attyp);
            Ntype++;
        }
    }
    fclose(fp);
    fprintf(stderr,"\033[1;31mTotal number of images:\033[0m %d\n", Nimages);fflush(stderr);
    if (maxImages > 0){
        Nimages = MIN(maxImages, Nimages-skip);
    } else {
        Nimages = Nimages-skip;
    }
    fprintf(stderr,"Reading from image %d up to image %d (total = %d)\n", skip+1, skip+Nimages, Nimages);fflush(stderr);

    count = calloc(Nbin, sizeof(double));
    integ = calloc(Nbin, sizeof(double));
    countij = calloc(Nbin, sizeof(double**));
    integij = calloc(Nbin, sizeof(double**));
    for (i = 0; i < Ntype; ++i) {
        countij[i] = calloc(Ntype, sizeof(double *));
        integij[i] = calloc(Ntype, sizeof(double *));
        for (j = 0; j < Ntype; ++j) {
            countij[i][j] = calloc(Nbin, sizeof(double));
            integij[i][j] = calloc(Nbin, sizeof(double));
        }
    }

    atom = calloc(N, sizeof(double *));
    type = calloc(N, sizeof(int));
    for(i = 0; i < N; i++){atom[i] = calloc(3, sizeof(double));}

    /* Let's read the trajectory */

    fp = fopen(input,"r");
    for(i=0;j<skip;j++){
        for(i=0;i<N+9;i++){
            fgets(ligne, sizeof(ligne), fp);
        }
    }
    n = 0;
    while(n<Nimages){
        /* Header */
        fgets(ligne, sizeof(ligne), fp); fgets(ligne, sizeof(ligne), fp);
        fgets(ligne, sizeof(ligne), fp); fgets(ligne, sizeof(ligne), fp);
        fgets(ligne, sizeof(ligne), fp);
        fgets(ligne, sizeof(ligne), fp); sscanf(ligne,"%lf %lf",&xlo,&xhi);
        fgets(ligne, sizeof(ligne), fp); sscanf(ligne,"%lf %lf",&ylo,&yhi);
        fgets(ligne, sizeof(ligne), fp); sscanf(ligne,"%lf %lf",&zlo,&zhi);
        fgets(ligne, sizeof(ligne), fp); sscanf(ligne,"%s %s %s %s %s",trash,trash,trash,trash,typex);
        Vtot = (xhi - xlo) * (yhi - ylo) * (zhi - zlo);
        Vmean += Vtot/Nimages;

        i = 0;
        Ni= calloc(Ntype, sizeof(int));
        // Reading image
        while(i<N){
            fgets(ligne, sizeof(ligne), fp);
            sscanf(ligne,"%d %s %lf %lf %lf",&id,attyp,&x,&y,&z);
            if (!strcmp(typex, "x")){
                atom[i][0] = x;
                atom[i][1] = y;
                atom[i][2] = z;
            }
            if (!strcmp(typex, "xs")){
                atom[i][0] = x*fabs(xhi-xlo);
                atom[i][1] = y*fabs(yhi-ylo);
                atom[i][2] = z*fabs(zhi-zlo);
            }
            type[i] = which(typelist, Ntype, attyp);
            Ni[type[i]]++;
            i++;
        }
        // If first image, print some information
        if (n==0){
            fprintf(stderr, "\n\033[1;31mAtoms types:\033[0m %d\n", Ntype);
            fflush(stderr);
            for (i = 0; i < Ntype; i++){
                fprintf(stderr, "   Type %d = %s (%d)\n", i + 1, typelist[i], Ni[i]);fflush(stderr);
            }
            fprintf(stderr,"\033[1;31mBox size:\n\033[0m   a = %.3lf\n", fabs(xhi-xlo));fflush(stderr);
            fprintf(stderr,"   b = %.3lf\n", fabs(yhi-ylo));fflush(stderr);
            fprintf(stderr,"   c = %.3lf\n", fabs(zhi-zlo));fflush(stderr);
            fprintf(stderr, "\n");fflush(stderr);
            if(Zproj==1){fprintf(stderr, "Computing \033[1;31mz-density\033[0m.\n");fflush(stderr);}
            if(Zproj==0){
                fprintf(stderr, "Computing \033[1;31mradial pair distribution\033[0m.\n");
                fflush(stderr);
            }
            if(!strcmp(typex, "x")){fprintf(stderr, "\033[1;31mAbsolute\033[0m coordinates detected.\n");fflush(stderr);}
            if(!strcmp(typex, "xs")){fprintf(stderr, "\033[1;31mRelative\033[0m coordinates detected.\n");fflush(stderr);}
            fprintf(stderr, "\n\n");fflush(stderr);
        }
        progressbar("Reading LAMMPS trajectory...", n, Nimages, 10);
        // Compute distances histograms
        for(i=0;i<N;i++){
            if (Zproj == 0){ // "regular" radial distance
                for(j=0;j<N;j++){
                    if( i==j ) continue;
                    x = Dist(xlo, xhi, ylo, yhi, zlo, zhi,
                            atom[i][0], atom[i][1], atom[i][2],
                            atom[j][0], atom[j][1], atom[j][2]);
                    l = (int) (x*invstep);
                    if( l >= Nbin ) continue;
                    count[l] += 1. / N / N * Vtot / Nimages;
                    ti = type[i];
                    tj = type[j];
                    countij[ti][tj][l] += 1. / Ni[ti] / Ni[tj] * Vtot / Nimages;
                }
            }
            if (Zproj == 1){ // consider only the z projection
                x = Dist(xlo, xhi, ylo, yhi, zlo, zhi,
                         0, 0, atom[i][2], 0, 0, 0);
                l = (int)(x * invstep);
                if (l >= Nbin) continue;
                count[l] += 1. / N;
                countij[type[i]][0][l] += 1. / Ni[type[i]];
            }
        }
        n++;
    }//end k
    fclose(fp);
    fprintf(stderr, "\n\n");fflush(stderr);

    double rlo,rup,nideal;

    // compute G(r) and its cumulative integral
    if(Zproj==0){
        for(l=1;l<Nbin;l++){
            rlo = l*dr;
            rup = (l+1)*dr;
            nideal = (rup * rup * rup - rlo * rlo * rlo) * 4.0 * PI / 3.0;
            count[l] = 1.0 * count[l] / nideal;
            for(i=0;i<Ntype;i++){
                for(j=i;j<Ntype;j++){
                    countij[i][j][l] = 1.0 * countij[i][j][l] / nideal;
                }
            }
        }
    }
    for(l=0;l<Nbin;l++){
        for(m=0;m<l;m++){
            r = dr * m + dr / 2.;
            integ[l] += count[m]*dr*4*PI*r*r*N/Vmean;
            for(i=0;i<Ntype;i++){
                for(j=i;j<Ntype;j++){
                    integij[i][j][l] += countij[i][j][m]*dr*4*PI*r*r*Ni[type[j]]/Vmean;
                }
            }
        }
    }
    // Output in wide csv format
    if(tidy==0) {
        // G(r)
        if(Zproj==0){
            fprintf(stdout, "%s,%s", "r", "G(r)");fflush(stdout);
            for(i=0;i<Ntype;i++){
                for(j=i;j<Ntype;j++){
                    sprintf(toprint, "%s{%s%s}(r)", "G_", typelist[i], typelist[j]);
                    fprintf(stdout, ",%s", toprint); fflush(stdout);
                }
            }
            for(i=0;i<Ntype;i++){
                for(j=i;j<Ntype;j++){
                    sprintf(toprint, "%s{%s%s}(r)", "N_", typelist[i], typelist[j]);
                    fprintf(stdout, ",%s", toprint); fflush(stdout);
                }
            }
        }
        // G(z)
        if(Zproj==1){
            fprintf(stdout, "%s,%s", "z", "D(z)");fflush(stdout);
            for(i=0;i<Ntype;i++){
                sprintf(toprint, "%s{%s}(z)", "D_", typelist[i]);
                fprintf(stdout, ",%s", toprint); fflush(stdout);
            }
        }
        fprintf(stdout, "\n");fflush(stdout);
        for(l=0;l<Nbin;l++){
            if(Zproj==0 && l==0){
                fprintf(stdout, "%.6lf,%.6lf", dr * l + dr / 2., 0.0); fflush(stdout);
                for(i=0;i<Ntype;i++){for(j=i;j<Ntype;j++){
                    fprintf(stdout, " %.6lf", 0.0); fflush(stdout);
                }}
                for(i=0;i<Ntype;i++){for(j=i;j<Ntype;j++){
                    fprintf(stdout, ",%.6lf", 0.0); fflush(stdout);
                }}
                fprintf(stdout, "\n");fflush(stdout);
            } else {
                fprintf(stdout, "%.6lf,%.6lf", dr * l + dr / 2., count[l]); fflush(stdout);
                if(Zproj==1){
                    for(i=0;i<Ntype;i++){
                        fprintf(stdout, ",%.6lf", countij[i][0][l]); fflush(stdout);
                    }
                }
                if(Zproj==0){
                    for(i=0;i<Ntype;i++){for(j=i;j<Ntype;j++){
                        fprintf(stdout, ",%.6lf", countij[i][j][l]); fflush(stdout);
                    }}
                    for(i=0;i<Ntype;i++){for(j=i;j<Ntype;j++){
                        fprintf(stdout, ",%.6lf", integij[i][j][l]); fflush(stdout);
                    }}
                }
                fprintf(stdout, "\n");fflush(stdout);
            }
        }
    }
    // Output in tidy csv format
    if(tidy==1){
        // G(r)
        if(Zproj==0){
            fprintf(stdout, "%s,%s,%s,%s\n", "r", "elements", "G(r)", "integral");
            fflush(stdout);
            for(l=0;l<Nbin;l++){
                if(l==0){
                    for(i=0;i<Ntype;i++){for(j=i;j<Ntype;j++){
                        sprintf(toprint, "%s%s", typelist[i], typelist[j]);
                        fprintf(stdout, "%lf,%s,%lf,%lf\n", 
                                dr * l + dr / 2., 
                                toprint,
                                0.0, 
                                0.0); 
                    }}
                } else {
                    fprintf(stdout, "%lf,%s,%lf,%lf\n",
                            dr * l + dr / 2.,
                            "total",
                            count[l],
                            integ[l]);
                    fflush(stdout);
                    for(i=0;i<Ntype;i++){
                        for(j=i;j<Ntype;j++){
                        sprintf(toprint, "%s%s", typelist[i], typelist[j]);
                        fprintf(stdout, "%lf,%s,%lf,%lf\n",
                                dr * l + dr / 2.,
                                toprint,
                                countij[i][j][l],
                                integij[i][j][l]);
                        fflush(stdout);
                        }
                    }
                }
            }
        }
        // G(z)
        if (Zproj == 1){
            fprintf(stdout, "%s,%s,%s\n", "z", "element", "density");
            fflush(stdout);
            for(l=0;l<Nbin;l++){
                fprintf(stdout, "%.6lf,%s,%.6lf\n",
                        dr * l + dr / 2.,
                        "total",
                        count[l]);
                fflush(stdout);
                for(i=0;i<Ntype;i++){
                    fprintf(stdout, "%.6lf,%s,%.6lf\n",
                            dr * l + dr / 2.,
                            typelist[i],
                            countij[i][0][l]);
                    fflush(stdout);
                }
            }
        }
    }
    return 0;
}


/* read_line reads a specific ligne in a file and stores it in (char) ligne */
char * read_line (char const * const s_filename, unsigned int i_line_num, char ligne[500]){
    FILE * p_file = NULL;
    unsigned int cpt = 1;
    /* ----- Ouverture du fichier ----- */
    p_file = fopen (s_filename, "r");
    
    if (!p_file){
        /* Erreur: impossible d'ouvrir le fichier. */
        return NULL;
    }
    /* ----- Lecture de la ligne du fichier ----- */
    while (fgets (ligne, 500, p_file)){
        if (cpt == i_line_num){
            /* La ligne a ete trouvee, on enleve le caractere
             de saut de ligne s'il est present. */
            char * p = strchr (ligne, '\n');
            if (p)
            {
                *p = 0;
            }
            /* On sort de la boucle. */
            break;
        }
        cpt++;
    }
    /* ----- Fermeture du fichier ----- */
    fclose (p_file);
    return (ligne);
}

/* This function gives the number of lines in an input file */
int nb_lines (char const * const s_filename){
    FILE *   p_file   = NULL;
    char     s_line     [BUFSIZ];
    int      nb_lines = 0;
    /* ----- Ouverture du fichier ----- */
    p_file = fopen (s_filename, "r");
    if (!p_file){
        /* Erreur: impossible d'ouvrir le fichier. */
        return -1;
    }
    /* ----- Lecture de la ligne du fichier ----- */
    while (fgets (s_line, BUFSIZ, p_file)){
        if (s_line [strlen (s_line) - 1] == '\n'){
            nb_lines++;
        }
    }
    
    if (ferror (p_file)){
        /* Erreur pendant la lecture du fichier. */
        fclose (p_file);
        return -1;
    }
    /* ----- Fermeture du fichier ----- */
    fclose (p_file);
    /* ----- */
    return nb_lines;
}


double Dist(double xlo,double xhi,double ylo,double yhi,double zlo,double zhi, 
            double point1, double point2, double point3, double pos1, double pos2, double pos3){
    double Dx,Dy,Dz;
    
        Dx = point1 - pos1;
        Dy = point2 - pos2;
        Dz = point3 - pos3;
        if (fabs(Dx) > (xhi-xlo) * 0.5) Dx = Dx - fabs(xhi-xlo)*SIGN(Dx);
        if (fabs(Dy) > (yhi-ylo) * 0.5) Dy = Dy - fabs(yhi-ylo)*SIGN(Dy);
        if (fabs(Dz) > (zhi-zlo) * 0.5) Dz = Dz - fabs(zhi-zlo)*SIGN(Dz);
        
    return sqrt( Dx*Dx + Dy*Dy + Dz*Dz );
}

void progressbar(const char *message, int i, int N, int Nchar)
{
    float percentage = (i + 1.) / N * 100;
    char *bar;
    char *ligne;
    ligne = calloc(500, sizeof(char));
    int dr = floor(100. / Nchar);
    bar = calloc(Nchar, sizeof(char));
    for (int j = 0; j < (int)(100. / dr); ++j)
        bar[j] = ' ';
    for (int j = 0; j < (int)(percentage / dr); ++j)
        bar[j] = '#';
    sprintf(ligne, "%-*s [%s] %.0f%%", Nchar, message, bar, percentage);
    fprintf(stderr, "\r%s", ligne);
    fflush(stderr);
}

int in(char **arr, int len, char *target)
{
    for (int i = 0; i < len; i++){
        if (strcmp(arr[i], target) == 0){ // if strings are equal
            return 1;
        }
    }
    return 0;
}

int which(char **arr, int len, char *target)
{
    int i;
    for (i = 0; i < len; i++){
        if (strcmp(arr[i], target) == 0){ // if strings are equal
            return (i);
        }
    }
    return 0;
}