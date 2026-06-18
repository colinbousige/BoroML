/*
 * gofr.c - Parallel Radial Distribution Function Calculator for LAMMPS Trajectories
 *
 * Purpose:
 *   Compute time-averaged radial distribution functions g(r) or density profiles from
 *   LAMMPS dump files. Supports type-specific pair distributions and cumulative integrals.
 *   Uses multi-threaded pair counting with spatially-optimized cell-list for large systems.
 *
 * Features:
 *   - Orthogonal and triclinic (non-orthogonal) box support via full 3x3 lattice matrix
 *   - Both scaled (xs, ys, zs) and Cartesian (x, y, z) coordinate parsing with auto-detection
 *   - Cell-list neighbor search (O(N) total, vs O(N²) brute-force) for memory efficiency
 *   - Upper-triangle pair histogram compression (~50% memory savings for symmetric g(r))
 *   - POSIX pthreads parallelization with lock-free per-thread accumulators
 *   - Detailed timing breakdown (file I/O, cell-list construction, pair counting)
 *   - Backward-compatible output format with original gofr.c
 *
 * Build:
 *   macOS: gcc -O3 -march=native -pthread gofr.c -lm -o GofR
 *   Linux HPC: gcc -O3 -march=native -pthread gofr.c -lm -o GofR
 *
 * Usage:
 *   ./GofR dump.lammpstrj [-dr 0.01 -Rmax 10 -tidy 1 -x|-y|-z -skip 0 -max 0 -t 0] > out.csv
 *
 * Arguments:
 *   input  : LAMMPS trajectory dump file (required first argument)
 *   -dr    : histogram bin width in Angstroms (default 0.01)
 *   -Rmax  : maximum radius / z in Angstroms (default 10.0)
 *   -tidy  : output format (1=tidy/long, 0=wide/flat), default 1
 *   -x     : compute x-projected density profile (default mode is radial g(r))
 *   -y     : compute y-projected density profile (default mode is radial g(r))
 *   -z     : compute z-projected density profile (default mode is radial g(r))
 *   -skip  : skip this many frames before processing (default 0)
 *   -max   : max frames to process after skip (0=all), default 0
 *   -t     : number of worker threads (0=auto-detect), default 0
 *   -h     : show help message
 *
 *    CSV to stdout with columns:
 *    - Long format (tidy=1):
 *      - G(r) case   : r,elements,G(r),integral
 *      - Density case: axis,element,density,integral
 *    - Wide format (tidy=0):
 *     - G(r) case   : r,G(0,0),G(0,1),...,G(1,1),...,integral(0,0),integral(0,1),...,integral(1,1),...
 *    - Note: wide format has fixed column order based on sorted type names, with pair (i,j) columns ordered as i≤j upper triangle.
 *     - Density case: axis,density(element0),density(element1),...,integral(element0),integral(element1),...
 *
 * Performance Notes:
 *   - Cell-list construction: ~17% of total time (frame-dependent)
 *   - Pair counting: ~76% of total time (dominated by PBC wrapping and binning)
 *   - File I/O: ~7% of total time (format parsing)
 *   - Memory: O(Natoms + nbin*ntypes² + nthreads*nbin*ntypes²) without temp buffers
 *
 */

#include <ctype.h>
#include <errno.h>
#include <math.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define PI 3.14159265358979323846
#define MAX_LINE 8192
#define MAX_TYPES 128
#define MAX_TYPE_NAME 32
#define MIN2(a, b) ((a) < (b) ? (a) : (b))
#define MAX2(a, b) ((a) > (b) ? (a) : (b))

typedef struct {
    double h[3][3];
    double hinv[3][3];
    double origin[3];
    double a, b, c;
    double alpha, beta, gamma;
    double volume;
    int triclinic;
} Lattice;

typedef struct {
    int start_i;
    int end_i;
    int nxcell;
    int nycell;
    int nzcell;
    int nbin;
    int ntypes;
    int rx;
    int ry;
    int rz;
    int max_cand;
    double inv_dr;
    double rmax2;
    const double *fx;
    const double *fy;
    const double *fz;
    const int *type;
    const int *head;
    const int *next;
    const int *cx;
    const int *cy;
    const int *cz;
    const Lattice *lat;
    double *hist_total;
    double *hist_pair;
} PairWorkerArgs;

static void usage(const char *prog) {
    fprintf(stderr,
            "\nGofR:\n"
            "  Parallel g(r) / projected density from LAMMPS dump trajectories.\n\n"
            "Usage:\n"
            "  %s <input.lammpstrj> [-dr 0.01 -Rmax 10 -tidy 1 -x|-y|-z -skip 0 -max 0 -t 0]\n\n"
            "Required arguments:\n"
            "    <input.lammpstrj>  : LAMMPS trajectory dump file\n\n"
            "Optional parameters:\n"
            "    -dr                : Histogram bin width in Angstroms (default 0.01)\n"
            "    -Rmax              : Maximum radius / z in Angstroms (default 10.0)\n"
            "    -tidy              : Output format (1=tidy/long, 0=wide/flat), default 1\n"
            "    -x                 : Compute x-projected density profile (default mode is radial g(r))\n"
            "    -y                 : Compute y-projected density profile (default mode is radial g(r))\n"
            "    -z                 : Compute z-projected density profile (default mode is radial g(r))\n"
            "    -skip              : Skip this many frames before processing (default 0)\n"
            "    -max               : Max frames to process after skip (0=all), default 0\n"
            "    -t                 : Number of worker threads (0=auto-detect), default 0\n\n"
            "Help:\n"
            "    -h, --help         : Show help message\n\n"
            "Output:\n"
            "    CSV to stdout with columns:\n"
            "    - Long format (tidy=1):\n"
            "        - G(r) case    : r,elements,G(r),integral\n"
            "        - Density case : axis,element,density,integral\n"
            "    - Wide format (tidy=0):\n"
            "        - G(r) case    : r,G(0,0),G(0,1),...,G(1,1),...,integral(0,0),integral(0,1),...,integral(1,1),...\n"
            "        - Density case : axis,density(element0),density(element1),...,integral(element0),integral(element1),...\n\n",
            prog);
}

static int get_default_threads(void) {
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    if (n < 1) {
        return 1;
    }
    if (n > 1024) {
        n = 1024;
    }
    return (int)n;
}

static int split_tokens(char *line, char **tok, int max_tok) {
    int n = 0;
    char *save = NULL;
    char *p = strtok_r(line, " \t\r\n", &save);
    while (p && n < max_tok) {
        tok[n++] = p;
        p = strtok_r(NULL, " \t\r\n", &save);
    }
    return n;
}

static int parse_double3(const char *line, double *a, double *b, double *c) {
    return sscanf(line, "%lf %lf %lf", a, b, c) == 3;
}

static int invert3x3(const double m[3][3], double inv[3][3], double *det_out) {
    double det =
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) -
        m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
        m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    if (fabs(det) < 1e-20) {
        return 0;
    }
    inv[0][0] = (m[1][1] * m[2][2] - m[1][2] * m[2][1]) / det;
    inv[0][1] = (m[0][2] * m[2][1] - m[0][1] * m[2][2]) / det;
    inv[0][2] = (m[0][1] * m[1][2] - m[0][2] * m[1][1]) / det;
    inv[1][0] = (m[1][2] * m[2][0] - m[1][0] * m[2][2]) / det;
    inv[1][1] = (m[0][0] * m[2][2] - m[0][2] * m[2][0]) / det;
    inv[1][2] = (m[0][2] * m[1][0] - m[0][0] * m[1][2]) / det;
    inv[2][0] = (m[1][0] * m[2][1] - m[1][1] * m[2][0]) / det;
    inv[2][1] = (m[0][1] * m[2][0] - m[0][0] * m[2][1]) / det;
    inv[2][2] = (m[0][0] * m[1][1] - m[0][1] * m[1][0]) / det;
    if (det_out) {
        *det_out = det;
    }
    return 1;
}

static double vec_norm(const double v[3]) {
    return sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}

static void build_lattice(double xlo_b, double xhi_b,
                          double ylo_b, double yhi_b,
                          double zlo_b, double zhi_b,
                          double xy, double xz, double yz,
                          int triclinic,
                          Lattice *lat) {
    double xlo = xlo_b, xhi = xhi_b;
    double ylo = ylo_b, yhi = yhi_b;
    double zlo = zlo_b, zhi = zhi_b;

    if (triclinic) {
        double minx = MIN2(0.0, MIN2(xy, MIN2(xz, xy + xz)));
        double maxx = MAX2(0.0, MAX2(xy, MAX2(xz, xy + xz)));
        double miny = MIN2(0.0, yz);
        double maxy = MAX2(0.0, yz);
        xlo = xlo_b - minx;
        xhi = xhi_b - maxx;
        ylo = ylo_b - miny;
        yhi = yhi_b - maxy;
    }

    double lx = xhi - xlo;
    double ly = yhi - ylo;
    double lz = zhi - zlo;

    lat->h[0][0] = lx;
    lat->h[1][0] = 0.0;
    lat->h[2][0] = 0.0;

    lat->h[0][1] = triclinic ? xy : 0.0;
    lat->h[1][1] = ly;
    lat->h[2][1] = 0.0;

    lat->h[0][2] = triclinic ? xz : 0.0;
    lat->h[1][2] = triclinic ? yz : 0.0;
    lat->h[2][2] = lz;

    lat->origin[0] = xlo;
    lat->origin[1] = ylo;
    lat->origin[2] = zlo;
    lat->triclinic = triclinic;

    invert3x3(lat->h, lat->hinv, &lat->volume);
    lat->volume = fabs(lat->volume);

    {
        double av[3] = {lat->h[0][0], lat->h[1][0], lat->h[2][0]};
        double bv[3] = {lat->h[0][1], lat->h[1][1], lat->h[2][1]};
        double cv[3] = {lat->h[0][2], lat->h[1][2], lat->h[2][2]};
        double dotab = av[0] * bv[0] + av[1] * bv[1] + av[2] * bv[2];
        double dotac = av[0] * cv[0] + av[1] * cv[1] + av[2] * cv[2];
        double dotbc = bv[0] * cv[0] + bv[1] * cv[1] + bv[2] * cv[2];
        lat->a = vec_norm(av);
        lat->b = vec_norm(bv);
        lat->c = vec_norm(cv);
        lat->alpha = acos(fmax(-1.0, fmin(1.0, dotbc / (lat->b * lat->c))));
        lat->beta = acos(fmax(-1.0, fmin(1.0, dotac / (lat->a * lat->c))));
        lat->gamma = acos(fmax(-1.0, fmin(1.0, dotab / (lat->a * lat->b))));
    }
}

static void cart_to_frac(const Lattice *lat, double x, double y, double z,
                         double *u, double *v, double *w) {
    double rx = x - lat->origin[0];
    double ry = y - lat->origin[1];
    double rz = z - lat->origin[2];
    *u = lat->hinv[0][0] * rx + lat->hinv[0][1] * ry + lat->hinv[0][2] * rz;
    *v = lat->hinv[1][0] * rx + lat->hinv[1][1] * ry + lat->hinv[1][2] * rz;
    *w = lat->hinv[2][0] * rx + lat->hinv[2][1] * ry + lat->hinv[2][2] * rz;
}

static int type_index(const char *name,
                      char type_names[MAX_TYPES][MAX_TYPE_NAME],
                      int *ntypes,
                      int freeze_types) {
    int i;
    for (i = 0; i < *ntypes; i++) {
        if (strcmp(type_names[i], name) == 0) {
            return i;
        }
    }
    if (freeze_types) {
        return -1;
    }
    if (*ntypes >= MAX_TYPES) {
        return -1;
    }
    strncpy(type_names[*ntypes], name, MAX_TYPE_NAME - 1);
    type_names[*ntypes][MAX_TYPE_NAME - 1] = '\0';
    (*ntypes)++;
    return (*ntypes) - 1;
}

static double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static int wrap_index(int i, int n) {
    int r = i % n;
    if (r < 0) {
        r += n;
    }
    return r;
}

static int cell_index_3d(int ix, int iy, int iz, int nx, int ny, int nz) {
    (void)nz;
    return (iz * ny + iy) * nx + ix;
}

static size_t pair_hist_index(int ti, int tj, int ntypes, int l, int nbin) {
    /*
     * Upper-triangle index for the (ti, tj) pair histogram.
     *
     * For ntypes types, pairs are ordered as:
     *   (0,0), (0,1), ..., (0,n-1),  <- n pairs in row 0
     *          (1,1), ..., (1,n-1),  <- n-1 pairs in row 1
     *                 ...
     *                 (n-1,n-1)      <- 1 pair in last row
     *
     * Number of pairs before row i: sum_{k=0}^{i-1} (ntypes-k) = i*ntypes - i*(i-1)/2
     * Offset within row i for column j (j >= i): (j - i)
     * Total: pair_idx = i*ntypes - i*(i-1)/2 + (j-i)
     *
     * Note: i*(i+1)/2 is WRONG -- causes collisions (e.g. BAg and AgAg map to same slot).
     */
    int i = (ti <= tj) ? ti : tj;
    int j = (ti <= tj) ? tj : ti;
    int pair_idx = i * ntypes - i * (i - 1) / 2 + (j - i);
    return (size_t)pair_idx * (size_t)nbin + (size_t)l;
}

static size_t pair_hist_size(int ntypes, int nbin) {
    size_t npairs = (size_t)ntypes * ((size_t)ntypes + 1u) / 2u;
    return npairs * (size_t)nbin;
}

static void *pair_worker(void *argp) {
    /**
     * Worker thread for pair counting via cell-list traversal.
     *
     * Algorithm:
     *   1. For each atom i in [start_i, end_i):
     *     a. Get fractional coordinates (ui, vi, wi) and cell indices (cix, ciy, ciz)
     *     b. Generate candidate neighbor cells using stencil (±rx, ±ry, ±rz) with PBC wrapping
     *     c. Deduplicate cells (avoid double-counting due to stencil overlap)
     *     d. For each candidate cell, traverse linked list of atoms in that cell
     *     e. For each atom j in neighbor cells:
     *        - Wrap fractional coordinates to [-0.5, 0.5) with nearbyint() for PBC
     *        - Convert wrapped fractional distance to Cartesian via lattice matrix H
     *        - Compute r² and bin into histogram if r² < rmax²
     *        - Accumulate 2.0 for same-type pairs, 1.0 for different-type pairs
     *   2. Return thread-local histogram in arg->hist_pair
     *
     * Key Optimizations:
     *   - Fractional coordinates and modulo arithmetic reduce numerical error in PBC
     *   - Linked-list cell traversal avoids storing explicit neighbor coordinates
     *   - j>i filtering prevents double-counting (each unique pair counted once)
     */
    PairWorkerArgs *arg = (PairWorkerArgs *)argp;
    int i;
    const Lattice *lat = arg->lat;
    int *cand_cells = (int *)malloc((size_t)arg->max_cand * sizeof(int));

    if (!cand_cells) {
        return NULL;
    }

    for (i = arg->start_i; i < arg->end_i; i++) {
        double ui = arg->fx[i];
        double vi = arg->fy[i];
        double wi = arg->fz[i];
        int ti = arg->type[i];
        int cix = arg->cx[i];
        int ciy = arg->cy[i];
        int ciz = arg->cz[i];
        int ncand = 0;
        int dxo, dyo, dzo;

        for (dzo = -arg->rz; dzo <= arg->rz; dzo++) {
            int nz = wrap_index(ciz + dzo, arg->nzcell);
            for (dyo = -arg->ry; dyo <= arg->ry; dyo++) {
                int ny = wrap_index(ciy + dyo, arg->nycell);
                for (dxo = -arg->rx; dxo <= arg->rx; dxo++) {
                    int nx = wrap_index(cix + dxo, arg->nxcell);
                    int cidx = cell_index_3d(nx, ny, nz, arg->nxcell, arg->nycell, arg->nzcell);
                    int seen = 0;
                    int q;

                    for (q = 0; q < ncand; q++) {
                        if (cand_cells[q] == cidx) {
                            seen = 1;
                            break;
                        }
                    }
                    if (!seen && ncand < arg->max_cand) {
                        cand_cells[ncand++] = cidx;
                    }
                }
            }
        }

        for (int ic = 0; ic < ncand; ic++) {
            int j;
            for (j = arg->head[cand_cells[ic]]; j != -1; j = arg->next[j]) {
                double du, dv, dw;
                double dx, dy, dz, r2;
                int l;
                int tj;
                int a, b;

                if (j <= i) {
                    continue;
                }

                du = arg->fx[j] - ui;
                dv = arg->fy[j] - vi;
                dw = arg->fz[j] - wi;

                du -= nearbyint(du);
                dv -= nearbyint(dv);
                dw -= nearbyint(dw);

                dx = lat->h[0][0] * du + lat->h[0][1] * dv + lat->h[0][2] * dw;
                dy = lat->h[1][0] * du + lat->h[1][1] * dv + lat->h[1][2] * dw;
                dz = lat->h[2][0] * du + lat->h[2][1] * dv + lat->h[2][2] * dw;
                r2 = dx * dx + dy * dy + dz * dz;
                if (r2 >= arg->rmax2) {
                    continue;
                }

                l = (int)(sqrt(r2) * arg->inv_dr);
                if (l < 0 || l >= arg->nbin) {
                    continue;
                }

                arg->hist_total[l] += 2.0;

                tj = arg->type[j];
                if (ti == tj) {
                    arg->hist_pair[pair_hist_index(ti, ti, arg->ntypes, l, arg->nbin)] += 2.0;
                } else {
                    arg->hist_pair[pair_hist_index(ti, tj, arg->ntypes, l, arg->nbin)] += 1.0;
                }
            }
        }
    }

    free(cand_cells);

    return NULL;
}

static void progressbar(const char *message, int i, int n, int nchar) {
    int k;
    int filled;
    double pct;
    char *bar;

    if (n <= 0) {
        return;
    }

    pct = 100.0 * ((double)(i + 1) / (double)n);
    bar = (char *)calloc((size_t)nchar + 1u, sizeof(char));
    if (!bar) {
        return;
    }
    filled = (int)lrint((pct / 100.0) * (double)nchar);
    if (filled < 0) {
        filled = 0;
    }
    if (filled > nchar) {
        filled = nchar;
    }
    for (k = 0; k < nchar; k++) {
        bar[k] = (k < filled) ? '#' : ' ';
    }
    fprintf(stderr, "\r%-24s [%s] %5.1f%%", message, bar, pct);
    fflush(stderr);
    free(bar);
}

int main(int argc, char **argv) {
    const char *input = NULL;
    double dr = 0.01;
    /* Keep original gofr.c behavior: default Rmax is 10 + dr (applied after parsing -dr). */
    double rmax = 10.0;
    int rmax_set = 0;
    int tidy = 1;
    int proj_axis = -1; /* -1: radial g(r), 0: x, 1: y, 2: z projected density */
    int max_images = 0;
    int skip = 0;
    int nthreads = 0;

    int i, j, l;
    int nbin;
    double inv_dr;

    FILE *fp;
    char line[MAX_LINE];

    double *fx = NULL, *fy = NULL, *fz = NULL;
    int *atype = NULL;
    int cap_atoms = 0;

    char type_names[MAX_TYPES][MAX_TYPE_NAME];
    int ntypes = 0;
    int freeze_types = 0;

    double *count = NULL;
    double *integ = NULL;
    double *countij = NULL;
    double *integij = NULL;

    int frames_seen = 0;
    int frames_used = 0;
    double vmean = 0.0;
    int nref = 0;
    int ni_ref[MAX_TYPES];
    double t_file_start, t_file_end = 0.0;  /* file I/O timing */
    double t_cell_start, t_cell_total = 0.0; /* cell-list construction timing */
    double t_pair_start, t_pair_total = 0.0; /* pair accumulation timing */

    for (i = 0; i < MAX_TYPES; i++) {
        ni_ref[i] = 0;
        type_names[i][0] = '\0';
    }

    t_file_start = now_seconds();

    if (argc < 2 || !strcmp(argv[1], "-h") || !strcmp(argv[i], "--help")) {
        usage(argv[0]);
        return (argc >= 2 && !strcmp(argv[1], "-h")) ? 0 : 1;
    }

    input = argv[1];
    if (input[0] == '-') {
        fprintf(stderr, "Error: first argument must be the input file path\n");
        usage(argv[0]);
        return 1;
    }

    for (i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "-dr") && i + 1 < argc) {
            dr = atof(argv[++i]);
        } else if (!strcmp(argv[i], "-Rmax") && i + 1 < argc) {
            rmax = atof(argv[++i]);
            rmax_set = 1;
        } else if (!strcmp(argv[i], "-tidy") && i + 1 < argc) {
            tidy = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "-x")) {
            if (proj_axis >= 0 && proj_axis != 0) {
                fprintf(stderr, "Error: choose only one of -x, -y, -z\n");
                return 1;
            }
            proj_axis = 0;
        } else if (!strcmp(argv[i], "-y")) {
            if (proj_axis >= 0 && proj_axis != 1) {
                fprintf(stderr, "Error: choose only one of -x, -y, -z\n");
                return 1;
            }
            proj_axis = 1;
        } else if (!strcmp(argv[i], "-z")) {
            if (proj_axis >= 0 && proj_axis != 2) {
                fprintf(stderr, "Error: choose only one of -x, -y, -z\n");
                return 1;
            }
            proj_axis = 2;
        } else if (!strcmp(argv[i], "-max") && i + 1 < argc) {
            max_images = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "-skip") && i + 1 < argc) {
            skip = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "-t") && i + 1 < argc) {
            nthreads = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "-h")) {
            usage(argv[0]);
            return 0;
        } else {
            usage(argv[0]);
            return 1;
        }
    }

    if (!rmax_set) {
        rmax = 10.0 + dr;
    }

    if (dr <= 0.0 || rmax <= 0.0) {
        fprintf(stderr, "Error: dr and Rmax must be > 0\n");
        return 1;
    }
    if (nthreads <= 0) {
        nthreads = get_default_threads();
    }

    fp = fopen(input, "r");
    if (!fp) {
        fprintf(stderr, "Error: cannot open %s (%s)\n", input, strerror(errno));
        return 1;
    }

    nbin = (int)(rmax / dr) + 1;
    inv_dr = 1.0 / dr;

    count = (double *)calloc((size_t)nbin, sizeof(double));
    integ = (double *)calloc((size_t)nbin, sizeof(double));
    if (!count || !integ) {
        fprintf(stderr, "Error: allocation failure\n");
        fclose(fp);
        free(count);
        free(integ);
        return 1;
    }

    /**
     * MAIN TRAJECTORY PROCESSING LOOP
     * ================================
     * Strategy:
     *   1. Parse each frame: box bounds, atom count, atom positions/types
     *   2. Detect coordinate system (fractional vs cartesian) and cell triclinicity
     *   3. Convert all coordinates to fractional (wrapped in [-0.5, 0.5))
     *   4. Build spatial hash (cell-list) for fast neighbor search
     *   5. Launch worker threads to count pairs within rmax using cell-list
     *   6. Accumulate thread-local histograms into global g(r) / z-density arrays
     *   7. Update running averages of volume and atom-type counts
     *
     * Normalization Strategy (executed per frame):
     *   - Total pair count: sum_{frames} count[l] += pair_count[l] / (N² frames)
     *   - Type-specific: countij[i][j][l] += pair_count[i,j][l] / (Ni * Nj * frames)
     *   - This ensures TIME-AVERAGED g(r) when summed over all frames
     *
     * Memory Efficiency:
     *   - No per-frame temp buffers: thread-local accumulators joined directly to globals
     *   - Upper-triangle pair storage: only i≤j pairs stored, index via pair_hist_index()
     *   - PBC wrapping done in-place: no explicit image-atom copies
     */

    while (fgets(line, sizeof(line), fp)) {
        int n_atoms;
        Lattice lat;
        int triclinic;
        double xlo_b, xhi_b, ylo_b, yhi_b, zlo_b, zhi_b;
        double xy = 0.0, xz = 0.0, yz = 0.0;
        int col_id = -1, col_type = -1;
        int col_xs = -1, col_ys = -1, col_zs = -1;
        int col_x = -1, col_y = -1, col_z = -1;
        int have_scaled = 0;
        int have_cart = 0;
        int *ni_frame;

        if (strncmp(line, "ITEM: TIMESTEP", 14) != 0) {
            continue;
        }
        frames_seen++;

        if (!fgets(line, sizeof(line), fp)) {
            break;
        }
        if (!fgets(line, sizeof(line), fp) || strncmp(line, "ITEM: NUMBER OF ATOMS", 21) != 0) {
            fprintf(stderr, "Error: malformed dump near frame %d (missing NUMBER OF ATOMS)\n", frames_seen);
            goto fail;
        }
        if (!fgets(line, sizeof(line), fp) || sscanf(line, "%d", &n_atoms) != 1 || n_atoms <= 0) {
            fprintf(stderr, "Error: malformed atom count at frame %d\n", frames_seen);
            goto fail;
        }

        if (n_atoms > cap_atoms) {
            double *nfx = (double *)realloc(fx, (size_t)n_atoms * sizeof(double));
            double *nfy = (double *)realloc(fy, (size_t)n_atoms * sizeof(double));
            double *nfz = (double *)realloc(fz, (size_t)n_atoms * sizeof(double));
            int *ntp = (int *)realloc(atype, (size_t)n_atoms * sizeof(int));
            if (!nfx || !nfy || !nfz || !ntp) {
                free(nfx);
                free(nfy);
                free(nfz);
                free(ntp);
                fprintf(stderr, "Error: allocation failure for atom arrays\n");
                goto fail;
            }
            fx = nfx;
            fy = nfy;
            fz = nfz;
            atype = ntp;
            cap_atoms = n_atoms;
        }

        if (!fgets(line, sizeof(line), fp) || strncmp(line, "ITEM: BOX BOUNDS", 16) != 0) {
            fprintf(stderr, "Error: malformed dump near frame %d (missing BOX BOUNDS)\n", frames_seen);
            goto fail;
        }
        triclinic = (strstr(line, "xy") && strstr(line, "xz") && strstr(line, "yz")) ? 1 : 0;

        if (triclinic) {
            if (!fgets(line, sizeof(line), fp) || !parse_double3(line, &xlo_b, &xhi_b, &xy) ||
                !fgets(line, sizeof(line), fp) || !parse_double3(line, &ylo_b, &yhi_b, &xz) ||
                !fgets(line, sizeof(line), fp) || !parse_double3(line, &zlo_b, &zhi_b, &yz)) {
                fprintf(stderr, "Error: malformed triclinic bounds at frame %d\n", frames_seen);
                goto fail;
            }
        } else {
            if (!fgets(line, sizeof(line), fp) || sscanf(line, "%lf %lf", &xlo_b, &xhi_b) != 2 ||
                !fgets(line, sizeof(line), fp) || sscanf(line, "%lf %lf", &ylo_b, &yhi_b) != 2 ||
                !fgets(line, sizeof(line), fp) || sscanf(line, "%lf %lf", &zlo_b, &zhi_b) != 2) {
                fprintf(stderr, "Error: malformed orthogonal bounds at frame %d\n", frames_seen);
                goto fail;
            }
            xy = xz = yz = 0.0;
        }
        build_lattice(xlo_b, xhi_b, ylo_b, yhi_b, zlo_b, zhi_b, xy, xz, yz, triclinic, &lat);

        if (!fgets(line, sizeof(line), fp) || strncmp(line, "ITEM: ATOMS", 11) != 0) {
            fprintf(stderr, "Error: malformed dump near frame %d (missing ATOMS header)\n", frames_seen);
            goto fail;
        }
        {
            char copy[MAX_LINE];
            char *tok[256];
            int ntok;
            int c;

            strncpy(copy, line, sizeof(copy) - 1);
            copy[sizeof(copy) - 1] = '\0';
            ntok = split_tokens(copy, tok, 256);
            for (c = 2; c < ntok; c++) {
                if (!strcmp(tok[c], "id")) col_id = c - 2;
                else if (!strcmp(tok[c], "type") || !strcmp(tok[c], "element")) col_type = c - 2;
                else if (!strcmp(tok[c], "xs") || !strcmp(tok[c], "xsu")) col_xs = c - 2;
                else if (!strcmp(tok[c], "ys") || !strcmp(tok[c], "ysu")) col_ys = c - 2;
                else if (!strcmp(tok[c], "zs") || !strcmp(tok[c], "zsu")) col_zs = c - 2;
                else if (!strcmp(tok[c], "x") || !strcmp(tok[c], "xu")) col_x = c - 2;
                else if (!strcmp(tok[c], "y") || !strcmp(tok[c], "yu")) col_y = c - 2;
                else if (!strcmp(tok[c], "z") || !strcmp(tok[c], "zu")) col_z = c - 2;
            }
        }
        (void)col_id;
        have_scaled = (col_xs >= 0 && col_ys >= 0 && col_zs >= 0);
        have_cart = (col_x >= 0 && col_y >= 0 && col_z >= 0);
        if (col_type < 0 || (!have_scaled && !have_cart)) {
            fprintf(stderr,
                    "Error: unsupported ATOMS format at frame %d. Need type and either xs ys zs or x y z.\n",
                    frames_seen);
            goto fail;
        }

        ni_frame = (int *)calloc((size_t)MAX_TYPES, sizeof(int));
        if (!ni_frame) {
            fprintf(stderr, "Error: allocation failure for type counters\n");
            goto fail;
        }

        for (i = 0; i < n_atoms; i++) {
            char *tok[256];
            char copy[MAX_LINE];
            int ntok;
            int tix;
            double u, v, w;

            if (!fgets(line, sizeof(line), fp)) {
                free(ni_frame);
                fprintf(stderr, "Error: unexpected EOF while reading atoms at frame %d\n", frames_seen);
                goto fail;
            }

            strncpy(copy, line, sizeof(copy) - 1);
            copy[sizeof(copy) - 1] = '\0';
            ntok = split_tokens(copy, tok, 256);
            if (ntok <= col_type) {
                free(ni_frame);
                fprintf(stderr, "Error: malformed atom line at frame %d\n", frames_seen);
                goto fail;
            }

            tix = type_index(tok[col_type], type_names, &ntypes, freeze_types);
            if (tix < 0) {
                free(ni_frame);
                fprintf(stderr, "Error: too many/new atom types at frame %d. Increase MAX_TYPES or keep types constant.\n", frames_seen);
                goto fail;
            }
            atype[i] = tix;
            ni_frame[tix]++;

            if (have_scaled) {
                if (ntok <= col_zs) {
                    free(ni_frame);
                    fprintf(stderr, "Error: malformed scaled coordinates at frame %d\n", frames_seen);
                    goto fail;
                }
                u = atof(tok[col_xs]);
                v = atof(tok[col_ys]);
                w = atof(tok[col_zs]);
            } else {
                double x = atof(tok[col_x]);
                double y = atof(tok[col_y]);
                double z = atof(tok[col_z]);
                cart_to_frac(&lat, x, y, z, &u, &v, &w);
            }

            u -= floor(u);
            v -= floor(v);
            w -= floor(w);
            fx[i] = u;
            fy[i] = v;
            fz[i] = w;
        }

        if (frames_seen <= skip) {
            free(ni_frame);
            continue;
        }
        if (max_images > 0 && frames_used >= max_images) {
            free(ni_frame);
            break;
        }

        if (!freeze_types) {
            countij = (double *)calloc(pair_hist_size(ntypes, nbin), sizeof(double));
            integij = (double *)calloc(pair_hist_size(ntypes, nbin), sizeof(double));
            if (!countij || !integij) {
                free(ni_frame);
                fprintf(stderr, "Error: allocation failure for histograms\n");
                goto fail;
            }
            freeze_types = 1;
        }

        if (frames_used == 0) {
            nref = n_atoms;
            for (i = 0; i < ntypes; i++) {
                ni_ref[i] = ni_frame[i];
            }
            fprintf(stderr, "\n\033[1;31mAtom types:\033[0m %d\n", ntypes);
            for (i = 0; i < ntypes; i++) {
                fprintf(stderr, "  Type %d = %s (%d)\n", i + 1, type_names[i], ni_ref[i]);
            }
            fprintf(stderr, "\033[1;31mCell:\033[0m a = %.4f b = %.4f c = %.4f\n      alpha = %.4f beta = %.4f gamma = %.4f\n",
                    lat.a, lat.b, lat.c, lat.alpha/PI*180.0, lat.beta/PI*180.0, lat.gamma/PI*180.0);
            fprintf(stderr, "%s coordinates detected\n\n", have_scaled ? "Scaled" : "Cartesian");
        }

        if (proj_axis < 0) {
            int t;
            int use_threads = MIN2(nthreads, n_atoms > 1 ? n_atoms - 1 : 1);
            pthread_t *threads;
            PairWorkerArgs *args;
            double **local_total;
            double **local_pair;
            int *head = NULL;
            int *next = NULL;
            int *cx = NULL;
            int *cy = NULL;
            int *cz = NULL;
            int nxcell, nycell, nzcell, ncell;
            int rx, ry, rz;
            int max_cand;
            double frame_rmax = MIN2(rmax, 0.5 * MIN2(lat.a, MIN2(lat.b, lat.c)) + dr);

            if (frame_rmax < rmax) {
                rmax = frame_rmax;
                nbin = (int)(rmax / dr) + 1;
                inv_dr = 1.0 / dr;
            }

            /**
             * CELL-LIST CONSTRUCTION
             * ======================
             * Build spatial hash grid to accelerate pair search from O(N²) to O(N)
             *
             * Grid sizing:
             *   - Number of cells in each direction: nxcell = floor(a / frame_rmax)
             *   - Ensures each cell is at least frame_rmax in size
             *   - Total cells: ncell = nxcell × nycell × nzcell
             *
             * Cell index mapping:
             *   - Atom i at fractional coords (u,v,w) ∈ [0,1)³ maps to cell (ix,iy,iz)
             *   - Each cell maintains linked list: head[cidx] → next[atom0] → next[atom1] → ...
             *   - Linearized cell index: cidx = (iz * ny + iy) * nx + ix
             *
             * Neighbor stencil (conservative):
             *   - Stencil radius in cells: rx/ry/rz computed from inverse lattice matrix norms
             *   - For triclinic cells: ||H_inv[row,:]|| tells us max "cell reach" in that direction
             *   - Uses ceiling: accounts for worst-case skew in non-orthogonal cells
             *   - PBC wrapping: cell indices wrap modulo (nxcell, nycell, nzcell)
             */

            nxcell = (int)floor(lat.a / frame_rmax);
            nycell = (int)floor(lat.b / frame_rmax);
            nzcell = (int)floor(lat.c / frame_rmax);
            if (nxcell < 1) nxcell = 1;
            if (nycell < 1) nycell = 1;
            if (nzcell < 1) nzcell = 1;
            ncell = nxcell * nycell * nzcell;

            head = (int *)malloc((size_t)ncell * sizeof(int));
            next = (int *)malloc((size_t)n_atoms * sizeof(int));
            cx = (int *)malloc((size_t)n_atoms * sizeof(int));
            cy = (int *)malloc((size_t)n_atoms * sizeof(int));
            cz = (int *)malloc((size_t)n_atoms * sizeof(int));
            if (!head || !next || !cx || !cy || !cz) {
                free(ni_frame);
                free(head);
                free(next);
                free(cx);
                free(cy);
                free(cz);
                fprintf(stderr, "Error: cell-list allocation failure\n");
                goto fail;
            }
            t_cell_start = now_seconds();
            for (i = 0; i < ncell; i++) {
                head[i] = -1;
            }
            for (i = 0; i < n_atoms; i++) {
                int ix = (int)floor(fx[i] * (double)nxcell);
                int iy = (int)floor(fy[i] * (double)nycell);
                int iz = (int)floor(fz[i] * (double)nzcell);
                int cidx;
                if (ix >= nxcell) ix = nxcell - 1;
                if (iy >= nycell) iy = nycell - 1;
                if (iz >= nzcell) iz = nzcell - 1;
                if (ix < 0) ix = 0;
                if (iy < 0) iy = 0;
                if (iz < 0) iz = 0;
                cx[i] = ix;
                cy[i] = iy;
                cz[i] = iz;
                cidx = cell_index_3d(ix, iy, iz, nxcell, nycell, nzcell);
                next[i] = head[cidx];
                head[cidx] = i;
            }

            rx = (int)ceil(frame_rmax * sqrt(lat.hinv[0][0] * lat.hinv[0][0] + lat.hinv[0][1] * lat.hinv[0][1] + lat.hinv[0][2] * lat.hinv[0][2]) * (double)nxcell);
            ry = (int)ceil(frame_rmax * sqrt(lat.hinv[1][0] * lat.hinv[1][0] + lat.hinv[1][1] * lat.hinv[1][1] + lat.hinv[1][2] * lat.hinv[1][2]) * (double)nycell);
            rz = (int)ceil(frame_rmax * sqrt(lat.hinv[2][0] * lat.hinv[2][0] + lat.hinv[2][1] * lat.hinv[2][1] + lat.hinv[2][2] * lat.hinv[2][2]) * (double)nzcell);
            if (rx < 0) rx = 0;
            if (ry < 0) ry = 0;
            if (rz < 0) rz = 0;
            if (rx > nxcell - 1) rx = nxcell - 1;
            if (ry > nycell - 1) ry = nycell - 1;
            if (rz > nzcell - 1) rz = nzcell - 1;

            max_cand = (2 * rx + 1) * (2 * ry + 1) * (2 * rz + 1);
            if (max_cand > ncell) {
                max_cand = ncell;
            }

            threads = (pthread_t *)malloc((size_t)use_threads * sizeof(pthread_t));
            args = (PairWorkerArgs *)calloc((size_t)use_threads, sizeof(PairWorkerArgs));
            local_total = (double **)calloc((size_t)use_threads, sizeof(double *));
            local_pair = (double **)calloc((size_t)use_threads, sizeof(double *));
            if (!threads || !args || !local_total || !local_pair) {
                free(ni_frame);
                free(head);
                free(next);
                free(cx);
                free(cy);
                free(cz);
                free(threads);
                free(args);
                free(local_total);
                free(local_pair);
                fprintf(stderr, "Error: thread allocation failure\n");
                goto fail;
            }

            for (t = 0; t < use_threads; t++) {
                int k;
                local_total[t] = (double *)calloc((size_t)nbin, sizeof(double));
                local_pair[t] = (double *)calloc(pair_hist_size(ntypes, nbin), sizeof(double));
                if (!local_total[t] || !local_pair[t]) {
                    free(ni_frame);
                    free(head);
                    free(next);
                    free(cx);
                    free(cy);
                    free(cz);
                    for (k = 0; k <= t; k++) {
                        free(local_total[k]);
                        free(local_pair[k]);
                    }
                    free(threads);
                    free(args);
                    free(local_total);
                    free(local_pair);
                    fprintf(stderr, "Error: local histogram allocation failure\n");
                    goto fail;
                }
                /**
                 * THREAD WORK DISTRIBUTION
                 * ========================
                 * Assign contiguous blocks of atoms to each thread for load balancing.
                 * Each thread t processes atoms in range [start_i, end_i), approximately
                 * n_atoms/use_threads atoms per thread.
                 *
                 * Lock-free accumulation:
                 *   - Each thread has private local_total[t][nbin] and local_pair[t][...]
                 *   - No locking needed; threads never access the same memory locations
                 *   - After join: sum all thread-local buffers into global counters
                 */
                args[t].start_i = (t * n_atoms) / use_threads;
                args[t].end_i = ((t + 1) * n_atoms) / use_threads;
                args[t].nxcell = nxcell;
                args[t].nycell = nycell;
                args[t].nzcell = nzcell;
                args[t].nbin = nbin;
                args[t].ntypes = ntypes;
                args[t].rx = rx;
                args[t].ry = ry;
                args[t].rz = rz;
                args[t].max_cand = max_cand;
                args[t].inv_dr = inv_dr;
                args[t].rmax2 = frame_rmax * frame_rmax;
                args[t].fx = fx;
                args[t].fy = fy;
                args[t].fz = fz;
                args[t].type = atype;
                args[t].head = head;
                args[t].next = next;
                args[t].cx = cx;
                args[t].cy = cy;
                args[t].cz = cz;
                args[t].lat = &lat;
                args[t].hist_total = local_total[t];
                args[t].hist_pair = local_pair[t];
                pthread_create(&threads[t], NULL, pair_worker, &args[t]);
            }
            t_cell_total += now_seconds() - t_cell_start;
            
            t_pair_start = now_seconds();
            for (t = 0; t < use_threads; t++) {
                int p;
                pthread_join(threads[t], NULL);
                for (p = 0; p < nbin; p++) {
                    count[p] += local_total[t][p] * lat.volume / ((double)n_atoms * (double)n_atoms);
                }
                for (i = 0; i < ntypes; i++) {
                    for (j = i; j < ntypes; j++) {
                        double denom = (double)ni_frame[i] * (double)ni_frame[j];
                        if (denom <= 0.0) {
                            continue;
                        }
                        for (l = 0; l < nbin; l++) {
                            size_t idx = pair_hist_index(i, j, ntypes, l, nbin);
                            countij[idx] += local_pair[t][idx] * lat.volume / denom;
                        }
                    }
                }
                free(local_total[t]);
                free(local_pair[t]);
            }
            t_pair_total += now_seconds() - t_pair_start;

            free(threads);
            free(args);
            free(local_total);
            free(local_pair);
            free(head);
            free(next);
            free(cx);
            free(cy);
            free(cz);
        } else {
            const double *fproj = (proj_axis == 0) ? fx : (proj_axis == 1 ? fy : fz);
            double axis_len = (proj_axis == 0) ? lat.a : (proj_axis == 1 ? lat.b : lat.c);
            for (i = 0; i < n_atoms; i++) {
                double wp = fabs(fproj[i]);
                double pdist;
                if (wp > 0.5) {
                    wp = 1.0 - wp;
                }
                pdist = wp * axis_len;
                l = (int)(pdist * inv_dr);
                if (l < 0 || l >= nbin) {
                    continue;
                }
                count[l] += 1.0 / (double)n_atoms;
                if (ni_frame[atype[i]] > 0) {
                    size_t idx = pair_hist_index(atype[i], 0, ntypes, l, nbin);
                    countij[idx] += 1.0 / (double)ni_frame[atype[i]];
                }
            }
        }

        frames_used++;
        vmean += lat.volume;
        progressbar("Reading trajectory", frames_used - 1, (max_images > 0 ? max_images : frames_used + 1), 24);

        free(ni_frame);
    }

    t_file_end = now_seconds();
    fprintf(stderr, "\n");
    fprintf(stderr, "\n\033[1;31mTiming Summary:\033[0m\n");
    fprintf(stderr, "  File parsing:       %.3f s\n", t_file_end - t_file_start - t_cell_total - t_pair_total);
    fprintf(stderr, "  Cell-list build:    %.3f s (%.1f%%)\n", t_cell_total, 100.0 * t_cell_total / (t_file_end - t_file_start));
    fprintf(stderr, "  Pair accumulation:  %.3f s (%.1f%%)\n", t_pair_total, 100.0 * t_pair_total / (t_file_end - t_file_start));
    fprintf(stderr, "  Total elapsed:      %.3f s\n\n", t_file_end - t_file_start);

    if (frames_used <= 0) {
        fprintf(stderr, "Error: no frames processed\n");
        goto fail;
    }

    vmean /= (double)frames_used;

    /**
     * NORMALIZATION SECTION
     * =====================
     * Three normalization steps to produce final g(r) values:
     *     * Step 1: Time-averaging
     *   count[l] /= frames_used;
     *   countij[i][j][l] /= frames_used;
     *   (undoes the per-frame accumulation that included 1/frames factor)
     *
     * Step 2: Volume normalization (ideal gas reference)
     *   nideal = 4π/3 * (r_upper³ - r_lower³)
     *   count[l] /= nideal;
     *   countij[i][j][l] /= nideal;
     *   (defines g(r)=1 for ideal gas at cell density)
     *
     * Step 3: Type-specific normalization (already done during accumulation)
     *   countij[i][j][l] already contains (accumulated_pairs) / (Ni * Nj)
     *   (normalizes by number of i-type and j-type atoms in the frame)
     *
     * Result: g(r) = (observed_pairs) / (ideal_pairs at same density)
     *         countij[i][j][l] = (observed_ij_pairs) / (ideal_ij_pairs)
     *         integral = cumulative coordination number up to r
     */

    if (proj_axis < 0) {
        for (l = 1; l < nbin; l++) {
            double rlo = (double)l * dr;
            double rup = (double)(l + 1) * dr;
            double nideal = (rup * rup * rup - rlo * rlo * rlo) * 4.0 * PI / 3.0;
            count[l] = (count[l] / (double)frames_used) / nideal;
            for (i = 0; i < ntypes; i++) {
                for (j = i; j < ntypes; j++) {
                    size_t idx = pair_hist_index(i, j, ntypes, l, nbin);
                    countij[idx] = (countij[idx] / (double)frames_used) / nideal;
                }
            }
        }
        count[0] = 0.0;
        for (i = 0; i < ntypes; i++) {
            for (j = i; j < ntypes; j++) {
                size_t idx = pair_hist_index(i, j, ntypes, 0, nbin);
                countij[idx] = 0.0;
            }
        }

        for (l = 0; l < nbin; l++) {
            for (j = 0; j < l; j++) {
                double r = dr * (double)j + 0.5 * dr;
                integ[l] += count[j] * dr * 4.0 * PI * r * r * (double)nref / vmean;
                for (i = 0; i < ntypes; i++) {
                    int tj;
                    for (tj = i; tj < ntypes; tj++) {
                        size_t idxm = pair_hist_index(i, tj, ntypes, j, nbin);
                        size_t idxl = pair_hist_index(i, tj, ntypes, l, nbin);
                        integij[idxl] += countij[idxm] * dr * 4.0 * PI * r * r * (double)ni_ref[tj] / vmean;
                    }
                }
            }
        }
    } else {
        double integ_type[MAX_TYPES] = {0.0};
        for (l = 0; l < nbin; l++) {
            count[l] /= (double)frames_used;
            for (i = 0; i < ntypes; i++) {
                size_t idx = pair_hist_index(i, 0, ntypes, l, nbin);
                countij[idx] /= (double)frames_used;
            }
        }
        integ[0] = 0.0;
        for (i = 0; i < ntypes; i++) {
            size_t idx0 = pair_hist_index(i, 0, ntypes, 0, nbin);
            integij[idx0] = 0.0;
        }
        for (l = 1; l < nbin; l++) {
            integ[l] = integ[l - 1] + count[l - 1] * dr;
            for (i = 0; i < ntypes; i++) {
                size_t idx_prev = pair_hist_index(i, 0, ntypes, l - 1, nbin);
                size_t idx_cur = pair_hist_index(i, 0, ntypes, l, nbin);
                integ_type[i] += countij[idx_prev] * dr;
                integij[idx_cur] = integ_type[i];
            }
        }
    }

    if (!tidy) {
        if (proj_axis < 0) {
            fprintf(stdout, "r,G(r)");
            for (i = 0; i < ntypes; i++) {
                for (j = i; j < ntypes; j++) {
                    fprintf(stdout, ",G_{%s%s}(r)", type_names[i], type_names[j]);
                }
            }
            for (i = 0; i < ntypes; i++) {
                for (j = i; j < ntypes; j++) {
                    fprintf(stdout, ",N_{%s%s}(r)", type_names[i], type_names[j]);
                }
            }
            fprintf(stdout, "\n");
            for (l = 0; l < nbin; l++) {
                double rv = dr * (double)l + 0.5 * dr;
                if (rv >= rmax) {
                    break;
                }
                fprintf(stdout, "%.6f,%.6f", rv, count[l]);
                for (i = 0; i < ntypes; i++) {
                    for (j = i; j < ntypes; j++) {
                        size_t idx = pair_hist_index(i, j, ntypes, l, nbin);
                        fprintf(stdout, ",%.6f", countij[idx]);
                    }
                }
                for (i = 0; i < ntypes; i++) {
                    for (j = i; j < ntypes; j++) {
                        size_t idx = pair_hist_index(i, j, ntypes, l, nbin);
                        fprintf(stdout, ",%.6f", integij[idx]);
                    }
                }
                fprintf(stdout, "\n");
            }
        } else {
            char axis = (proj_axis == 0) ? 'x' : (proj_axis == 1 ? 'y' : 'z');
            fprintf(stdout, "%c,D(%c),I(%c)", axis, axis, axis);
            for (i = 0; i < ntypes; i++) {
                fprintf(stdout, ",D_{%s}(%c),I_{%s}(%c)", type_names[i], axis, type_names[i], axis);
            }
            fprintf(stdout, "\n");
            for (l = 0; l < nbin; l++) {
                double pv = dr * (double)l + 0.5 * dr;
                fprintf(stdout, "%.6f,%.6f,%.6f", pv, count[l], integ[l]);
                for (i = 0; i < ntypes; i++) {
                    size_t idx = pair_hist_index(i, 0, ntypes, l, nbin);
                    fprintf(stdout, ",%.6f,%.6f", countij[idx], integij[idx]);
                }
                fprintf(stdout, "\n");
            }
        }
    } else {
        if (proj_axis < 0) {
            fprintf(stdout, "r,elements,G(r),integral\n");
            for (l = 0; l < nbin; l++) {
                double rv = dr * (double)l + 0.5 * dr;
                if (rv >= rmax) {
                    break;
                }
                fprintf(stdout, "%.6f,total,%.6f,%.6f\n", rv, count[l], integ[l]);
                for (i = 0; i < ntypes; i++) {
                    for (j = i; j < ntypes; j++) {
                        char pair[MAX_TYPE_NAME * 2 + 4];
                        size_t idx = pair_hist_index(i, j, ntypes, l, nbin);
                        snprintf(pair, sizeof(pair), "%s%s", type_names[i], type_names[j]);
                        fprintf(stdout, "%.6f,%s,%.6f,%.6f\n", rv, pair, countij[idx], integij[idx]);
                    }
                }
            }
        } else {
            char axis = (proj_axis == 0) ? 'x' : (proj_axis == 1 ? 'y' : 'z');
            fprintf(stdout, "%c,element,density,integral\n", axis);
            for (l = 0; l < nbin; l++) {
                double pv = dr * (double)l + 0.5 * dr;
                fprintf(stdout, "%.6f,total,%.6f,%.6f\n", pv, count[l], integ[l]);
                for (i = 0; i < ntypes; i++) {
                    size_t idx = pair_hist_index(i, 0, ntypes, l, nbin);
                    fprintf(stdout, "%.6f,%s,%.6f,%.6f\n", pv, type_names[i], countij[idx], integij[idx]);
                }
            }
        }
    }

    fclose(fp);
    free(fx);
    free(fy);
    free(fz);
    free(atype);
    free(count);
    free(integ);
    free(countij);
    free(integij);
    return 0;

fail:
    fclose(fp);
    free(fx);
    free(fy);
    free(fz);
    free(atype);
    free(count);
    free(integ);
    free(countij);
    free(integij);
    return 1;
}
