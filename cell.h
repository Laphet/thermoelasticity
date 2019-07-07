#ifndef YCQ_CELL_H
#define YCQ_CELL_H

#include "time.h"
#include "stdio.h"

typedef struct Cell
{
    double lambda0, lambda1, mu0, mu1, beta0, beta1, kappa0, kappa1;
    double ratio;
} Cell;

void get_wall_time(char *);
void load_data(double *, FILE *, int);
void get_default_cell(Cell *);
void get_homo_thermoequ_cell(Cell *, double);

#endif