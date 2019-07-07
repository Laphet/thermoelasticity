#include "cell.h"

void get_wall_time(char *buffer)
{
    time_t timer;
    struct tm *tm_info;
    time(&timer);
    tm_info = localtime(&timer);
    strftime(buffer, 26, "%H:%M:%S", tm_info);
}

void load_data(double *data, FILE *f, int max_length)
{
    int i;
    for (i = 0; i < max_length; ++i)
        fscanf(f, "%lf", &data[i]);
}

void get_default_cell(Cell *default_cell)
{
    default_cell->lambda0 = 1.0;
    default_cell->lambda1 = 2.0;
    default_cell->mu0 = 3.0;
    default_cell->mu1 = 4.0;
    default_cell->beta0 = 1.5;
    default_cell->beta1 = 2.5;
    default_cell->kappa0 = 2.9;
    default_cell->kappa1 = 3.2;
    default_cell->ratio = 0.25;
}

void get_homo_thermoequ_cell(Cell *cell, double homo_kappa)
{
    get_default_cell(cell);
    cell->kappa0 = homo_kappa;
    cell->kappa1 = homo_kappa;
}