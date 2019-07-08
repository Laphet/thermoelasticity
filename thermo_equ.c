static char help[] = "gogogo children!\n\n";
static char wall_time[26];

#include "petscksp.h"
#include "petscdm.h"
#include "petscdmda.h"
#include "petscviewerhdf5.h"
#include "math.h"
#include "string.h"
#include "cell.h"

#define DIM 3
#define NODE_NUM_PER_ELE 8
#define NODE_NUM_PER_FACE 4
#define MAX_LENGTH_FILE_NAME 57

typedef struct Context
{
    Cell cell;
    int nc, nse;
    double H;
    double Tx0, Tx1, Ty0, Ty1;
    double Tz0, alpha;
} Context;

static double stiff[NODE_NUM_PER_ELE * NODE_NUM_PER_ELE];
static double stiffRB[NODE_NUM_PER_FACE * NODE_NUM_PER_FACE];

void set_default_bdrycons(Context *ctx)
{
    ctx->H = 3.14;
    ctx->Tx0 = 1.00;
    ctx->Tx1 = 1.50;
    ctx->Ty0 = 1.25;
    ctx->Ty1 = 1.75;
    ctx->Tz0 = 1.00;
}

void set_default_mesh(Context *ctx)
{
    ctx->nc = 16;
    ctx->nse = 4;
}

void set_test_context(Context *ctx)
{
    // -Delta u = 0.0 u = 1-z
    ctx->cell.kappa0 = 1.0;
    ctx->cell.kappa1 = 1.0;
    ctx->H = 0.0;
    ctx->Tx0 = 0.0;
    ctx->Tx1 = 0.0;
    ctx->Ty0 = 0.0;
    ctx->Ty1 = 0.0;
    ctx->Tz0 = 2.0;
    ctx->alpha = 1.0;
}

void load_stiffness_data(FILE *f, double *stiff)
{
    int t = DIM * NODE_NUM_PER_ELE;
    double stiffA[t * t];
    load_data(stiffA, f, t * t);
    int i, j;
    for (i = 0; i < NODE_NUM_PER_ELE; ++i)
        for (j = 0; j < NODE_NUM_PER_ELE; ++j)
            stiff[i * NODE_NUM_PER_ELE + j] = stiffA[(DIM * i) * t + (DIM * j)] +
                                              stiffA[(DIM * i + 1) * t + (DIM * j + 1)] +
                                              stiffA[(DIM * i + 2) * t + (DIM * j + 2)];
}

double get_kappa(int ele_ind_x, int ele_ind_y, int ele_ind_z, Context *ctx)
{
    int width = round(ctx->nse * ctx->cell.ratio);
    ele_ind_x = ele_ind_x % ctx->nse;
    ele_ind_y = ele_ind_y % ctx->nse;
    ele_ind_z = ele_ind_z % ctx->nse;
    int flag = (width <= ele_ind_x) && (ele_ind_x < ctx->nse - width) &&
               (width <= ele_ind_y) && (ele_ind_y < ctx->nse - width) &&
               (width <= ele_ind_z) && (ele_ind_z < ctx->nse - width);
    if (flag)
        return ctx->cell.kappa0;
    else
        return ctx->cell.kappa1;
}

void get_stencil(MatStencil base_st, MatStencil *st, int local_index, Context *ctx)
{
    int lidx = local_index % 2;
    int lidy = (local_index / 2) % 2;
    int lidz = local_index / 4;
    st->i = base_st.i + lidx;
    st->j = base_st.j + lidy;
    st->k = base_st.k + lidz;
    st->k = (st->k >= ctx->nc * ctx->nse) ? -1 : st->k;
}

extern PetscErrorCode ComputeMatrix(KSP, Mat, Mat, void *);
extern PetscErrorCode ComputeRHS(KSP, Vec, void *);
extern PetscErrorCode ComputeInitialGuess(KSP, Vec, void *);
extern PetscErrorCode ComputeError(KSP, Context *);

int main(int argc, char **argv)
{
    FILE *f;
    f = fopen("stiffA.dat", "r");
    load_stiffness_data(f, stiff);
    fclose(f);
    f = fopen("stiffRB.dat", "r");
    load_data(stiffRB, f, NODE_NUM_PER_FACE * NODE_NUM_PER_FACE);
    fclose(f);
    Context ctx;
    get_default_cell(&(ctx.cell));
    set_default_bdrycons(&ctx);
    set_default_mesh(&ctx);

    PetscErrorCode ierr;
    KSP ksp;
    PetscReal norm;
    DM da;
    Vec x, b, r;
    Mat A;
    PetscViewer fw;
    char file_name[MAX_LENGTH_FILE_NAME];
    int total_ne, iter_count;
    PetscBool test_flag = PETSC_FALSE;

    ierr = PetscInitialize(&argc, &argv, (char *)0, help);
    if (ierr)
        return ierr;
    ierr = PetscOptionsGetInt(NULL, NULL, "-nc", &(ctx.nc), NULL);
    ierr = PetscOptionsGetInt(NULL, NULL, "-nse", &(ctx.nse), NULL);
    ierr = PetscOptionsGetBool(NULL, NULL, "-test", &test_flag, NULL);
    CHKERRQ(ierr);
    if (test_flag)
        set_test_context(&ctx);
    total_ne = ctx.nc * ctx.nse;
    get_wall_time(wall_time);
    PetscPrintf(PETSC_COMM_WORLD,
                "%s\tSTART!\tuse cells (%d*%d*%d), subelements (%d*%d*%d) and total elements (%d*%d*%d).\n",
                wall_time, ctx.nc, ctx.nc, ctx.nc, ctx.nse, ctx.nse, ctx.nse, total_ne, total_ne, total_ne);
    ierr = KSPCreate(PETSC_COMM_WORLD, &ksp);
    CHKERRQ(ierr);
    ierr = DMDACreate3d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,
                        DMDA_STENCIL_BOX, total_ne + 1, total_ne + 1, total_ne,
                        PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE, 1, 1,
                        0, 0, 0, &da);
    CHKERRQ(ierr);
    ierr = DMSetFromOptions(da);
    CHKERRQ(ierr);
    ierr = DMSetUp(da);
    CHKERRQ(ierr);
    ierr = KSPSetDM(ksp, da);
    CHKERRQ(ierr);
    ierr = KSPSetComputeInitialGuess(ksp, ComputeInitialGuess, NULL);
    CHKERRQ(ierr);
    ierr = KSPSetComputeRHS(ksp, ComputeRHS, &ctx);
    CHKERRQ(ierr);
    ierr = KSPSetComputeOperators(ksp, ComputeMatrix, &ctx);
    CHKERRQ(ierr);
    ierr = DMDestroy(&da);
    CHKERRQ(ierr);

    ierr = KSPSetFromOptions(ksp);
    CHKERRQ(ierr);
    ierr = KSPSolve(ksp, NULL, NULL);
    CHKERRQ(ierr);
    ierr = KSPGetIterationNumber(ksp, &iter_count);
    CHKERRQ(ierr);
    ierr = KSPGetSolution(ksp, &x);
    ierr = PetscObjectSetName((PetscObject)x, "theta");
    CHKERRQ(ierr);
    ierr = KSPGetRhs(ksp, &b);
    CHKERRQ(ierr);
    ierr = VecDuplicate(b, &r);
    CHKERRQ(ierr);
    ierr = KSPGetOperators(ksp, &A, NULL);
    CHKERRQ(ierr);

    ierr = MatMult(A, x, r);
    CHKERRQ(ierr);
    ierr = VecAXPY(r, -1.0, b);
    CHKERRQ(ierr);
    ierr = VecNorm(r, NORM_2, &norm);
    CHKERRQ(ierr);
    get_wall_time(wall_time);
    PetscPrintf(PETSC_COMM_WORLD, "%s\tThe linear system has been solved, iteration num=%d and residual norm=%.4f.\n",
                wall_time, iter_count, (double)norm);

    if (test_flag)
    {
        ierr = ComputeError(ksp, &ctx);
        CHKERRQ(ierr);
    }

    snprintf(file_name, MAX_LENGTH_FILE_NAME, "data/theta-nc%d-nse%d.hdf5", ctx.nc, ctx.nse);
    ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD, file_name, FILE_MODE_WRITE, &fw);
    CHKERRQ(ierr);
    ierr = VecView(x, fw);
    CHKERRQ(ierr);
    get_wall_time(wall_time);
    PetscPrintf(PETSC_COMM_WORLD, "%s\tThe solution has been saved into file [%s].\n",
                wall_time, file_name);
    ierr = PetscViewerDestroy(&fw);
    CHKERRQ(ierr);

    ierr = VecDestroy(&r);
    CHKERRQ(ierr);
    ierr = KSPDestroy(&ksp);
    CHKERRQ(ierr);
    ierr = PetscFinalize();
    return ierr;
}

PetscErrorCode ComputeMatrix(KSP ksp, Mat jac, Mat B, void *ctx_)
{
    DM dm;
    PetscErrorCode ierr;
    MatStencil row[NODE_NUM_PER_ELE], col[NODE_NUM_PER_ELE], base_st;
    int i, j, k, xm, ym, zm, xs, ys, zs, a, b, a_, b_;
    double v[NODE_NUM_PER_ELE * NODE_NUM_PER_ELE], h, kappa;

    Context *ctx = (Context *)ctx_;
    int total_ne = ctx->nc * ctx->nse;
    h = 1.0 / (double)total_ne;

    PetscFunctionBeginUser;
    ierr = KSPGetDM(ksp, &dm);
    CHKERRQ(ierr);
    ierr = DMDAGetCorners(dm, &xs, &ys, &zs, &xm, &ym, &zm);

    for (i = xs; (i < xs + xm) && (i < total_ne); ++i)
        for (j = ys; (j < ys + ym) && (j < total_ne); ++j)
            for (k = zs; (k < zs + zm) && (k < total_ne); ++k)
            {
                base_st.i = i;
                base_st.j = j;
                base_st.k = k;
                kappa = get_kappa(i, j, k, ctx);
                for (a = 0; a < NODE_NUM_PER_ELE; ++a)
                {
                    get_stencil(base_st, &row[a], a, ctx);
                    for (b = 0; b < NODE_NUM_PER_ELE; ++b)
                    {
                        get_stencil(base_st, &col[b], b, ctx);
                        v[a * NODE_NUM_PER_ELE + b] = h * kappa * stiff[a * NODE_NUM_PER_ELE + b];
                        // Robin boundary condition introduces another stiffness
                        if (row[a].k == 0 && col[b].k == 0)
                        {
                            a_ = a % NODE_NUM_PER_FACE;
                            b_ = b % NODE_NUM_PER_FACE;
                            v[a * NODE_NUM_PER_ELE + b] += h * h * ctx->alpha * stiffRB[a_ * NODE_NUM_PER_FACE + b_];
                        }
                    }
                }
                ierr = MatSetValuesBlockedStencil(B, NODE_NUM_PER_ELE, row, NODE_NUM_PER_ELE, col, v, ADD_VALUES);
                CHKERRQ(ierr);
            }

    ierr = MatAssemblyBegin(B, MAT_FINAL_ASSEMBLY);
    CHKERRQ(ierr);
    ierr = MatAssemblyEnd(B, MAT_FINAL_ASSEMBLY);
    CHKERRQ(ierr);
    get_wall_time(wall_time);
    PetscPrintf(PETSC_COMM_WORLD, "%s\tFinish constructing operator.\n",
                wall_time);
    PetscFunctionReturn(0);
}

PetscErrorCode ComputeRHS(KSP ksp, Vec b, void *ctx_)
{
    PetscErrorCode ierr;
    PetscInt i, j, k, xm, ym, zm, xs, ys, zs, a, i_, j_, k_;
    DM dm;
    Vec local_b;
    PetscScalar ***barray;
    double h;
    int total_ne;
    MatStencil base_st, row;

    Context *ctx = (Context *)ctx_;
    total_ne = ctx->nc * ctx->nse;
    h = 1.0 / (double)(total_ne);

    PetscFunctionBeginUser;
    ierr = KSPGetDM(ksp, &dm);
    CHKERRQ(ierr);

    ierr = DMDAGetCorners(dm, &xs, &ys, &zs, &xm, &ym, &zm);
    CHKERRQ(ierr);
    ierr = DMGetLocalVector(dm, &local_b);
    CHKERRQ(ierr);
    ierr = VecZeroEntries(local_b);
    ierr = DMDAVecGetArray(dm, local_b, &barray);
    CHKERRQ(ierr);

    for (k = zs; (k < zs + zm) && (k < total_ne); k++)
    {
        for (j = ys; (j < ys + ym) && (j < total_ne); j++)
        {
            for (i = xs; (i < xs + xm) && (i < total_ne); i++)
            {
                base_st.i = i;
                base_st.j = j;
                base_st.k = k;
                for (a = 0; a < NODE_NUM_PER_ELE; ++a)
                {
                    get_stencil(base_st, &row, a, ctx);
                    if (((i_ = row.i) >= 0) && ((j_ = row.j) >= 0) && ((k_ = row.k) >= 0))
                    {
                        // Source term
                        barray[k_][j_][i_] += h * h * h * ctx->H / (double)(NODE_NUM_PER_ELE);
                        // Neumann terms
                        if (i_ == 0)
                            barray[k_][j_][i_] += h * h * ctx->Tx0 / (double)(NODE_NUM_PER_FACE);
                        if (i_ == total_ne)
                            barray[k_][j_][i_] += h * h * ctx->Tx1 / (double)(NODE_NUM_PER_FACE);
                        if (j_ == 0)
                            barray[k_][j_][i_] += h * h * ctx->Ty0 / (double)(NODE_NUM_PER_FACE);
                        if (j_ == total_ne)
                            barray[k_][j_][i_] += h * h * ctx->Ty1 / (double)(NODE_NUM_PER_FACE);
                        // Robin term
                        if (k_ == 0)
                            barray[k_][j_][i_] += h * h * ctx->Tz0 / (double)(NODE_NUM_PER_FACE);
                        /* 
                        get_wall_time(wall_time);
                        PetscPrintf(PETSC_COMM_WORLD,
                        "%s\ti_=%d\tj_%d\tk_=%d.\n", wall_time, i_, j_, k_);
                        */
                    }
                }
            }
        }
    }
    ierr = DMDAVecRestoreArray(dm, local_b, &barray);
    CHKERRQ(ierr);
    ierr = DMLocalToGlobal(dm, local_b, ADD_VALUES, b);
    CHKERRQ(ierr);
    get_wall_time(wall_time);
    PetscPrintf(PETSC_COMM_WORLD, "%s\tFinish constructing rhs.\n", wall_time);

    PetscFunctionReturn(0);
}

PetscErrorCode ComputeInitialGuess(KSP ksp, Vec b, void *ctx)
{
    PetscErrorCode ierr;

    PetscFunctionBeginUser;
    ierr = VecSet(b, 0);
    CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

PetscErrorCode ComputeError(KSP ksp, Context *ctx)
{
    PetscErrorCode ierr;
    PetscInt i, j, k, xm, ym, zm, xs, ys, zs;
    DM dm;
    Vec x, u;
    PetscScalar ***barray;
    double h, error;
    int total_ne;

    total_ne = ctx->nc * ctx->nse;
    h = 1.0 / (double)(total_ne);

    PetscFunctionBeginUser;
    ierr = KSPGetDM(ksp, &dm);
    ierr = KSPGetSolution(ksp, &x);
    ierr = VecDuplicate(x, &u);
    CHKERRQ(ierr);

    ierr = DMDAGetCorners(dm, &xs, &ys, &zs, &xm, &ym, &zm);
    CHKERRQ(ierr);
    ierr = DMDAVecGetArray(dm, u, &barray);
    CHKERRQ(ierr);

    for (k = zs; k < zs + zm; ++k)
        for (j = ys; j < ys + ym; ++j)
            for (i = xs; i < xs + xm; ++i)
                barray[k][j][i] = 1.0 - k * h;

    ierr = DMDAVecRestoreArray(dm, u, &barray);
    CHKERRQ(ierr);
    ierr = VecAXPY(u, -1.0, x);
    CHKERRQ(ierr);
    ierr = VecNorm(u, NORM_INFINITY, &error);
    CHKERRQ(ierr);
    get_wall_time(wall_time);
    PetscPrintf(PETSC_COMM_WORLD, "%s\tUse (-Laplace u = 0, u=1-z) as test problem, the error in l^infinity norm is %.4f.\n",
                wall_time, error);

    PetscFunctionReturn(0);
}