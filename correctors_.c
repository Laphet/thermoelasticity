static char help[] = "gogo children!\n\n";
static char wall_time[26];

#include "petscksp.h"
#include "petscviewerhdf5.h"
#include "stdio.h"
#include "math.h"
#include "time.h"
#include "string.h"

#define DIM 3
#define FREEDOM_DEG_PER_ELE 24
#define CORRECTORS_NUM_N 6
#define CORRECTORS_NUM_M 3
#define NODE_NUM_PER_ELE 8
#define MAX_LENGTH_FILE_NAME 26

void get_wall_time(char *buffer)
{
    time_t timer;
    struct tm *tm_info;
    time(&timer);
    tm_info = localtime(&timer);
    strftime(buffer, 26, "%H:%M:%S", tm_info);
}

typedef struct Context
{
    double lambda0, lambda1, mu0, mu1, beta0, beta1, kappa0, kappa1;
    double ratio;
    double tol;
    int ne;
} Context;

static double stiffA[FREEDOM_DEG_PER_ELE * FREEDOM_DEG_PER_ELE];
static double stiffB[FREEDOM_DEG_PER_ELE * FREEDOM_DEG_PER_ELE];
static double stiffC[FREEDOM_DEG_PER_ELE * FREEDOM_DEG_PER_ELE];
static double load[DIM * NODE_NUM_PER_ELE];
static char correctors_name[CORRECTORS_NUM_N + 1 + CORRECTORS_NUM_M][MAX_LENGTH_FILE_NAME] = {"N11", "N12", "N13", "N22", "N23", "N33", "P", "M1", "M2", "M3"};

void get_local_coefficients(int ele_ind_x, int ele_ind_y, int ele_ind_z, double *lambda, double *mu, double *beta, double *kappa, Context *ctx)
{
    int width = round(ctx->ne * ctx->ratio);
    if ((width <= ele_ind_x) && (ele_ind_x < ctx->ne - width) && (width <= ele_ind_y) && (ele_ind_y < ctx->ne - width) && (width <= ele_ind_z) && (ele_ind_z < ctx->ne - width))
    {
        *lambda = ctx->lambda0;
        *mu = ctx->mu0;
        *beta = ctx->beta0;
        *kappa = ctx->kappa0;
    }
    else
    {
        *lambda = ctx->lambda1;
        *mu = ctx->mu1;
        *beta = ctx->beta1;
        *kappa = ctx->kappa1;
    }
}

int get_freedom_index(int is_scalar, int idx, int idy, int idz, int i, Context *ctx)
{
    idx = idx % ctx->ne;
    idy = idy % ctx->ne;
    idz = idz % ctx->ne;
    if (is_scalar)
        return idx + ctx->ne * idy + ctx->ne * ctx->ne * idz;
    else
        return DIM * (idx + ctx->ne * idy + ctx->ne * ctx->ne * idz) + i;
}

void decode_local_index(int is_scalar, int a, int *ax, int *ay, int *az, int *i)
// a = 3(ax+2ay+4az)+i  --- vector
// a = ax+2ay+4az       --- scalar
{
    if (is_scalar)
    {
        *az = a / 4;
        *ay = (a / 2) % 2;
        *ax = a % 2;
    }
    else
    {
        *i = a % DIM;
        *az = (a / DIM) / 4;
        *ay = (((a / DIM)) / 2) % 2;
        *ax = (a / DIM) % 2;
    }
}

double get_local_grad_pair_integral(double *local_data, int i, int j)
// Calculate int_hat(T) partial N_i/partial y_j+partial N_j/partial y_i dy
// len(local_data)=24
{
    double ans = 0.0;
    int k;
    for (k = 0; k < NODE_NUM_PER_ELE; ++k)
    {
        ans += local_data[DIM * k + i] * load[DIM * k + j];
        ans += local_data[DIM * k + j] * load[DIM * k + i];
    }
    return ans;
}

double
get_local_div_integral(double *local_data)
// Calculate int_hat(T) Div(N) dy
// len(local_data) = 24
{
    double ans = 0.0;
    int k;
    for (k = 0; k < FREEDOM_DEG_PER_ELE; ++k)
        ans += local_data[k] * load[k];
    return ans;
}

double
get_local_grad_integral(double *local_data, int i)
// Calculate int_hat(T) partial M/partial y_i dy
// len(local_data) = 8
{
    double ans = 0.0;
    int k;
    for (k = 0; k < NODE_NUM_PER_ELE; ++k)
    {
        ans += local_data[k] * load[DIM * k + i];
    }
    return ans;
}

void load_data(double *data, FILE *f, int max_length)
{
    int i;
    for (i = 0; i < max_length; ++i)
        fscanf(f, "%lf", &data[i]);
}

int main(int argc, char **args)
{
    // Load datas of local stiffness and local load.
    FILE *f;
    f = fopen("stiffA.dat", "r");
    load_data(stiffA, f, FREEDOM_DEG_PER_ELE * FREEDOM_DEG_PER_ELE);
    fclose(f);
    f = fopen("stiffB.dat", "r");
    load_data(stiffB, f, FREEDOM_DEG_PER_ELE * FREEDOM_DEG_PER_ELE);
    fclose(f);
    f = fopen("stiffC.dat", "r");
    load_data(stiffC, f, FREEDOM_DEG_PER_ELE * FREEDOM_DEG_PER_ELE);
    fclose(f);
    f = fopen("load.dat", "r");
    load_data(load, f, FREEDOM_DEG_PER_ELE);
    fclose(f);
    Context ctx = {.lambda0 = 1.0, .lambda1 = 2.0, .mu0 = 3.0, .mu1 = 4.0, .beta0 = 1.5, .beta1 = 2.5, .kappa0 = 2.9, .kappa1 = 3.2, .ratio = 0.25, .ne = 16, .tol = 1.0e-7};
    Mat A, A_;
    Vec *rhs, *rhs_;
    Vec *N, P, *M;
    Vec ones, ones_;
    // A x = rhs    --- vector
    // A_ x_ = rhs_ --- scalar
    KSP ksp, ksp_;
    int total_freedom_deg, total_element_num;
    PetscErrorCode ierr;
    ierr = PetscInitialize(&argc, &args, (char *)0, help);
    if (ierr)
        return ierr;
    ierr = PetscOptionsGetInt(NULL, NULL, "-ne", &(ctx.ne), NULL);
    CHKERRQ(ierr);
    get_wall_time(wall_time);
    PetscPrintf(PETSC_COMM_SELF, "%s\tSTART!\tuse grid (%d*%d*%d).\n", wall_time, ctx.ne, ctx.ne, ctx.ne);
    // Create PETSc vector and matrix objects, here we use sequencial form.
    total_element_num = ctx.ne * ctx.ne * ctx.ne;
    total_freedom_deg = DIM * total_element_num;
    ierr = VecCreateSeq(PETSC_COMM_SELF, total_freedom_deg, &ones);
    ierr = VecCreateSeq(PETSC_COMM_SELF, total_element_num, &ones_);
    CHKERRQ(ierr);
    ierr = VecSet(ones, 1.0);
    ierr = VecSet(ones_, 1.0);
    CHKERRQ(ierr);
    ierr = VecDuplicateVecs(ones, CORRECTORS_NUM_N + 1, &rhs);
    ierr = VecDuplicateVecs(ones, CORRECTORS_NUM_N, &N);
    ierr = VecDuplicate(ones, &P);
    ierr = VecDuplicateVecs(ones_, CORRECTORS_NUM_M, &rhs_);
    ierr = VecDuplicateVecs(ones_, CORRECTORS_NUM_M, &M);
    ierr = PetscObjectSetName((PetscObject)(N[0]), "N11");
    ierr = PetscObjectSetName((PetscObject)(N[1]), "N12");
    ierr = PetscObjectSetName((PetscObject)(N[2]), "N13");
    ierr = PetscObjectSetName((PetscObject)(N[3]), "N22");
    ierr = PetscObjectSetName((PetscObject)(N[4]), "N23");
    ierr = PetscObjectSetName((PetscObject)(N[5]), "N33");
    ierr = PetscObjectSetName((PetscObject)(P), "P");
    ierr = PetscObjectSetName((PetscObject)(M[0]), "M1");
    ierr = PetscObjectSetName((PetscObject)(M[1]), "M2");
    ierr = PetscObjectSetName((PetscObject)(M[2]), "M3");
    CHKERRQ(ierr);
    ierr = MatCreateSeqAIJ(PETSC_COMM_SELF, total_freedom_deg, total_freedom_deg, 3 * 3 * 3 * DIM, NULL, &A); // Each freedom degree intersects with 27*3 freedom degrees.
    ierr = MatCreateSeqAIJ(PETSC_COMM_SELF, total_element_num, total_element_num, 3 * 3 * 3, NULL, &A_);
    CHKERRQ(ierr);
    // Construct linear systems.
    get_wall_time(wall_time);
    PetscPrintf(PETSC_COMM_SELF, "%s\tBegin to construct linear systems.\n", wall_time);
    int ele_ind, ele_ind_x, ele_ind_y, ele_ind_z; // Element index components ele_ind = ele_ind_x+ne*ele_ind_y+ne*ne*ele_ind_z
    int ai, bj;                                   // Local index
    int aix, aiy, aiz, aik;                       // Local index components
    int bjx, bjy, bjz, bjk;                       // Local index components
    int mn;                                       // Correctors index
    double data_addto_A[FREEDOM_DEG_PER_ELE * FREEDOM_DEG_PER_ELE];
    double data_addto_rhs[CORRECTORS_NUM_N + 1][FREEDOM_DEG_PER_ELE];
    double data_addto_A_[NODE_NUM_PER_ELE * NODE_NUM_PER_ELE];
    double data_addto_rhs_[CORRECTORS_NUM_M][NODE_NUM_PER_ELE];
    double h = 1.0 / (double)ctx.ne, lambda, mu, beta, kappa;
    int ida[FREEDOM_DEG_PER_ELE];
    int idb[FREEDOM_DEG_PER_ELE];
    int ida_[NODE_NUM_PER_ELE];
    int idb_[NODE_NUM_PER_ELE];
    for (ele_ind = 0; ele_ind < total_element_num; ++ele_ind)
    {
        ele_ind_x = ele_ind % ctx.ne;
        ele_ind_y = (ele_ind / ctx.ne) % ctx.ne;
        ele_ind_z = ele_ind / (ctx.ne * ctx.ne);
        get_local_coefficients(ele_ind_x, ele_ind_y, ele_ind_z, &lambda, &mu, &beta, &kappa, &ctx);
        // vector system
        for (ai = 0; ai < FREEDOM_DEG_PER_ELE; ++ai)
        {
            decode_local_index(0, ai, &aix, &aiy, &aiz, &aik);
            ida[ai] = get_freedom_index(0, ele_ind_x + aix, ele_ind_y + aiy, ele_ind_z + aiz, aik, &ctx);
            for (bj = 0; bj < FREEDOM_DEG_PER_ELE; ++bj)
            {
                decode_local_index(0, bj, &bjx, &bjy, &bjz, &bjk);
                idb[bj] = get_freedom_index(0, ele_ind_x + bjx, ele_ind_y + bjy, ele_ind_z + bjz, bjk, &ctx);
                data_addto_A[ai * FREEDOM_DEG_PER_ELE + bj] = h * (lambda * stiffA[ai * FREEDOM_DEG_PER_ELE + bj] +
                                                                   mu * (stiffB[ai * FREEDOM_DEG_PER_ELE + bj] +
                                                                         stiffC[ai * FREEDOM_DEG_PER_ELE + bj]));
            }
            data_addto_rhs[0][ai] = -h * h * (lambda * load[ai] + mu * (2.0 * (double)(aik == 0) * load[ai - aik + 0]));               // m=1, n=1
            data_addto_rhs[3][ai] = -h * h * (lambda * load[ai] + mu * (2.0 * (double)(aik == 1) * load[ai - aik + 1]));               // m=2, n=2
            data_addto_rhs[5][ai] = -h * h * (lambda * load[ai] + mu * (2.0 * (double)(aik == 2) * load[ai - aik + 2]));               // m=3, n=3
            data_addto_rhs[1][ai] = -h * h * mu * ((double)(aik == 0) * load[ai - aik + 1] + (double)(aik == 1) * load[ai - aik + 0]); // m=1, n=2
            data_addto_rhs[2][ai] = -h * h * mu * ((double)(aik == 0) * load[ai - aik + 2] + (double)(aik == 2) * load[ai - aik + 0]); // m=1, n=3
            data_addto_rhs[4][ai] = -h * h * mu * ((double)(aik == 1) * load[ai - aik + 2] + (double)(aik == 2) * load[ai - aik + 1]); // m=2, n=3
            data_addto_rhs[6][ai] = -h * h * beta * load[ai];
        }
        // scalar system
        for (ai = 0; ai < NODE_NUM_PER_ELE; ++ai)
        {
            decode_local_index(1, ai, &aix, &aiy, &aiz, NULL);
            ida_[ai] = get_freedom_index(1, ele_ind_x + aix, ele_ind_y + aiy, ele_ind_z + aiz, 0, &ctx);
            for (bj = 0; bj < NODE_NUM_PER_ELE; ++bj)
            {
                decode_local_index(1, bj, &bjx, &bjy, &bjz, NULL);
                idb_[bj] = get_freedom_index(1, ele_ind_x + bjx, ele_ind_y + bjy, ele_ind_z + bjz, 0, &ctx);
                data_addto_A_[ai * NODE_NUM_PER_ELE + bj] = h * kappa * (stiffA[(DIM * ai) * FREEDOM_DEG_PER_ELE + DIM * bj] + stiffA[(DIM * ai + 1) * FREEDOM_DEG_PER_ELE + DIM * bj + 1] + stiffA[(DIM * ai + 2) * FREEDOM_DEG_PER_ELE + DIM * bj + 2]);
            }
            data_addto_rhs_[0][ai] = -h * h * kappa * load[DIM * ai];
            data_addto_rhs_[1][ai] = -h * h * kappa * load[DIM * ai + 1];
            data_addto_rhs_[2][ai] = -h * h * kappa * load[DIM * ai + 2];
        }
        // add values to linear systems
        ierr = MatSetValues(A, FREEDOM_DEG_PER_ELE, ida, FREEDOM_DEG_PER_ELE, idb, data_addto_A, ADD_VALUES);
        CHKERRQ(ierr);
        for (mn = 0; mn < CORRECTORS_NUM_N + 1; ++mn)
        {
            ierr = VecSetValues(rhs[mn], FREEDOM_DEG_PER_ELE, ida, &data_addto_rhs[mn][0], ADD_VALUES);
            CHKERRQ(ierr);
        }
        ierr = MatSetValues(A_, NODE_NUM_PER_ELE, ida_, NODE_NUM_PER_ELE, idb_, data_addto_A_, ADD_VALUES);
        CHKERRQ(ierr);
        ierr = VecSetValues(rhs_[0], NODE_NUM_PER_ELE, ida_, &data_addto_rhs_[0][0], ADD_VALUES);
        ierr = VecSetValues(rhs_[1], NODE_NUM_PER_ELE, ida_, &data_addto_rhs_[1][0], ADD_VALUES);
        ierr = VecSetValues(rhs_[2], NODE_NUM_PER_ELE, ida_, &data_addto_rhs_[2][0], ADD_VALUES);
        CHKERRQ(ierr);
    }
    ierr = MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
    CHKERRQ(ierr);
    ierr = MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);
    CHKERRQ(ierr);
    for (mn = 0; mn < CORRECTORS_NUM_N + 1; ++mn)
    {
        ierr = VecAssemblyBegin(rhs[mn]);
        CHKERRQ(ierr);
        ierr = VecAssemblyEnd(rhs[mn]);
        CHKERRQ(ierr);
    }
    ierr = MatAssemblyBegin(A_, MAT_FINAL_ASSEMBLY);
    CHKERRQ(ierr);
    ierr = MatAssemblyEnd(A_, MAT_FINAL_ASSEMBLY);
    CHKERRQ(ierr);
    for (mn = 0; mn < CORRECTORS_NUM_M; ++mn)
    {
        ierr = VecAssemblyBegin(rhs_[mn]);
        CHKERRQ(ierr);
        ierr = VecAssemblyEnd(rhs_[mn]);
        CHKERRQ(ierr);
    }
    // Need to tell the solver that the constant solution should not be included.
    MatNullSpace mns, mns_;
    double norm2, norm2_;
    // vector null space
    VecNorm(ones, NORM_2, &norm2);
    VecScale(ones, 1.0 / norm2);
    ierr = MatNullSpaceCreate(PETSC_COMM_SELF, PETSC_FALSE, 1, &ones, &mns);
    CHKERRQ(ierr);
    ierr = MatSetNullSpace(A, mns);
    ierr = MatSetTransposeNullSpace(A, mns);
    ierr = MatNullSpaceDestroy(&mns);
    CHKERRQ(ierr);
    // scalar null space
    VecNorm(ones_, NORM_2, &norm2_);
    VecScale(ones_, 1.0 / norm2_);
    ierr = MatNullSpaceCreate(PETSC_COMM_SELF, PETSC_FALSE, 1, &ones_, &mns_);
    CHKERRQ(ierr);
    ierr = MatSetNullSpace(A_, mns_);
    ierr = MatSetTransposeNullSpace(A_, mns_);
    ierr = MatNullSpaceDestroy(&mns_);
    CHKERRQ(ierr);
    get_wall_time(wall_time);
    PetscPrintf(PETSC_COMM_SELF, "%s\tFinish constructing linear system.\n", wall_time);
    // Build vector solver
    ierr = KSPCreate(PETSC_COMM_SELF, &ksp);
    CHKERRQ(ierr);
    ierr = KSPSetOperators(ksp, A, A);
    ierr = KSPSetFromOptions(ksp);
    CHKERRQ(ierr);
    // Build scalar solver
    ierr = KSPCreate(PETSC_COMM_SELF, &ksp_);
    CHKERRQ(ierr);
    ierr = KSPSetOperators(ksp_, A_, A_);
    ierr = KSPSetFromOptions(ksp_);
    CHKERRQ(ierr);
    // Solving process begins
    int iter_count;
    for (mn = 0; mn < CORRECTORS_NUM_N + 1 + CORRECTORS_NUM_M; ++mn)
    {
        if (mn < CORRECTORS_NUM_N + 1)
        {
            if (mn == CORRECTORS_NUM_N)
                ierr = KSPSolve(ksp, rhs[mn], P);
            else
                ierr = KSPSolve(ksp, rhs[mn], N[mn]);
            CHKERRQ(ierr);
            ierr = KSPGetIterationNumber(ksp, &iter_count);
            CHKERRQ(ierr);
        }
        else
        {
            ierr = KSPSolve(ksp_, rhs_[mn - CORRECTORS_NUM_N - 1], M[mn - CORRECTORS_NUM_N - 1]);
            CHKERRQ(ierr);
            ierr = KSPGetIterationNumber(ksp_, &iter_count);
            CHKERRQ(ierr);
        }
        get_wall_time(wall_time);
        PetscPrintf(PETSC_COMM_SELF, "%s\t%s has been solved, iteration num=%d.\n", wall_time, correctors_name[mn], iter_count);
    }
    // Saving correctors
    PetscViewer fw;
    char filename[MAX_LENGTH_FILE_NAME];
    for (mn = 0; mn < CORRECTORS_NUM_N + 1 + CORRECTORS_NUM_M; ++mn)
    {
        snprintf(filename, MAX_LENGTH_FILE_NAME, "data/%s-ne%d.hdf5", correctors_name[mn], ctx.ne);
        ierr = PetscViewerHDF5Open(PETSC_COMM_SELF, filename, FILE_MODE_WRITE, &fw);
        if (mn < CORRECTORS_NUM_N)
            ierr = VecView(N[mn], fw);
        else if (mn == CORRECTORS_NUM_N)
            ierr = VecView(P, fw);
        else
            ierr = VecView(M[mn - CORRECTORS_NUM_N - 1], fw);
        CHKERRQ(ierr);
        get_wall_time(wall_time);
        PetscPrintf(PETSC_COMM_SELF, "%s\t%s has been saved.\n", wall_time, correctors_name[mn]);
    }
    ierr = PetscViewerDestroy(&fw);
    CHKERRQ(ierr);
    // Calculate homogenized coefficients
    get_wall_time(wall_time);
    PetscPrintf(PETSC_COMM_SELF, "%s\tBegin to calculate homogenized coefficients.\n", wall_time);
    double C1111 = 0.0, C2222 = 0.0, C3333 = 0.0, C1122 = 0.0, C1133 = 0.0, C2233 = 0.0;
    double C1212 = 0.0, C1313 = 0.0, C2323 = 0.0, C1213 = 0.0, C1223 = 0.0, C1323 = 0.0;
    double C1112 = 0.0, C1113 = 0.0, C1123 = 0.0, C2213 = 0.0, C2223 = 0.0, C3323 = 0.0;
    double B11 = 0.0, B22 = 0.0, B33 = 0.0, B12 = 0.0, B13 = 0.0, B23 = 0.0;
    double K11 = 0.0, K22 = 0.0, K33 = 0.0, K12 = 0.0, K13 = 0.0, K23 = 0.0;
    double local_data[CORRECTORS_NUM_N + 1 + CORRECTORS_NUM_M][FREEDOM_DEG_PER_ELE];
    for (ele_ind = 0; ele_ind < total_element_num; ++ele_ind)
    {
        ele_ind_x = ele_ind % ctx.ne;
        ele_ind_y = (ele_ind / ctx.ne) % ctx.ne;
        ele_ind_z = ele_ind / (ctx.ne * ctx.ne);
        get_local_coefficients(ele_ind_x, ele_ind_y, ele_ind_z, &lambda, &mu, &beta, &kappa, &ctx);
        // Vector get local data index
        for (ai = 0; ai < FREEDOM_DEG_PER_ELE; ++ai)
        {
            decode_local_index(0, ai, &aix, &aiy, &aiz, &aik);
            ida[ai] = get_freedom_index(0, ele_ind_x + aix, ele_ind_y + aiy, ele_ind_z + aiz, aik, &ctx);
        }
        // Scalar get local data index
        for (ai = 0; ai < NODE_NUM_PER_ELE; ++ai)
        {
            decode_local_index(1, ai, &aix, &aiy, &aiz, NULL);
            ida_[ai] = get_freedom_index(1, ele_ind_x + aix, ele_ind_y + aiy, ele_ind_z + aiz, 0, &ctx);
        }
        for (mn = 0; mn < CORRECTORS_NUM_N + 1 + CORRECTORS_NUM_M; ++mn)
        {
            if (mn < CORRECTORS_NUM_N)
                ierr = VecGetValues(N[mn], FREEDOM_DEG_PER_ELE, ida, &local_data[mn][0]);
            else if (mn == CORRECTORS_NUM_N)
                ierr = VecGetValues(P, FREEDOM_DEG_PER_ELE, ida, &local_data[mn][0]);
            else
                ierr = VecGetValues(M[mn - CORRECTORS_NUM_N - 1], NODE_NUM_PER_ELE, ida_, &local_data[mn][0]);
            CHKERRQ(ierr);
        }
        C1111 += h * h * h * (lambda + 2.0 * mu) + h * h * (lambda * get_local_div_integral(&local_data[0][0]) + mu * get_local_grad_pair_integral(&local_data[0][0], 0, 0));
        C2222 += h * h * h * (lambda + 2.0 * mu) + h * h * (lambda * get_local_div_integral(&local_data[3][0]) + mu * get_local_grad_pair_integral(&local_data[3][0], 1, 1));
        C3333 += h * h * h * (lambda + 2.0 * mu) + h * h * (lambda * get_local_div_integral(&local_data[5][0]) + mu * get_local_grad_pair_integral(&local_data[5][0], 2, 2));
        C1122 += h * h * h * lambda + h * h * (lambda * get_local_div_integral(&local_data[3][0]) + mu * get_local_grad_pair_integral(&local_data[3][0], 0, 0));
        C1133 += h * h * h * lambda + h * h * (lambda * get_local_div_integral(&local_data[5][0]) + mu * get_local_grad_pair_integral(&local_data[5][0], 0, 0));
        C2233 += h * h * h * lambda + h * h * (lambda * get_local_div_integral(&local_data[5][0]) + mu * get_local_grad_pair_integral(&local_data[5][0], 1, 1));
        C1212 += h * h * h * mu + h * h * mu * get_local_grad_pair_integral(&local_data[1][0], 0, 1);
        C1313 += h * h * h * mu + h * h * mu * get_local_grad_pair_integral(&local_data[2][0], 0, 2);
        C2323 += h * h * h * mu + h * h * mu * get_local_grad_pair_integral(&local_data[4][0], 1, 2);
        C1213 += h * h * mu * get_local_grad_pair_integral(&local_data[2][0], 0, 1);
        C1223 += h * h * mu * get_local_grad_pair_integral(&local_data[4][0], 0, 1);
        C1323 += h * h * mu * get_local_grad_pair_integral(&local_data[4][0], 0, 2);
        C1112 += h * h * (lambda * get_local_div_integral(&local_data[1][0]) + mu * get_local_grad_pair_integral(&local_data[1][0], 0, 0));
        C1113 += h * h * (lambda * get_local_div_integral(&local_data[2][0]) + mu * get_local_grad_pair_integral(&local_data[2][0], 0, 0));
        C1123 += h * h * (lambda * get_local_div_integral(&local_data[4][0]) + mu * get_local_grad_pair_integral(&local_data[4][0], 0, 0));
        C2213 += h * h * (lambda * get_local_div_integral(&local_data[2][0]) + mu * get_local_grad_pair_integral(&local_data[2][0], 1, 1));
        C2223 += h * h * (lambda * get_local_div_integral(&local_data[4][0]) + mu * get_local_grad_pair_integral(&local_data[4][0], 1, 1));
        C3323 += h * h * (lambda * get_local_div_integral(&local_data[4][0]) + mu * get_local_grad_pair_integral(&local_data[4][0], 2, 2));
        B11 += h * h * h * beta + h * h * (lambda * get_local_div_integral(&local_data[6][0]) + mu * get_local_grad_pair_integral(&local_data[6][0], 0, 0));
        B22 += h * h * h * beta + h * h * (lambda * get_local_div_integral(&local_data[6][0]) + mu * get_local_grad_pair_integral(&local_data[6][0], 1, 1));
        B33 += h * h * h * beta + h * h * (lambda * get_local_div_integral(&local_data[6][0]) + mu * get_local_grad_pair_integral(&local_data[6][0], 2, 2));
        B12 += h * h * mu * get_local_grad_pair_integral(&local_data[6][0], 0, 1);
        B13 += h * h * mu * get_local_grad_pair_integral(&local_data[6][0], 0, 2);
        B23 += h * h * mu * get_local_grad_pair_integral(&local_data[6][0], 1, 2);
        K11 += h * h * h * kappa + h * h * (kappa * get_local_grad_integral(&local_data[7][0], 0));
        K22 += h * h * h * kappa + h * h * (kappa * get_local_grad_integral(&local_data[8][0], 1));
        K33 += h * h * h * kappa + h * h * (kappa * get_local_grad_integral(&local_data[9][0], 2));
        K12 += h * h * (kappa * get_local_grad_integral(&local_data[8][0], 0));
        K13 += h * h * (kappa * get_local_grad_integral(&local_data[9][0], 0));
        K23 += h * h * (kappa * get_local_grad_integral(&local_data[9][0], 1));
    }
    f = fopen("homogenized_coefficients.dat", "a");
    fprintf(f, "lambda0=%.4f\tmu0=%.4f\tbeta0=%.4f\tkappa0=%.4f\n", ctx.lambda0, ctx.mu0, ctx.beta0, ctx.kappa0);
    fprintf(f, "lambda1=%.4f\tmu1=%.4f\tbeta1=%.4f\tkappa1=%.4f\n", ctx.lambda1, ctx.mu1, ctx.beta1, ctx.kappa1);
    fprintf(f, "Use grid (%d*%d*%d), ratio=%.4f\n", ctx.ne, ctx.ne, ctx.ne, ctx.ratio);
    fprintf(f, "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n");
    fprintf(f, "C1111=%.4f\tC2222=%.4f\tC3333=%.4f\tC1122=%.4f\tC1133=%.4f\tC2233=%.4f\n", C1111, C2222, C3333, C1122, C1133, C2233);
    fprintf(f, "C1212=%.4f\tC1313=%.4f\tC2323=%.4f\tC1213=%.4f\tC1223=%.4f\tC1323=%.4f\n", C1212, C1313, C2323, C1213, C1223, C1323);
    fprintf(f, "C1112=%.4f\tC1113=%.4f\tC1123=%.4f\tC2213=%.4f\tC2223=%.4f\tC3323=%.4f\n", C1112, C1113, C1123, C2213, C2223, C3323);
    fprintf(f, "C1111-C1122-2*C1212=%.4f\n", C1111 - C1122 - 2.0 * C1212);
    fprintf(f, "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n");
    fprintf(f, "B11=%.4f\tB22=%.4f\tB33=%.4f\tB12=%.4f\tB13=%.4f\tB23=%.4f\n", B11, B22, B33, B12, B13, B23);
    fprintf(f, "K11=%.4f\tK22=%.4f\tK33=%.4f\tK12=%.4f\tK13=%.4f\tK23=%.4f\n", K11, K22, K33, K12, K13, K23);
    fprintf(f, "===============================================================================================\n\n");
    fclose(f);
    get_wall_time(wall_time);
    PetscPrintf(PETSC_COMM_SELF, "%s\tSave homogenized coefficients into file.\n", wall_time);
    // Free resource
    ierr = MatDestroy(&A);
    ierr = VecDestroyVecs(CORRECTORS_NUM_N + 1, &rhs);
    ierr = VecDestroyVecs(CORRECTORS_NUM_N, &N);
    ierr = VecDestroy(&P);
    ierr = VecDestroy(&ones);
    ierr = KSPDestroy(&ksp);
    CHKERRQ(ierr);
    ierr = MatDestroy(&A_);
    ierr = VecDestroyVecs(CORRECTORS_NUM_M, &rhs_);
    ierr = VecDestroyVecs(CORRECTORS_NUM_M, &M);
    ierr = VecDestroy(&ones_);
    ierr = KSPDestroy(&ksp_);
    CHKERRQ(ierr);
    ierr = PetscFinalize();
    return ierr;
}
