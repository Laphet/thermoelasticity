static char help[] = "gogo children!\n\n";
static char wall_time[26];

#include "petscksp.h"
#include "petscviewerhdf5.h"
#include "stdio.h"
#include "math.h"
#include "time.h"
#include "string.h"

#define DIM 3
#define FREEDOM_DEG_PER_NODE 3
#define FREEDOM_DEG_PER_ELE 24
#define CORRECTORS_NUM 6
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
    double lambda0, lambda1, mu0, mu1;
    double ratio;
    double tol;
    int ne;
} Context;

static double stiffA[FREEDOM_DEG_PER_ELE * FREEDOM_DEG_PER_ELE];
static double stiffB[FREEDOM_DEG_PER_ELE * FREEDOM_DEG_PER_ELE];
static double stiffC[FREEDOM_DEG_PER_ELE * FREEDOM_DEG_PER_ELE];
static double load[FREEDOM_DEG_PER_ELE];
static char name[CORRECTORS_NUM][MAX_LENGTH_FILE_NAME] = {"N11", "N12", "N13", "N22", "N23", "N33"};

double
get_lambda(int ele_ind_x, int ele_ind_y, int ele_ind_z, Context *ctx)
{
    int width = round(ctx->ne * ctx->ratio);
    if ((width <= ele_ind_x) && (ele_ind_x < ctx->ne - width) && (width <= ele_ind_y) && (ele_ind_y < ctx->ne - width) && (width <= ele_ind_z) && (ele_ind_z < ctx->ne - width))
        return ctx->lambda0;
    else
        return ctx->lambda1;
}

double
get_mu(int ele_ind_x, int ele_ind_y, int ele_ind_z, Context *ctx)
{
    int width = round(ctx->ne * ctx->ratio);
    if ((width <= ele_ind_x) && (ele_ind_x < ctx->ne - width) && (width <= ele_ind_y) && (ele_ind_y < ctx->ne - width) && (width <= ele_ind_z) && (ele_ind_z < ctx->ne - width))
        return ctx->mu0;
    else
        return ctx->mu1;
}

int get_freedom_index(int idx, int idy, int idz, int i, Context *ctx)
{
    idx = idx % ctx->ne;
    idy = idy % ctx->ne;
    idz = idz % ctx->ne;
    return FREEDOM_DEG_PER_NODE * (idx + ctx->ne * idy + ctx->ne * ctx->ne * idz) + i;
}

void decode_local_index(int a, int *ax, int *ay, int *az, int *i)
// a = 3(ax+2ay+4az)+i
{
    *i = a % FREEDOM_DEG_PER_NODE;
    *az = (a / FREEDOM_DEG_PER_NODE) / 4;
    *ay = ((a / FREEDOM_DEG_PER_NODE)) / 2 % 2;
    *ax = (a / FREEDOM_DEG_PER_NODE) % 2;
}

double
get_local_grad_pair_integral(double *local_data, int i, int j)
// Calculate int_hat(T) partial N_i/partial y_j+partial N_j/partial y_i
{
    double ans = 0.0;
    int k;
    for (k = 0; k < NODE_NUM_PER_ELE; ++k)
    {
        ans += local_data[FREEDOM_DEG_PER_NODE * k + i] * load[FREEDOM_DEG_PER_NODE * k + j];
        ans += local_data[FREEDOM_DEG_PER_NODE * k + j] * load[FREEDOM_DEG_PER_NODE * k + i];
    }
    return ans;
}

double
get_local_div_integral(double *local_data)
{
    double ans = 0.0;
    int k;
    for (k = 0; k < FREEDOM_DEG_PER_ELE; ++k)
        ans += local_data[k] * load[k];
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
    Context ctx = {.lambda0 = 1.0, .lambda1 = 2.0, .mu0 = 3.0, .mu1 = 4.0, .ratio = 0.25, .ne = 16, .tol = 1.0e-7};
    Mat A;
    Vec ones, *b, *N;
    KSP ksp;
    int total_element_num = ctx.ne * ctx.ne * ctx.ne;
    int total_freedom_deg = total_element_num * FREEDOM_DEG_PER_NODE;
    PetscErrorCode ierr;
    ierr = PetscInitialize(&argc, &args, (char *)0, help);
    if (ierr)
        return ierr;
    ierr = PetscOptionsGetInt(NULL, NULL, "-ne", &(ctx.ne), NULL);
    CHKERRQ(ierr);
    get_wall_time(wall_time);
    PetscPrintf(PETSC_COMM_SELF, "%s\tSTART!\tuse grid (%d*%d*%d).\n", wall_time, ctx.ne, ctx.ne, ctx.ne);
    // Create PETSc vector and matrix objects, here we use sequencial form.
    ierr = VecCreateSeq(PETSC_COMM_SELF, total_freedom_deg, &ones);
    CHKERRQ(ierr);
    ierr = VecSet(ones, 1.0);
    CHKERRQ(ierr);
    ierr = VecDuplicateVecs(ones, CORRECTORS_NUM, &b);
    ierr = VecDuplicateVecs(ones, CORRECTORS_NUM, &N);
    ierr = PetscObjectSetName((PetscObject)(N[0]), "N11");
    ierr = PetscObjectSetName((PetscObject)(N[1]), "N12");
    ierr = PetscObjectSetName((PetscObject)(N[2]), "N13");
    ierr = PetscObjectSetName((PetscObject)(N[3]), "N22");
    ierr = PetscObjectSetName((PetscObject)(N[4]), "N23");
    ierr = PetscObjectSetName((PetscObject)(N[5]), "N33");
    CHKERRQ(ierr);
    ierr = MatCreateSeqAIJ(PETSC_COMM_SELF, total_freedom_deg, total_freedom_deg, 3 * 3 * 3 * FREEDOM_DEG_PER_NODE, NULL, &A); // Each freedom degree intersects with 27*3 freedom degrees.
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
    double data_addto_b[CORRECTORS_NUM][FREEDOM_DEG_PER_ELE];
    double h = 1.0 / (double)ctx.ne, lambda, mu;
    int indexa[FREEDOM_DEG_PER_ELE];
    int indexb[FREEDOM_DEG_PER_ELE];
    for (ele_ind = 0; ele_ind < total_element_num; ++ele_ind)
    {
        ele_ind_x = ele_ind % ctx.ne;
        ele_ind_y = (ele_ind / ctx.ne) % ctx.ne;
        ele_ind_z = ele_ind / (ctx.ne * ctx.ne);
        lambda = get_lambda(ele_ind_x, ele_ind_y, ele_ind_z, &ctx);
        mu = get_mu(ele_ind_x, ele_ind_y, ele_ind_z, &ctx);
        for (ai = 0; ai < FREEDOM_DEG_PER_ELE; ++ai)
        {
            decode_local_index(ai, &aix, &aiy, &aiz, &aik);
            indexa[ai] = get_freedom_index(ele_ind_x + aix, ele_ind_y + aiy, ele_ind_z + aiz, aik, &ctx);
            for (bj = 0; bj < FREEDOM_DEG_PER_ELE; ++bj)
            {
                decode_local_index(bj, &bjx, &bjy, &bjz, &bjk);
                indexb[bj] = get_freedom_index(ele_ind_x + bjx, ele_ind_y + bjy, ele_ind_z + bjz, bjk, &ctx);
                data_addto_A[ai * FREEDOM_DEG_PER_ELE + bj] = h * (lambda * stiffA[ai * FREEDOM_DEG_PER_ELE + bj] + mu * (stiffB[ai * FREEDOM_DEG_PER_ELE + bj] + stiffC[ai * FREEDOM_DEG_PER_ELE + bj]));
            }
            data_addto_b[0][ai] = -h * h * (lambda * load[ai] + mu * (2.0 * (double)(aik == 0) * load[ai - aik + 0]));               // m=1, n=1
            data_addto_b[3][ai] = -h * h * (lambda * load[ai] + mu * (2.0 * (double)(aik == 1) * load[ai - aik + 1]));               // m=2, n=2
            data_addto_b[5][ai] = -h * h * (lambda * load[ai] + mu * (2.0 * (double)(aik == 2) * load[ai - aik + 2]));               // m=3, n=3
            data_addto_b[1][ai] = -h * h * mu * ((double)(aik == 0) * load[ai - aik + 1] + (double)(aik == 1) * load[ai - aik + 0]); // m=1, n=2
            data_addto_b[2][ai] = -h * h * mu * ((double)(aik == 0) * load[ai - aik + 2] + (double)(aik == 2) * load[ai - aik + 0]); // m=1, n=3
            data_addto_b[4][ai] = -h * h * mu * ((double)(aik == 1) * load[ai - aik + 2] + (double)(aik == 2) * load[ai - aik + 1]); // m=2, n=3
        }
        ierr = MatSetValues(A, FREEDOM_DEG_PER_ELE, indexa, FREEDOM_DEG_PER_ELE, indexb, data_addto_A, ADD_VALUES);
        CHKERRQ(ierr);
        for (mn = 0; mn < CORRECTORS_NUM; ++mn)
        {
            ierr = VecSetValues(b[mn], FREEDOM_DEG_PER_ELE, indexa, &data_addto_b[mn][0], ADD_VALUES);
            CHKERRQ(ierr);
        }
    }
    ierr = MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
    CHKERRQ(ierr);
    ierr = MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);
    CHKERRQ(ierr);
    for (mn = 0; mn < CORRECTORS_NUM; ++mn)
    {
        ierr = VecAssemblyBegin(b[mn]);
        CHKERRQ(ierr);
        ierr = VecAssemblyEnd(b[mn]);
        CHKERRQ(ierr);
    }
    // Need to tell the solver that the constant solution should not be included.
    MatNullSpace mns;
    double norm2;
    VecNorm(ones, NORM_2, &norm2);
    VecScale(ones, 1.0 / norm2);
    ierr = MatNullSpaceCreate(PETSC_COMM_SELF, PETSC_FALSE, 1, &ones, &mns);
    CHKERRQ(ierr);
    ierr = MatSetNullSpace(A, mns);
    ierr = MatSetTransposeNullSpace(A, mns);
    ierr = MatNullSpaceDestroy(&mns);
    CHKERRQ(ierr);
    get_wall_time(wall_time);
    PetscPrintf(PETSC_COMM_SELF, "%s\tFinish constructing linear system.\n", wall_time);
    // Build solver
    ierr = KSPCreate(PETSC_COMM_SELF, &ksp);
    CHKERRQ(ierr);
    ierr = KSPSetOperators(ksp, A, A);
    ierr = KSPSetFromOptions(ksp);
    CHKERRQ(ierr);
    // Solving process begins
    int iter_count;
    for (mn = 0; mn < CORRECTORS_NUM; ++mn)
    {
        ierr = KSPSolve(ksp, b[mn], N[mn]);
        CHKERRQ(ierr);
        ierr = KSPGetIterationNumber(ksp, &iter_count);
        CHKERRQ(ierr);
        get_wall_time(wall_time);
        PetscPrintf(PETSC_COMM_SELF, "%s\t%s has been solved, iteration num=%d.\n", wall_time, name[mn], iter_count);
    }
    // Saving correctors
    PetscViewer fw;
    char filename[MAX_LENGTH_FILE_NAME];
    for (mn = 0; mn < CORRECTORS_NUM; ++mn)
    {
        snprintf(filename, MAX_LENGTH_FILE_NAME, "%s-ne%d.hdf5", name[mn], ctx.ne);
        ierr = PetscViewerHDF5Open(PETSC_COMM_SELF, filename, FILE_MODE_WRITE, &fw);
        ierr = VecView(N[mn], fw);
        CHKERRQ(ierr);
        get_wall_time(wall_time);
        PetscPrintf(PETSC_COMM_SELF, "%s\t%s has been saved.\n", wall_time, name[mn]);
    }
    ierr = PetscViewerDestroy(&fw);
    CHKERRQ(ierr);
    // Calculate homogenized coefficients
    get_wall_time(wall_time);
    PetscPrintf(PETSC_COMM_SELF, "%s\tBegin to calculate homogenized coefficients.\n", wall_time);
    double C1111 = 0.0, C2222 = 0.0, C3333 = 0.0, C1122 = 0.0, C1133 = 0.0, C2233 = 0.0;
    double C1212 = 0.0, C1313 = 0.0, C2323 = 0.0, C1213 = 0.0, C1223 = 0.0, C1323 = 0.0;
    double C1112 = 0.0, C1113 = 0.0, C1123 = 0.0, C2213 = 0.0, C2223 = 0.0, C3323 = 0.0;
    double local_data[CORRECTORS_NUM][FREEDOM_DEG_PER_ELE];
    for (ele_ind = 0; ele_ind < total_element_num; ++ele_ind)
    {
        ele_ind_x = ele_ind % ctx.ne;
        ele_ind_y = (ele_ind / ctx.ne) % ctx.ne;
        ele_ind_z = ele_ind / (ctx.ne * ctx.ne);
        lambda = get_lambda(ele_ind_x, ele_ind_y, ele_ind_z, &ctx);
        mu = get_mu(ele_ind_x, ele_ind_y, ele_ind_z, &ctx);
        for (ai = 0; ai < FREEDOM_DEG_PER_ELE; ++ai)
        {
            decode_local_index(ai, &aix, &aiy, &aiz, &aik);
            indexa[ai] = get_freedom_index(ele_ind_x + aix, ele_ind_y + aiy, ele_ind_z + aiz, aik, &ctx);
        }
        for (mn = 0; mn < CORRECTORS_NUM; ++mn)
        {
            ierr = VecGetValues(N[mn], FREEDOM_DEG_PER_ELE, indexa, &local_data[mn][0]);
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
    }
    f = fopen("homogenized_coefficients.dat", "w");
    fprintf(f, "lambda0=%.4f\tmu0=%.4f\n", ctx.lambda0, ctx.mu0);
    fprintf(f, "lambda1=%.4f\tmu1=%.4f\n", ctx.lambda1, ctx.mu1);
    fprintf(f, "Use grid (%d*%d*%d)\n", ctx.ne, ctx.ne, ctx.ne);
    fprintf(f, "===============================================================================================\n");
    fprintf(f, "C1111=%.4f\tC2222=%.4f\tC3333=%.4f\tC1122=%.4f\tC1133=%.4f\tC2233=%.4f\n", C1111, C2222, C3333, C1122, C1133, C2233);
    fprintf(f, "C1212=%.4f\tC1313=%.4f\tC2323=%.4f\tC1213=%.4f\tC1223=%.4f\tC1323=%.4f\n", C1212, C1313, C2323, C1213, C1223, C1323);
    fprintf(f, "C1112=%.4f\tC1113=%.4f\tC1123=%.4f\tC2213=%.4f\tC2223=%.4f\tC3323=%.4f\n", C1112, C1113, C1123, C2213, C2223, C3323);
    fprintf(f, "C1111-C1122-2*C1212=%.4f\n", C1111 - C1122 - 2.0 * C1212);
    fprintf(f, "===============================================================================================\n");
    fclose(f);
    get_wall_time(wall_time);
    PetscPrintf(PETSC_COMM_SELF, "%s\tSave homogenized coefficients into file.\n", wall_time);
    // Free resource
    ierr = MatDestroy(&A);
    ierr = VecDestroyVecs(CORRECTORS_NUM, &N);
    ierr = VecDestroyVecs(CORRECTORS_NUM, &b);
    ierr = VecDestroy(&ones);
    ierr = KSPDestroy(&ksp);
    CHKERRQ(ierr);
    ierr = PetscFinalize();
    return ierr;
}
