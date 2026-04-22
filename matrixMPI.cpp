#define _CRT_SECURE_NO_WARNINGS

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

double* create_matrix(int n)
{
    return (double*)malloc(n * n * sizeof(double));
}

void fill_random(double* m, int n)
{
    for (int i = 0; i < n * n; i++)\
        m[i] = rand() % 10;
}

void zero_matrix(double* m, int size)
{
    for (int i = 0; i < size; i++)
        m[i] = 0.0;
}

// 🔥 БЛОЧНОЕ УМНОЖЕНИЕ
void multiply_blocked(double* A, double* B, double* C, int n, int rows)
{
    int BS = 64;

    for (int ii = 0; ii < rows; ii += BS)
        for (int jj = 0; jj < n; jj += BS)
            for (int kk = 0; kk < n; kk += BS)

                for (int i = ii; i < ii + BS && i < rows; i++)
                    for (int j = jj; j < jj + BS && j < n; j++)
                    {
                        double sum = C[i * n + j];

                        for (int k = kk; k < kk + BS && k < n; k++)
                            sum += A[i * n + k] * B[k * n + j];

                        C[i * n + j] = sum;
                    }
}

// 📄 запись результатов
void append_results(const char* filename, int n, int proc, double time)
{
    FILE* f = fopen(filename, "a");

    fprintf(f, "%d\t%d\t%f\n", n, proc, time);

    fclose(f);
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int test_sizes[] = { 200,400,800,1200,1600,2000 };
    int experiments = 6;


    if (rank == 0)
    {
        FILE* f = fopen("results.txt", "w");
        fprintf(f, "Size\tProcesses\tTime\n");
        fclose(f);
    }

    for (int e = 0; e < experiments; e++)
    {
        int n = test_sizes[e];

        double* A = NULL;
        double* B = (double*)malloc(n * n * sizeof(double));
        double* C = NULL;

        if (rank == 0)
        {
            A = create_matrix(n);
            C = create_matrix(n);

            srand(time(NULL));
            fill_random(A, n);
            fill_random(B, n);
        }

        // 🔥 Scatterv подготовка
        int* sendcounts = (int*)malloc(size * sizeof(int));
        int* displs = (int*)malloc(size * sizeof(int));

        int base = n / size;
        int rem = n % size;

        int offset = 0;

        for (int i = 0; i < size; i++)
        {
            int rows_i = base + (i < rem ? 1 : 0);
            sendcounts[i] = rows_i * n;
            displs[i] = offset;
            offset += rows_i * n;
        }

        int local_size = sendcounts[rank];
        int local_rows = local_size / n;

        double* localA = (double*)malloc(local_size * sizeof(double));
        double* localC = (double*)malloc(local_size * sizeof(double));

        zero_matrix(localC, local_size);

        // 📡 передача данных
        MPI_Bcast(B, n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        MPI_Scatterv(A, sendcounts, displs, MPI_DOUBLE,
            localA, local_size, MPI_DOUBLE,
            0, MPI_COMM_WORLD);

        MPI_Barrier(MPI_COMM_WORLD);
        double t1 = MPI_Wtime();

        multiply_blocked(localA, B, localC, n, local_rows);

        MPI_Barrier(MPI_COMM_WORLD);
        double t2 = MPI_Wtime();

        MPI_Gatherv(localC, local_size, MPI_DOUBLE,
            C, sendcounts, displs, MPI_DOUBLE,
            0, MPI_COMM_WORLD);

        if (rank == 0)
        {
            double time = t2 - t1;

            printf("N=%d | P=%d | Time=%f \n",
                n, size, time);

            append_results("results.txt", n, size, time);
        }

        free(localA);
        free(localC);
        free(B);
        free(sendcounts);
        free(displs);

        if (rank == 0)
        {
            free(A);
            free(C);
        }
    }

    MPI_Finalize();
    return 0;
}