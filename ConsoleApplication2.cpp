#define _CRT_SECURE_NO_WARNINGS

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <locale.h>
#include <omp.h>

double** create_matrix(int n)
{
    double** m = (double**)malloc(n * sizeof(double*));
    for (int i = 0; i < n; i++)
        m[i] = (double*)malloc(n * sizeof(double));
    return m;
}

void delete_matrix(double** m, int n)
{
    for (int i = 0; i < n; i++)
        free(m[i]);
    free(m);
}

void fill_random(double** m, int n)
{
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            m[i][j] = rand() % 10;
}

void multiply_omp(double** A, double** B, double** C, int n)
{
#pragma omp parallel for collapse(2)
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
        {
            double sum = 0.0;
            for (int k = 0; k < n; k++)
                sum += A[i][k] * B[k][j];
            C[i][j] = sum;
        }
}

int main()
{
    setlocale(LC_ALL, "Russian");
    srand((unsigned)time(NULL));

    int sizes[] = { 200,400,800,1200,1600,2000 };
    int threads_list[] = { 1,2,4,8 };

    int size_count = sizeof(sizes) / sizeof(sizes[0]);
    int thread_count = sizeof(threads_list) / sizeof(threads_list[0]);

    FILE* f = fopen("results_2.txt", "w");
    fprintf(f, "Размер\tПотоки\tВремя\tУскорение\n");

    printf("Максимум потоков: %d\n", omp_get_max_threads());

    for (int s = 0; s < size_count; s++)
    {
        int n = sizes[s];

        printf("\n===== Размер %d x %d =====\n", n, n);

        double base_time = 0.0;

        for (int t = 0; t < thread_count; t++)
        {
            int num_threads = threads_list[t];
            omp_set_num_threads(num_threads);

            double** A = create_matrix(n);
            double** B = create_matrix(n);
            double** C = create_matrix(n);

            fill_random(A, n);
            fill_random(B, n);

            double t1 = omp_get_wtime();

            multiply_omp(A, B, C, n);

            double t2 = omp_get_wtime();

            double time = t2 - t1;

            if (num_threads == 1)
                base_time = time;

            double speedup = base_time / time;

            printf("Потоки: %d | Время: %f | Ускорение: %f\n",
                num_threads, time, speedup);

            fprintf(f, "%d\t%d\t%f\t%f\n",
                n, num_threads, time, speedup);

            delete_matrix(A, n);
            delete_matrix(B, n);
            delete_matrix(C, n);
        }
    }

    fclose(f);

    printf("\nЭксперименты завершены\n");
    return 0;
}
