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

void save_matrix(const char* name, double** m, int n)
{
    FILE* f = fopen(name, "w");

    fprintf(f, "%d\n", n);

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
            fprintf(f, "%.0f ", m[i][j]);

        fprintf(f, "\n");
    }

    fclose(f);
}

void multiply_omp(double** A, double** B, double** C, int n)
{
#pragma omp parallel for
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            double sum = 0.0;

            for (int k = 0; k < n; k++)
                sum += A[i][k] * B[k][j];

            C[i][j] = sum;
        }
    }
}

void export_results(const char* name, int sizes[], double times[], int count)
{
    FILE* f = fopen(name, "w");

    fprintf(f, "| Размер | Время |\n");
    fprintf(f, "|-------|-------|\n");

    for (int i = 0; i < count; i++)
        fprintf(f, "| %d | %f |\n", sizes[i], times[i]);

    fclose(f);
}

int main()
{
    setlocale(LC_ALL, "Russian");

    srand((unsigned)time(NULL));

    int test_sizes[] = { 200,400,800,1200,1600,2000 };
    int experiments = sizeof(test_sizes) / sizeof(test_sizes[0]);

    double* times = (double*)malloc(experiments * sizeof(double));

    printf("Потоков доступно: %d\n", omp_get_max_threads());
    printf("----------------------------------------\n");

    for (int e = 0; e < experiments; e++)
    {
        int n = test_sizes[e];

        printf("\nЭксперимент %d  |  размер %d x %d\n", e + 1, n, n);

        double** A = create_matrix(n);
        double** B = create_matrix(n);
        double** C = create_matrix(n);

        printf("Создание случайных матриц...\n");

        fill_random(A, n);
        fill_random(B, n);

        char nameA[64], nameB[64], nameC[64];

        sprintf(nameA, "matrix_a_%d.txt", n);
        sprintf(nameB, "matrix_b_%d.txt", n);
        sprintf(nameC, "matrix_c_%d.txt", n);

        save_matrix(nameA, A, n);
        save_matrix(nameB, B, n);

        printf("Запуск умножения...\n");

        double t1 = omp_get_wtime();

        multiply_omp(A, B, C, n);

        double t2 = omp_get_wtime();

        double elapsed = t2 - t1;

        times[e] = elapsed;

        printf("Время: %f секунд\n", elapsed);

        save_matrix(nameC, C, n);

        delete_matrix(A, n);
        delete_matrix(B, n);
        delete_matrix(C, n);

        printf("----------------------------------------\n");
    }

    printf("\nСохранение результатов...\n");

    export_results("experiment_results.txt", test_sizes, times, experiments);

    printf("\nИтоговая таблица:\n");

    printf("+------------+----------------------+\n");
    printf("| Размер     | Время (сек)          |\n");
    printf("+------------+----------------------+\n");

    for (int i = 0; i < experiments; i++)
        printf("| %-10d | %-20f |\n", test_sizes[i], times[i]);

    printf("+------------+----------------------+\n");

    free(times);

    printf("\nЭксперимент завершён\n");

    return 0;
}
