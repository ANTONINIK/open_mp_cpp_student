#include <iostream>
#include <omp.h>
#include <iomanip>

using namespace std;

const int N = 50000000;
const int N1 = N / 2;

void init_arr(int* array) {
    int i = 0;
#pragma omp parallel shared(array) private(i) 
    {
#pragma omp for
        for (i = 0; i < N; ++i) {
            array[i] = rand() % 10;
        }
    }
}

double variant2_single(int* A, int* B) {
    double sum = 0;
    {
        for (int i = 0; i < N; ++i) {
            double m = max((A[i] + B[i]), 4 * A[i] - B[i]);
            if (m > 1) sum += m;
        }
    }
    return sum;
}

double variant2_reduction_multi(int* A, int* B) {
    double sum = 0;
    int i = 0;
#pragma omp parallel shared(A,B) private(i) 
    {
#pragma omp for reduction(+:sum)
        for (i = 0; i < N; ++i) {
            double m = max((A[i] + B[i]), 4 * A[i] - B[i]);
            if (m > 1) sum += m;
        }
    }
    return sum;
}

double variant2_atomic(int* A, int* B) {
    double sum = 0;
    int i = 0;
#pragma omp parallel shared(sum,A,B) private(i) 
    {
#pragma omp for
        for (i = 0; i < N; ++i) {
            double m = max((A[i] + B[i]), 4 * A[i] - B[i]);
            if (m > 1)
            {
#pragma omp atomic
                sum += m;
            }
        }
    }
    return sum;
}

double variant2_critical(int* A, int* B) {
    double sum = 0;
    int i = 0;
#pragma omp parallel shared(sum,A,B) private(i) 
    {
#pragma omp for
        for (i = 0; i < N; ++i) {
            double m = max((A[i] + B[i]), 4 * A[i] - B[i]);
            if (m > 1)
            {
#pragma omp critical
                sum += m;
            }
        }
    }
    return sum;
}

double variant2_sections_N2(int* A, int* B) {
    int i = 0;
    double sum = 0;
#pragma omp parallel sections shared(A,B) private(i) reduction(+:sum)
    {
#pragma omp section
        {
            {
                for (int i = 0; i < N / 2; ++i) {
                    double m = max((A[i] + B[i]), 4 * A[i] - B[i]);
                    if (m > 1) sum += m;
                }
            }
        }
#pragma omp section
        {
            {
                for (int i = N / 2; i < N; ++i) {
                    double m = max((A[i] + B[i]), 4 * A[i] - B[i]);
                    if (m > 1) sum += m;
                }
            }
        }
    }
    return sum;
}

double variant2_sections_N4(int* A, int* B) {
    int i = 0;
    double sum = 0;
#pragma omp parallel sections shared(A,B) private(i) reduction(+:sum)
    {
#pragma omp section
        {
            {
                for (int i = 0; i < N / 4; ++i) {
                    double m = max((A[i] + B[i]), 4 * A[i] - B[i]);
                    if (m > 1) sum += m;
                }
            }
        }
#pragma omp section
        {
            {
                for (int i = N / 4; i < N / 2; ++i) {
                    double m = max((A[i] + B[i]), 4 * A[i] - B[i]);
                    if (m > 1) sum += m;
                }
            }
        }
#pragma omp section
        {
            {
                for (int i = N / 2; i < 3 * N / 4; ++i) {
                    double m = max((A[i] + B[i]), 4 * A[i] - B[i]);
                    if (m > 1) sum += m;
                }
            }
        }
#pragma omp section
        {
            {
                for (int i = 3 * N / 4; i < N; ++i) {
                    double m = max((A[i] + B[i]), 4 * A[i] - B[i]);
                    if (m > 1) sum += m;
                }
            }
        }
    }
    return sum;
}

int main()
{
    double start = 0;
    double end = 0;
    double sum = 0;
    int* array1 = new int[N];
    int* array2 = new int[N];

    init_arr(array1);
    init_arr(array2);

    start = omp_get_wtime();
    sum = variant2_single(array1, array2);
    end = omp_get_wtime();
    cout << "Single SUM: " << sum << endl;
    cout << "Single TIME: " << end - start << "\n" << endl;

    start = omp_get_wtime();
    sum = variant2_reduction_multi(array1, array2);
    end = omp_get_wtime();
    cout << "Reduction SUM: " << sum << endl;
    cout << "Reduction TIME: " << end - start << "\n" << endl;

    start = omp_get_wtime();
    sum = variant2_atomic(array1, array2);
    end = omp_get_wtime();
    cout << "Atomic SUM: " << sum << endl;
    cout << "Atomic TIME: " << end - start << "\n" << endl;

    start = omp_get_wtime();
    sum = variant2_critical(array1, array2);
    end = omp_get_wtime();
    cout << "Critical SUM: " << sum << endl;
    cout << "Critical TIME: " << end - start << "\n" << endl;

    start = omp_get_wtime();
    sum = variant2_sections_N2(array1, array2);
    end = omp_get_wtime();
    cout << "2 sections SUM: " << sum << endl;
    cout << "2 sections TIME: " << end - start << "\n" << endl;

    start = omp_get_wtime();
    sum = variant2_sections_N4(array1, array2);
    end = omp_get_wtime();
    cout << "4 sections SUM: " << sum << endl;
    cout << "4 sections TIME: " << end - start << "\n" << endl;


    delete[] array1;
    delete[] array2;

    return 0;
}