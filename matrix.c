#pragma once
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

typedef struct Matrix{
    int rows; 
    int columns; 
    double** matrix; 
} Matrix;

Matrix* mat_init(int rows, int columns)
{

    Matrix *A = malloc(sizeof(Matrix));
    A->rows = rows; 
    A->columns = columns; 
    A->matrix = malloc(rows * sizeof(double*));
    for(int i = 0 ; i < rows; i++)
    {
        A->matrix[i] = malloc(columns * sizeof(double));
    }
    return A; 
}

void mat_free(Matrix* A, int only_matrix)
{
    for(int i = 0; i < A->rows; i++)
    {
        free(A->matrix[i]);
    }
    free(A->matrix);
    if (!only_matrix)
        free(A);
}

Matrix* mat_copy(Matrix* A)
{
    Matrix* O = mat_init(A->rows, A->columns); 
    for(int i = 0; i < A->rows; i++)
        for(int j=0; j < A->columns; j++)
            O->matrix[i][j] = A->matrix[i][j];
     
    return O;
}

void mat_print(Matrix* A)
{
    for(int i = 0;  i<A->rows; i++ )
    {
        for(int j = 0; j < A->columns; j++)
            printf("%f ", A->matrix[i][j]);
        printf("\n");
    }
}

void mat_write(Matrix* A, FILE* file)
{

    fprintf(file, "%d;%d\n", A->rows, A->columns);
    for(int i = 0;  i<A->rows; i++ )
    {
        for(int j = 0; j < A->columns; j++)
            fprintf(file, "%f;", A->matrix[i][j]);
        fprintf(file, "\n");
    }
}

Matrix* mat_read(FILE* file)
{
    int rows, columns; 
    fscanf(file, "%d;%d\n", &rows, &columns);
    Matrix* A = mat_init(rows, columns);
    for(int i = 0 ; i < rows; i++)
    {
        for(int j = 0; j < columns; j++)
        {
            fscanf(file, "%lf;", &A->matrix[i][j]);
        }
        fscanf(file, "\n");
    }
    return A;
}

Matrix* mat_mul(Matrix* A, Matrix* B)
{
    if (A->columns != B->rows)
    {
        printf("Cannot Multiply (%d, %d) with (%d, %d)\n", A->rows, A->columns, B->rows, B->columns);         

        return NULL; 
    }
    Matrix* O = mat_init(A->rows, B->columns);

    for(int i = 0; i < A->rows; i++)
    {
        for(int j = 0; j < B->columns; j++)
        {
            O->matrix[i][j] = 0;
            for(int k=0; k<A->columns; k++)
                O->matrix[i][j] += A->matrix[i][k] * B->matrix[k][j];
        }
    }
    return O;
}

Matrix* mat_scalar_mul(Matrix* A, Matrix* B, int in_place)
{
    if (A->rows != B->rows || A->columns != B->columns)
    {
        printf("Cannot scalar multiply (%d, %d) with (%d, %d)\n", A->rows, A->columns, B->rows, B->columns);
        return NULL;
    }

    Matrix* O; 
    if (in_place) O = A; 
    else O = mat_init(A->rows, A->columns);
    
    for(int i = 0; i < A->rows; i++)
        for(int j = 0; j<A->columns; j++)
            O->matrix[i][j] = A->matrix[i][j] * B->matrix[i][j];
            
    return O;
    
}

Matrix* mat_add(Matrix* A, Matrix* B, int in_place){
    if (A->rows != B->rows || A->columns != B->columns)
    {
        printf("Cannot add (%d, %d) with (%d, %d)\n", A->rows, A->columns, B->rows, B->columns);
        return NULL;
    }
    Matrix* O; 
    if (in_place) O = A; 
    else O = mat_init(A->rows, A->columns);

    for(int i = 0; i < A->rows; i++)
        for(int j = 0; j<A->columns; j++)
            O->matrix[i][j] = A->matrix[i][j] + B->matrix[i][j];
            
    return O;
}

Matrix* mat_map(Matrix* A, double f(double), int in_place)
{
    Matrix* O = NULL; 
    if (in_place == 0) O = mat_init(A->rows, A->columns);
    for(int i = 0; i < A->rows; i++)
       for(int j = 0; j < A->columns; j++) 
            if(in_place == 1) 
                A->matrix[i][j] = f(A->matrix[i][j]);
            else 
                O->matrix[i][j] = f(A->matrix[i][j]);

    return O;
}

Matrix* mat_transpose(Matrix* A, int in_place)
{
    Matrix* transpose = mat_init(A->columns, A->rows);


    for(int i = 0; i < transpose->rows; i++)
        for(int j = 0; j < transpose->columns; j++)
            transpose->matrix[i][j] = A->matrix[j][i];
    
    if (in_place)
    {
        mat_free(A, 1);
        A->matrix = transpose->matrix; 
        return NULL;
    }
    return transpose;
}

Matrix* arr_to_mat(int len, double arr[len], int column)
{
    Matrix* A = mat_init(len, 1);

    for(int i = 0; i < len; i++)
        A->matrix[i][0] = arr[i];

    if (!column)
        mat_transpose(A, 1);
    
    return A;
} 

double* mat_to_arr(Matrix* A)
{
    double* arr = malloc(A->rows * A->columns * sizeof(double));
    int k = 0;
    for(int i = 0; i < A->rows; i++)
        for(int j = 0; j < A->columns; j++)
            arr[k++] = A->matrix[i][j];

    return arr;
}


// void mat_rand(Matrix* A)
// {
//     double *normal_data = malloc(A->rows * A->columns * sizeof(double));
//     const double PI = 3.14;
//     for(int i = 0; i < A->rows; i++)
//         for(int j = 0; j < A->columns; j++)
//             A->matrix[i][j] = -1 + rand() % (2);
// }

void mat_zeros(Matrix* A)
{
    double *normal_data = malloc(A->rows * A->columns * sizeof(double));
    for(int i = 0; i < A->rows; i++)
        for(int j = 0; j < A->columns; j++)
            A->matrix[i][j] = 0;
}

void mat_rand(Matrix* A) {
    double stddev = sqrt(2.0 / (A->columns));  // He initialization for ReLU
    for (int i = 0; i < A->rows; i++) {
        for (int j = 0; j < A->columns; j++) {
            double u1 = (rand() + 1.0) / (RAND_MAX + 1.0);
            double u2 = (rand() + 1.0) / (RAND_MAX + 1.0);
            double rand_normal = sqrt(-2 * log(u1)) * cos(2 * 3.14159 * u2);
            A->matrix[i][j] = rand_normal * stddev;
        }
    }
}

void mat_rand_xavier(Matrix* A) {
    if (A->columns == 0 || A->rows == 0) {  // Safety check
        fprintf(stderr, "Invalid matrix dimensions for initialization\n");
        exit(EXIT_FAILURE);
    }

    double fan_in = A->columns;
    double fan_out = A->rows;
    double stddev = sqrt(2.0 / (fan_in + fan_out));  // Xavier/Glorot initialization

    for (int i = 0; i < A->rows; i++) {
        for (int j = 0; j < A->columns; j++) {
            // Box-Muller transform for normal distribution
            double u1 = (rand() + 1.0) / (RAND_MAX + 1.0);  // Avoid log(0)
            double u2 = (rand() + 1.0) / (RAND_MAX + 1.0);
            double rand_normal = sqrt(-2.0 * log(u1)) * cos(2.0 * 3.14159 * u2);
            
            A->matrix[i][j] = rand_normal * stddev;
        }
    }
}

Matrix* mat_sum(Matrix* A, Matrix* B, int in_place)
{
    Matrix* O; 
    if (in_place) O = A; 
    else O = mat_init(A->rows, A->columns);

    for(int i = 0; i < A->rows; i++)
        for(int j = 0; j<A->columns; j++)
            O->matrix[i][j] = A->matrix[i][j] + B->matrix[i][j]; 
    
    return O;
}

Matrix* mat_scale(Matrix* A, double n, int in_place)
{
    Matrix* O = mat_init(A->rows, A->columns);
    for(int i = 0;  i<A->rows; i++ )
        for(int j = 0; j < A->columns; j++)
            if (in_place) A->matrix[i][j] *= n; 
            else O->matrix[i][j] = A->matrix[i][j] * n; 

    if(in_place)
    {
        free(O);
        return NULL;
    }

    return O;
}

Matrix* mat_add_scalar(Matrix* A, double n, int in_place)
{
    Matrix* O = mat_init(A->rows, A->columns);
    for(int i = 0;  i<A->rows; i++ )
        for(int j = 0; j < A->columns; j++)
            if(in_place) A->matrix[i][j] += n;
            else O->matrix[i][j] = A->matrix[i][j] + n; 
    if (in_place)
    {
        free(O);
        return NULL;
    }
    return O;

}
void mat_normalize(Matrix* A)
{
    int N = A->rows * A->columns; 
    double mean = 0; 
    double variance = 0;
    for(int i = 0; i < A->rows; i++)
        for(int j = 0; j < A->columns; j++)
            mean += A->matrix[i][j];
    mean /= N; 

    for(int i = 0; i < A->rows; i++)
        for(int j = 0; j < A->columns; j++)
            variance += pow(A->matrix[i][j] - mean, 2);
    variance = sqrt(variance/N);
    

    mat_add_scalar(A, -mean, 1);
    if (variance != 0)
        mat_scale(A, 1/variance, 1);
}

