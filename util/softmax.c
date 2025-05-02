#pragma once
#include <math.h>
#include "../matrix.c"

void softmax(Matrix* O)
{
    double max_o = O->matrix[0][0];
    for (int i = 0; i < O->rows; i++)
    {
        max_o = (O->matrix[i][0] > max_o ? O->matrix[i][0] : max_o);
    }

    double divider = 0;
    for (int i = 0; i < O->rows; i++)
        divider += exp(O->matrix[i][0] - max_o);

    for(int i = 0 ; i < O->rows; i++)
        O->matrix[i][0] = exp(O->matrix[i][0] - max_o) / divider; 
}

double cross_entropy(Matrix* actual, Matrix* predicted)
{
    // Matrix* error = mat_init(actual->rows, actual->columns);
    double ce = 0;
    for(int i = 0; i < actual->rows; i++)
    {
        ce += -(actual->matrix[i][0] * log(predicted->matrix[i][0] + 1e-8));
    }
    return ce; 
}
