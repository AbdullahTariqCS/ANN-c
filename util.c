#include <stdio.h>
#include <stdlib.h>
#include <math.h>

double sigmoid(double x)
{
    return 1 / (1 + exp(-x));
}
double dsigmoid(double y)
{
    return y * (1 - y);
}

void print_arr(int s, double* arr)
{
    for(int i = 0 ; i < s; i++)
    {
        printf("%f ", arr[i]);
    }
    printf("\n");
}