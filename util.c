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

double relu(double x)
{
    return (x >= 0 ? x : 0);
}

double drelu(double y)
{
    return (y == 0 ? y : 1);
}

void print_arr(int s, double* arr)
{
    for(int i = 0 ; i < s; i++)
    {
        printf("%f ", arr[i]);
    }
    printf("\n");
}