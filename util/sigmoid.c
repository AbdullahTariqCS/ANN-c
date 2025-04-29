#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

double sigmoid(double x)
{
    return 1 / (1 + exp(-x));
}
double dsigmoid(double y)
{
    return y * (1 - y);
}
