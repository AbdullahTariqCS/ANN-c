#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

double relu(double x)
{
    return (x >= 0 ? x : 0);
}

double drelu(double y)
{
    return (y == 0 ? y : 1);
}
