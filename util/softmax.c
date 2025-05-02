#include <math.h>

double* softmax(int len, double input[len])
{
    double* O = malloc(sizeof(double) * len);
    double divider = 0;
    for (int i = 0; i < len; i++)
    {
        divider += exp(input[i]);
    }
    for(int i = 0; i < len; i++)
    {
        O[i] = exp(input[i]) / divider; 
    }
    return O;  

}