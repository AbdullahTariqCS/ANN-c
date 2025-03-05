#include <stdio.h>
#include <stdlib.h>

typedef struct Test 
{
    int** matrix;
} Test;

int main()
{
    Test test = malloc(sizof(Test));
    return 0;
}
