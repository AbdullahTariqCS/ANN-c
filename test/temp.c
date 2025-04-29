#include <stdio.h>

int main()
{
    int a = 1, b = 2, c; 
    c = (a = b + 2) + (b = b + 1);
    printf("%d, %d, %d\n", a, b , c);
    return 0;
}