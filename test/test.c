#include <stdio.h>
#include <stdlib.h>
#include "../util/pgm.c"

int main()
{
    int width = 24, height = 24; 
    unsigned char* image = NULL; 
    char filename[] = "../dataset/48/0.pgm";
    readPGM(filename, &image, &width, &height); 

    for (int i = 0; i < width * height; i++)
    {
        // printf("hello world");
        printf("%d ", image[i]);
    }
    return EXIT_SUCCESS;
}
