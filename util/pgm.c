#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

void readPGM(char* filename, unsigned char** image, int* width, int* height) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        fprintf(stderr, "Cannot open file: %s\n", filename);
        exit(EXIT_FAILURE);
    }

    // printf("file opened\n");
    char magic[3];
    if (fscanf(file, "%2s", magic) != 1 || strcmp(magic, "P5") != 0) {
        fprintf(stderr, "Not a valid PGM (P5) file\n");
        fclose(file);
        exit(EXIT_FAILURE);
    }

    // printf("File verified\n");
    if (fscanf(file, "%d %d", width, height) != 2) {
        fprintf(stderr, "Invalid image dimensions\n");
        fclose(file);
        exit(EXIT_FAILURE);
    }

    // printf("Dimensions: (%d, %d)\n", *width, *height);

    int max_val;
    if (fscanf(file, "%d", &max_val) != 1) {
        fprintf(stderr, "Invalid max value\n");
        fclose(file);
        exit(EXIT_FAILURE);
    }

    // printf("max value: %d\n", max_val);
    if (fgetc(file) == EOF)
    {
        fprintf(stderr, "Error reading file");
        fclose(file);
        exit(EXIT_FAILURE);
    }
    // printf("Got the next character\n");

    *image = (unsigned char*)malloc((*width) * (*height) * sizeof(unsigned char));
    // printf("%p\n", *image);
    if (!*image)
    {
        fprintf(stderr, "Error Allocation image pointer");
        fclose(file);
        exit(EXIT_FAILURE);
    }

    // printf("Image allocated\n");
    
    if (fread(*image, sizeof(unsigned char), (*width) * (*height), file) != (*width) * (*height)) {
        fprintf(stderr, "Error reading image data\n");
        fclose(file);
        exit(EXIT_FAILURE);
    }
    fclose(file);
    // printf("Image Transferred\n");
}