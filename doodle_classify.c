#include "./util/dir.c"
#include "model_softmax.c"
#include "util/pgm.c"
#include "util/relu.c"
#include "util/sigmoid.c"
#include <string.h>
#include <stdlib.h>
#include <unistd.h>

char PATH_TO_DATASET[] = "./dataset/";

int main(int argc, char *argv[])
{
    Model *model = read_model("doodle_classifier.txt");

    // getting the names of dataset
    int *dirs[10];
    int dirCount[10];
    for (int i = 0; i < 10; i++)
    {
        char class_name[24];
        sprintf(class_name, "%s%d", PATH_TO_DATASET, i + 48);
        dirCount[i] = -1;
        get_dir(&dirs[i], &dirCount[i], class_name);
        printf("Got directory %s (%d)\n", class_name, dirCount[i]);
    }


    printf("Forward Pass\n");
    for (int i = 0; i < 10; i++)
    {
        unsigned char *image;
        int width, height;

        char filename[24];
        int imageNum = 3 + rand() % dirCount[i];
        sprintf(filename, "%s%d/%d.pgm", PATH_TO_DATASET, i + 48, dirs[i][5]);
        printf("path: %s\n", filename);
        int opened = read_pgm(filename, &image, &width, &height);

        double input[width * height];
        double* output;
        for (int i = 0; i < width * height; i++)
        {
            input[i] = (double)image[i] / 255.0;
        }

        output = forward_pass(model, input, sigmoid);
        // double* s_output = softmax(model->layers[model->num_layers], output);
        print_arr(10, output);
        free(output); 
        free(image);
    }

    return EXIT_SUCCESS;
}
