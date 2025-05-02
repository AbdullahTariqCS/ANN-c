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
    int numLayers = 4;
    int layers[] = {576, 64, 64, 10};
    Model *model = initialize_model(numLayers - 1, layers);
    model->learning_rate = 0.01; 

    int epochs, images_per_epoch, images_per_class;
    if (argc <= 1) epochs = 10;
    else epochs = atoi(argv[1]);

    if (argc <= 2) images_per_epoch = 1000;
    else images_per_epoch = atoi(argv[2]);

    if (argc <= 3) images_per_class = 1000;
    else images_per_class = atoi(argv[3]);

    printf("Epochs %d\n", epochs);
    printf("Images Per Epoch %d\n", images_per_epoch);
    printf("Images Per Class %d\n", images_per_class);

    // getting the names of dataset
    int *dirs[10];
    int dirCount[10];
    for (int i = 0; i < 10; i++)
    {
        char class_name[24];
        sprintf(class_name, "%s%d", PATH_TO_DATASET, i + 48);
        dirCount[i] = images_per_class;
        get_dir(&dirs[i], &dirCount[i], class_name);
        printf("Got directory %s (%d)\n", class_name, dirCount[i]);
    }


    srand(time(NULL));
    for (int i = 0; i < epochs; i++)
    {
        printf("Training for Epoch %d. ", i);
        double e = 0.0; 
        int class;
        for (int j = 0; j < images_per_epoch; j++)
        {
            unsigned char *image;
            int width, height;
            class = rand() % 10;
            int imageNum = 3 + rand() % dirCount[class];

            char filename[24];
            sprintf(filename, "%s%d/%d.pgm", PATH_TO_DATASET, class + 48, dirs[class][imageNum]);
            // printf("path: %s\n", filename);
            if (!read_pgm(filename, &image, &width, &height))
                continue;

            double input[width * height];
            double output[10];
            for (int k = 0; k < width * height; k++)
                input[k] = (double)image[k] / 255.0;

            for (int k = 0; k < 10; k++)
                output[k] = (double)(k == class);

            e += backward_pass(model, input, output, relu, drelu);
            free(image);
        }
        printf("Error: %f\n", e / images_per_epoch);
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

        output = forward_pass(model, input, relu);
        // double* s_output = softmax(model->layers[model->num_layers], output);
        print_arr(10, output);
        free(output); 
        free(image);
    }

    return EXIT_SUCCESS;
}
