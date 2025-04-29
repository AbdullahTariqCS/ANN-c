#include "./util/dir.c"
#include "model.c"
#include "util/pgm.c"
#include <string.h>
#include <stdlib.h>
#include <unistd.h>

char PATH_TO_DATASET[] =  "./dataset/";

int main(int argc, char* argv[])
{
    int numLayers = 4; 
    int layers[] = {576, 16, 16, 10};
    Model* model = initialize_model(numLayers-1, layers);

    int epochs, images_per_epoch, images_per_class;
    if (argc <= 1) epochs = 100;  
    else epochs = atoi(argv[1]);
    
    if (argc <= 2) images_per_epoch = 1000;
    else images_per_epoch = atoi(argv[2]);

    if (argc <= 3) images_per_class = 1000;
    else images_per_class = atoi(argv[3]);

    printf("Epochs %d\n", epochs);
    printf("Images Per Epoch %d\n", images_per_epoch);
    printf("Images Per Class %d\n", images_per_class);


    //getting the names of dataset
    char *** dirs = malloc(sizeof(char*) * 10);
    // int dirCount[10];
    for (int i = 0; i < 10; i++)
    {
        char class_name[24]; 
        sprintf(class_name, "%s%d", PATH_TO_DATASET, i+48);
        printf("Got directory %s\n", class_name); 
        get_dir(&dirs[i], &images_per_class, class_name);
    }


    for (int i = 0; i < epochs; i++)
    {
        printf("Training for Epoch %d\n",  i); 
        for(int j = 0; j < images_per_epoch; j++)
        {
            unsigned char *image; 
            int width, height;
            int class = rand() % 10; 
            int imageNum = rand() % images_per_class;
            char filename[24]; 
            sprintf(filename, "%s%d/%s", PATH_TO_DATASET, class + 48, dirs[class][imageNum]);
            printf("path: %s", filename);
            readPGM(filename, &image, &width, &height);
            
            double input[width * height]; 
            double output[10];
            for(int i = 0; i < width * height; i++)
            {
                input[i] = (double)image[i];
            }

            for(int i = 0; i < 10; i++)
            {
                output[i] = (double)(i == class);
            }
            // backward_pass(model, input, output);
        }
    }
    return EXIT_SUCCESS;
    
}
