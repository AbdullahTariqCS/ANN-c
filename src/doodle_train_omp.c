#include <model_softmax_omp.h>
#include <time.h>
#include <dir.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <array.h>
#include <sigmoid.h>
#include <pgm.h>

char PATH_TO_DATASET[] = "./datasets/";

int main(int argc, char *argv[])
{
  Model *model;

  int epochs = 10, images_per_epoch = 100;
  if (argc > 1)
    epochs = atoi(argv[1]);
  if (argc > 2)
    images_per_epoch = atoi(argv[2]);
  if (argc > 3)
  {
    int num_layers = atoi(argv[3]);
    if (argc < 4 + num_layers)
    {
      fprintf(stderr, "not enough layer values");
      exit(EXIT_FAILURE);
    }
    int layers[num_layers];
    for (int i = 4; i < 4 + num_layers; i++)
    {
      layers[i - 4] = atoi(argv[i]);
    }

    model = initialize_model(num_layers - 1, layers);
  }
  else
  {
    int num_layers = 4;
    int layers[] = {576, 320, 64, 10};
    model = initialize_model(num_layers - 1, layers);
  }

  model->learning_rate = 0.01;
  printf("Model Layers (%d): ", model->num_layers + 1);
  for (int i = 0; i < model->num_layers + 1; i++)
    printf("%d ", model->layers[i]);
  printf("\n");

  printf("Epochs %d\n", epochs);
  printf("Images Per Epoch %d\n", images_per_epoch);

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

      // for (int k = 0; i < 10; k++)
      e += backward_pass(model, input, output, sigmoid, dsigmoid);
      free(image);
    }
    printf("Error: %f\n", e / images_per_epoch);
    // e /= (images_per_epoch * iteration_per_image);
    // if (e < min_e)
    // {
    //     printf("Error: %f\n", e);
    //     free_model(model);
    //     model = local_model;
    //     min_e = e;
    //     epoch_used = i;
    //     write_model(model, "doodle_classifier.txt");
    // }
    // else
    // {
    //     printf("Error: %f. Reusing model from epoch %d\n", e, epoch_used);
    //     free_model(local_model);

    // }
    write_model(model, "doodle_classifier_omp.txt");
  }

  printf("Forward Pass\n");
  printf("   ");
  for (int i = 0; i < 10; i++)
  {
    printf("%6d |", i);
  }
  printf("\n");
  for (int i = 0; i < 10; i++)
  {
    unsigned char *image;
    int width, height;

    char filename[24];
    int imageNum = 3 + rand() % dirCount[i];
    sprintf(filename, "%s%d/%d.pgm", PATH_TO_DATASET, i + 48, dirs[i][5]);
    // printf("path: %s\n", filename);
    int opened = read_pgm(filename, &image, &width, &height);

    double input[width * height];
    double *output;
    for (int i = 0; i < width * height; i++)
    {
      input[i] = (double)image[i] / 255.0;
    }

    output = forward_pass(model, input, sigmoid);
    printf("%d: ", i);
    for (int j = 0; j < 10; j++)
    {
      printf("%6.2f |", output[j] * 100);
    }
    printf("\n");

    free(output);
    free(image);
  }

  return EXIT_SUCCESS;
}
