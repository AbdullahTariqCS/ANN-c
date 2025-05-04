#include <model_softmax.h>
#include <time.h>
#include <dir.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <array.h>
#include <sigmoid.h>
#include <pgm.h>
#include <omp.h>
#include <linux/time.h>

char PATH_TO_DATASET[] = "./datasets/";

int main(int argc, char *argv[])
{
  Model *model;

  // Initializing the model
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

  printf("Model details:\n");
  printf("\tLayers (%d): ", model->num_layers + 1);
  for (int i = 0; i < model->num_layers + 1; i++)
    printf("%d ", model->layers[i]);
  printf("\n");

  printf("\tEpochs %d\n", epochs);
  printf("\tImages Per Epoch %d\n", images_per_epoch);

  // Querying available dataset items
  // Filenames are in the order [class][file number], where data is file name (which is a number)
  int *filenames[10];
  int totalFilesCount[10];
  for (int i = 0; i < 10; i++)
  {
    char class_name[24];
    sprintf(class_name, "%s%d", PATH_TO_DATASET, i + 48);
    totalFilesCount[i] = -1;
    get_dir(&filenames[i], &totalFilesCount[i], class_name);
    printf("Searched directory %s (%d items)\n", class_name, totalFilesCount[i]);
  }

  // Training the model
  srand(time(NULL));
  double total_total_time = 0.0;
  for (int i = 0; i < epochs; i++)
  {
    printf("Training for Epoch %d. ", i);
    double e = 0.0;
    double total_time = 0.0;
    int class;

    for (int j = 0; j < images_per_epoch; j++)
    {
      unsigned char *image;
      int width, height;
      class = rand() % 10;
      int imageNum = 3 + rand() % totalFilesCount[class];

      char filename[24];
      sprintf(filename, "%s%d/%d.pgm", PATH_TO_DATASET, class + 48, filenames[class][imageNum]);
      // printf("path: %s\n", filename);
      if (!read_pgm(filename, &image, &width, &height))
        continue;

      double input[width * height];
      double output[10];
      for (int k = 0; k < width * height; k++)
        input[k] = (double)image[k] / 255.0;

      for (int k = 0; k < 10; k++)
        output[k] = (double)(k == class);

      struct timespec start, end;
      clock_gettime(CLOCK_MONOTONIC, &start);
      e += backward_pass(model, input, output, sigmoid, dsigmoid);
      clock_gettime(CLOCK_MONOTONIC, &end);
      total_time += (end.tv_sec - start.tv_sec) +
                    (end.tv_nsec - start.tv_nsec) / 1e6;

      free(image);
    }

    total_total_time += total_time / images_per_epoch;

    printf("Error: %f. ", e / images_per_epoch);
    printf("Time taken: %.3f ms\n", total_time / images_per_epoch);

    // write_model(model, "doodle_classifier.txt"); Not needed for now
  }
  total_total_time /= epochs;

  // Testing it
  printf("Forward Pass\n");
  for (int i = 0; i < 10; i++)
  {
    unsigned char *image;
    int width, height;

    char filename[24];
    int imageNum = 3 + rand() % totalFilesCount[i];
    sprintf(filename, "%s%d/%d.pgm", PATH_TO_DATASET, i + 48, filenames[i][5]);
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

  printf("Average time taken: %.3f ms\n", total_total_time);

  return EXIT_SUCCESS;
}