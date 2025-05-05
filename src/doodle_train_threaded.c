#include <model_softmax_threaded.h>
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

char PATH_TO_DATASET[] = "./datasets/";

int main(int argc, char *argv[])
{
  Model *model;

  // Initializing the model
  int epochs = 10, images_per_epoch = 100, num_threads = 1, verbose = 1;
  double learning_rate = 0.01;

  if (argc > 1)
    epochs = atoi(argv[1]);
  if (argc > 2)
    images_per_epoch = atoi(argv[2]);
  if (argc > 3)
    num_threads = atoi(argv[3]);
  if (argc > 4)
    verbose = atoi(argv[4]);
  if (argc > 5)
    learning_rate = atof(argv[5]);

  int num_layers = 4;
  int layers[] = {576, 320, 64, 10};
  model = initialize_model(num_layers - 1, layers);
  model->learning_rate = learning_rate;

  if (verbose)
  {
    printf("Setting threads to %d\n", num_threads);
    printf("Model details:\n");
    printf("\tLayers (%d): ", model->num_layers + 1);
    for (int i = 0; i < model->num_layers + 1; i++)
      printf("%d ", model->layers[i]);
    printf("\n");
    printf("\tEpochs %d\n", epochs);
    printf("\tImages Per Epoch %d\n", images_per_epoch);
    printf("\tLearning Rate %f\n", model->learning_rate);
  }

  // Querying available dataset items
  // Filenames are in the order [class][file number], where data is file name (which is a number)
  if (verbose)
    printf("Querying directories:\n");
  int *filenames[10];
  int totalFilesCount[10];
  for (int i = 0; i < 10; i++)
  {
    char class_name[24];
    sprintf(class_name, "%s%d", PATH_TO_DATASET, i + 48);
    totalFilesCount[i] = -1;
    get_dir(&filenames[i], &totalFilesCount[i], class_name);
    if (verbose)
      printf("\tSearched directory %s (%d items)\n", class_name, totalFilesCount[i]);
  }

  // Training the model
  if (verbose)
    printf("Training model:\n");
  srand(time(NULL));
  double average_time_across_epochs = 0.0;
  double average_e;
  for (int i = 0; i < epochs; i++)
  {
    if (verbose)
      printf("\tEpoch %d:\n", i + 1);
    double e = 0.0;
    int class;

    double epoch_start_time = omp_get_wtime();

    for (int j = 0; j < images_per_epoch; j++)
    {
      unsigned char *image;
      int width, height;
      class = rand() % 10;
      int imageNum = 3 + rand() % totalFilesCount[class];

      char filename[24];
      sprintf(filename, "%s%d/%d.pgm", PATH_TO_DATASET, class + 48, filenames[class][imageNum]);

      if (!read_pgm(filename, &image, &width, &height))
        continue;

      double input[width * height];
      double output[10];
      for (int k = 0; k < width * height; k++)
        input[k] = (double)image[k] / 255.0;

      for (int k = 0; k < 10; k++)
        output[k] = (double)(k == class);

      e += backward_pass(model, input, output, sigmoid, dsigmoid, num_threads);

      free(image);
    }

    double epoch_end_time = omp_get_wtime();
    double epoch_time = epoch_end_time - epoch_start_time;
    average_time_across_epochs += epoch_time;

    if (verbose)
    {
      printf("\t\tTotal epoch time: %f seconds\n", epoch_time);
      printf("\t\tError: %f. \n", e / images_per_epoch);
    }
    average_e += e / images_per_epoch;

    // write_model(model, "doodle_classifier.txt"); Not needed for now
  }
  average_time_across_epochs /= epochs;
  average_e /= epochs;

  if (verbose)
  {
    printf("Forward Pass:\n");
    printf("   ");
    for (int i = 0; i < 10; i++)
    {
      printf("%6d |", i);
    }
    printf("\n");
  }
  for (int i = 0; i < 10; i++)
  {
    unsigned char *image;
    int width, height;

    char filename[24];
    int imageNum = 3 + rand() % totalFilesCount[i];
    sprintf(filename, "%s%d/%d.pgm", PATH_TO_DATASET, i + 48, filenames[i][5]);
    // printf("path: %s\n", filename);
    int opened = read_pgm(filename, &image, &width, &height);

    if (!opened)
    {
      i--;
      continue;
    }

    double input[width * height];
    double *output;
    for (int i = 0; i < width * height; i++)
    {
      input[i] = (double)image[i] / 255.0;
    }

    output = forward_pass(model, input, sigmoid, num_threads);
    if (verbose)
    {
      printf("%d: ", i);
      for (int j = 0; j < 10; j++)
      {
        printf("%6.2f |", output[j] * 100);
      }
      printf("\n");
    }
    else
    {
      for (int j = 0; j < 10; j++)
      {
        printf("%.5f,", output[j]);
      }
      printf("\n");
    }

    free(output);
    free(image);
  }

  if (verbose)
  {
    printf("Total time taken: %.5f ms\n", average_time_across_epochs);
  }
  else
  {
    printf("%0.10f\n", average_e);
    printf("%0.10f\n", average_time_across_epochs);
  }

  return EXIT_SUCCESS;
}
