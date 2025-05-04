#include <model.h>
#include <dir.h>
#include <sigmoid.h>
#include <pgm.h>
#include <unistd.h>
#include <array.h>

#define PATH_TO_DATASET "./datasets/"
#define NUM_OF_LAYERS 4
#define LAYERS {576, 320, 64, 10}
#define LEARNING_RATE 0.01
#define EPOCHS 10
#define IMAGE_PER_EPOCH 1000

int main(void)
{
  // Model Setup
  int layers[] = LAYERS;
  printf("Building model and setting learning rate to %d\n", LEARNING_RATE);
  Model *model = initialize_model(NUM_OF_LAYERS - 1, layers);
  model->learning_rate = LEARNING_RATE;

  // Loading directories
  int *directoriesPerClass[10]; // Array to hold directories for each class
  int imagesPerClass[10];       // Array to store the number of images per class

  // Load images from each class directory (0-9)
  for (int i = 0; i < 10; i++)
  {
    char class_name[24];
    sprintf(class_name, "%s%d", PATH_TO_DATASET, i + 48);

    imagesPerClass[i] = IMAGE_PER_EPOCH;
    get_dir(&directoriesPerClass[i], &imagesPerClass[i], class_name);
    printf("Loaded directory %s with %d images\n", class_name, imagesPerClass[i]);
  }

  // Training loop
  srand(time(NULL)); // Seed random number generator

  for (int epoch = 0; epoch < EPOCHS; epoch++)
  {
    printf("Training Epoch %d\n", epoch + 1);
    double total_error = 0.0;

    for (int i = 0; i < IMAGE_PER_EPOCH; i++)
    {
      // Randomly select a class for training
      int class = rand() % 10;

      // Randomly select an image from that class
      int image_index = rand() % imagesPerClass[class];
      char filename[24];
      int file_exists = 0;
      int max_retries = 10;
      int retries = 0;

      while (!file_exists && retries < max_retries)
      {
        sprintf(filename, "%s%d/%d.pgm", PATH_TO_DATASET, class + 48, directoriesPerClass[class][image_index]);

        if (access(filename, F_OK) != -1)
        {
          file_exists = 1; // File exists
        }
        else
        {
          retries++;
          image_index = rand() % imagesPerClass[class];
        }
      }

      if (!file_exists)
      {
        printf("Failed to find a valid file after %d retries.\n", max_retries);
        exit(EXIT_FAILURE);
      }

      // Read the PGM image file
      unsigned char *image;
      int width, height;
      if (!read_pgm(filename, &image, &width, &height))
      {
        continue; // Skip if image can't be read
      }

      // Normalize image to [0, 1] range
      double input[width * height];
      for (int j = 0; j < width * height; j++)
      {
        input[j] = (double)image[j] / 255.0;
      }

      // Prepare the output (one-hot encoding for the correct class)
      double output[10] = {0.0};
      output[class] = 1.0;

      // Perform the backpropagation step
      double error = backward_pass(model, input, output, sigmoid, dsigmoid);
      total_error += error;

      // Free the image memory
      free(image);
    }

    // Print average error for the epoch
    printf("Epoch %d Error: %.4f\n", epoch + 1, total_error / IMAGE_PER_EPOCH);

    // Save the model after each epoch
    write_model(model, "doodle_classifier.txt");
  }

  printf("Training Complete!\n");

  for (int i = 0; i < 10; i++)
  {
    unsigned char *image;
    int width, height;

    char filename[24];
    int imageNum = 3 + rand() % imagesPerClass[i];
    int file_exists = 0;
    int retries = 0;
    int max_retries = 10; // Maximum retries before giving up

    // Retry loop to find an existing file
    while (!file_exists && retries < max_retries)
    {
      // Construct the filename
      sprintf(filename, "%s%d/%d.pgm", PATH_TO_DATASET, i + 48, directoriesPerClass[i][imageNum]);
      printf("Trying path: %s\n", filename);

      // Check if the file exists
      if (access(filename, F_OK) != -1)
      {
        file_exists = 1; // File exists
      }
      else
      {
        retries++;
        imageNum = 3 + rand() % imagesPerClass[i]; // Try a different image if the current one doesn't exist
      }
    }

    // If file exists, proceed with reading and processing
    if (file_exists)
    {
      int opened = read_pgm(filename, &image, &width, &height);
      if (opened)
      {
        double input[width * height];
        double *output;
        for (int j = 0; j < width * height; j++)
        {
          input[j] = (double)image[j] / 255.0; // Normalize pixel values to [0, 1]
        }

        output = forward_pass(model, input, sigmoid); // Run the forward pass

        // Print the output
        print_arr(10, output);
        free(output);
        free(image);
      }
      else
      {
        printf("Error reading image from file: %s\n", filename);
      }
    }
    else
    {
      printf("Failed to find a valid file after %d retries.\n", max_retries);
    }
  }

  return 0;
}