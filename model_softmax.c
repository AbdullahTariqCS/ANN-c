#include <stdio.h>
#include <stdlib.h>
#include "matrix_threaded.c"
#include "util/softmax.c"
#include "util/array.c"
#include <time.h>
#include <unistd.h>
#include "util/absolute.c"

#define DEBUG 0
#define GRADIENT_PRINT 0

typedef struct Model
{
    int num_layers;
    int *layers; // number of nodes in a layer
    double learning_rate;
    Matrix **weights; // array of matrices
    Matrix **bias;
} Model;

void write_model(Model *model, char *filename)
{
    FILE *file;
    file = fopen(filename, "w");
    if (!file)
    {
        fprintf(stderr, "Cannot open file %s\n", filename);
        exit(EXIT_FAILURE);
    }

    // Structure of model
    // num_layers
    // layers
    // learning rate
    // weights
    // bias
    fprintf(file, "%d\n", model->num_layers);
    for (int i = 0; i < model->num_layers + 1; i++)
        fprintf(file, "%d;", model->layers[i]);
    fprintf(file, "\n");
    fprintf(file, "%f\n", model->learning_rate);

    for (int i = 0; i < model->num_layers; i++)
    {
        mat_write(model->weights[i], file);
        mat_write(model->bias[i], file);
    }
}

Model *read_model(char *filename)
{

    FILE *file = fopen(filename, "r");
    Model *model = (Model *)malloc(sizeof(Model));
    if (!file)
    {
        fprintf(stderr, "Cannot open file %s\n", filename);
        exit(EXIT_FAILURE);
    }

    fscanf(file, "%d\n", &model->num_layers);
    model->layers = (int *)malloc(sizeof(int) * model->num_layers + 1);
    model->weights = malloc(model->num_layers * sizeof(Matrix *));
    model->bias = malloc(model->num_layers * sizeof(Matrix *));
    for (int i = 0; i < model->num_layers + 1; i++)
    {
        fscanf(file, "%d;", &model->layers[i]);
    }
    fscanf(file, "\n");

    for (int i = 0; i < model->num_layers; i++)
    {
        model->weights[i] = mat_read(file);
        model->bias[i] = mat_read(file);
    }
}

Model *initialize_model(
    int num_layers, // excluding the input layer
    int layers[num_layers])
{
    Model *model = (Model *)malloc(sizeof(Model));
    model->num_layers = num_layers;
    model->layers = (int *)malloc((model->num_layers + 1) * sizeof(int));
    for (int i = 0; i < num_layers + 1; i++)
        model->layers[i] = layers[i];

    model->weights = malloc(num_layers * sizeof(Matrix *));
    model->bias = malloc(num_layers * sizeof(Matrix *));

    for (int i = 0; i < num_layers; i++)
    {
        model->weights[i] = mat_init(layers[i + 1], layers[i]);
        mat_rand_xavier(model->weights[i]);
        mat_normalize(model->weights[i]);

        model->bias[i] = mat_init(layers[i + 1], 1);
        // mat_rand(model->bias[i]);
        // mat_normalize(model->bias[i]);
        mat_zeros(model->bias[i]);
    }
    return model;
}

double *forward_pass(Model *model, double input[model->layers[0]], double activation(double))
{

    Matrix *frontier = arr_to_mat(model->layers[0], input, 1);
    // mat_normalize(frontier);
    // mat_map(frontier, activation, 1);

    for (int i = 0; i < model->num_layers; i++)
    {
        Matrix *frontier_t = mat_mul(model->weights[i], frontier);
        mat_add(frontier_t, model->bias[i], 1);
        if (i != model->num_layers - 1)
            mat_map(frontier_t, activation, 1);
        else
            softmax(frontier_t);

        if (DEBUG)
        {
            printf("\ni: %d\n", i);
            printf("\nFrontier (%d, %d)\n", frontier->rows, frontier->columns);
            mat_print(frontier);

            printf("\nWeights[%d] (%d, %d)\n", i, model->weights[i]->rows, model->weights[i]->columns);
            mat_print(model->weights[i]);

            printf("\nBias[%d] (%d, %d)\n", i, model->bias[i]->rows, model->bias[i]->columns);
            mat_print(model->bias[i]);

            printf("\nfrontier_t: sigmoid(mul(model->weights[%d], frontier) + model->bias[%d])\n", i, i);
            mat_print(frontier_t);
        }

        mat_free(frontier, 0);
        frontier = frontier_t;
    }

    double *res = mat_to_arr(frontier);
    mat_free(frontier, 0);
    return res;
}

double backward_pass(
    Model *model,
    double input[model->layers[0]],
    double output[model->layers[model->num_layers - 1]],
    double activation(double),
    double dactivation(double))
{
    Matrix *O = arr_to_mat(model->layers[model->num_layers], output, 1);
    Matrix *frontier[model->num_layers + 1];
    Matrix *softmaxed;
    frontier[0] = arr_to_mat(model->layers[0], input, 1);
    // mat_normalize(frontier[0]);
    // mat_map(frontier[0], activation, 1);

    if (DEBUG)
    {
        printf("\nFrontier[%d]: \n", 0);
        mat_print(frontier[0]);

        printf("\noutput: \n");
        mat_print(O);
    }

    for (int i = 0; i < model->num_layers; i++)
    {
        frontier[i + 1] = mat_mul(model->weights[i], frontier[i]);

        mat_add(frontier[i + 1], model->bias[i], 1);
        if (i != model->num_layers - 1)
            mat_map(frontier[i + 1], activation, 1);
        else
            softmax(frontier[i + 1]);
        if (DEBUG)
        {
            printf("\n model->weights[%d]\n", i);
            mat_print(model->weights[i]);

            printf("\n model->bias[%d] \n", i);
            mat_print(model->bias[i]);

            printf("\nfrontier[%d]: sigmoid(mat_mul(model->weights[%d], frontier[%d]) + model->bias[%d])\n", i + 1, i, i, i);
            mat_print(frontier[i + 1]);
        }
    }

    // softmaxed = mat_copy(frontier[model->num_layers]);
    // softmax(frontier[model->num_layers]);
    Matrix *error[model->num_layers + 1];

    // Residual Errors
    Matrix *neg_O = mat_scale(O, -1, 0);
    error[model->num_layers] = mat_add(frontier[model->num_layers], neg_O, 0);

    if (DEBUG)
    {
        printf("\n neg_O = -frontier[%d: model->num_layers]\n", model->num_layers);
        mat_print(neg_O);
        printf("error[%d: model->num_layer] = O + neg_O\n", model->num_layers);
        mat_print(error[model->num_layers]);
    }

    mat_free(neg_O, 0);

    for (int i = model->num_layers - 1; i >= 0; i--)
    {
        // gradient = error[i+1] * lr * dsigmoid(output: frontier[i+1])
        // d_weights[i] = gradient x input: frontier[i]

        Matrix *weights_t = mat_transpose(model->weights[i], 0);
        error[i] = mat_mul(weights_t, error[i + 1]);
        // mat_map(error[i], absolute, 1);

        Matrix *gradient;
        if (i != model->num_layers - 1)
        {
            gradient = mat_map(frontier[i + 1], dactivation, 0);
            mat_scalar_mul(gradient, error[i+1], 1);
        }
        else
        {
            gradient = mat_copy(error[i+1]);
        }
        
        // mat_scalar_mul(gradient, error[i+1], 1);
        mat_scale(gradient, model->learning_rate, 1);

        Matrix* neg_gradient = mat_scale(gradient, -1, 0);
        mat_add(model->bias[i], neg_gradient, 1);

        Matrix *frontier_t = mat_transpose(frontier[i], 0);
        Matrix *d_weights = mat_mul(neg_gradient, frontier_t);

        // mat_scale(d_weights, -1, 1);
        mat_add(model->weights[i], d_weights, 1);

        if (GRADIENT_PRINT)
        {
            printf("\n");
            mat_print(gradient);
        }

        if (DEBUG)
        {
            printf("\ni : %d\n", i);
            printf("\n weights_t= transpose(model->weights[%d])\n", i);
            mat_print(weights_t);

            printf("\n error[%d]\n", i);
            mat_print(error[i]);

            printf("\nfrontier[%d]\n", i + 1);
            mat_print(frontier[i + 1]);

            printf("\n gradient = map(frontier[%d], dsigmoid) * errror[%d] * lr: %f\n", i + 1, i + 1, model->learning_rate);
            mat_print(gradient);
        }

        mat_free(gradient, 0);
        mat_free(frontier_t, 0);
        mat_free(weights_t, 0);
        mat_free(d_weights, 0);
        mat_free(neg_gradient, 0);
    }

    double *e = mat_to_arr(error[model->num_layers]);
    double cross_entropy_error = cross_entropy(O, frontier[model->num_layers]);
    // for (int i = 0 ; i < model->num_layers+1; i++)
    // {
    //     cross_entropy_error +=e[i];
    // }

    for (int i = 0; i < model->num_layers + 1; i++)
    {
        mat_free(frontier[i], 0);
        mat_free(error[i], 0);
    }
    return cross_entropy_error;
}
