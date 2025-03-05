#include <stdio.h>
#include <stdlib.h>
#include "matrix.c"
#include "util.c"

#define DEBUG 1

typedef struct Model
{
    int num_layers;
    int *layers; //number of nodes in a layer
    double learning_rate;
    Matrix **weights; // array of matrices
    Matrix **bias;
} Model;

Model *initialize_model(
    int num_layers, //excluding the input layer
    int layers[num_layers]
){
    Model *model = (Model *)malloc(sizeof(Model));
    model->num_layers = num_layers;
    model->layers = (int *)malloc((model->num_layers + 1) * sizeof(int));
    for (int i = 0; i < num_layers+1; i++)
        model->layers[i] = layers[i];

    model->weights = malloc(num_layers * sizeof(Matrix *));
    model->bias = malloc(num_layers * sizeof(Matrix *));

    for (int i = 0; i < num_layers; i++)
    {
        model->weights[i] = mat_init(layers[i + 1], layers[i]);
        mat_rand(model->weights[i]);
        mat_normalize(model->weights[i]);

        model->bias[i] = mat_init(layers[i + 1], 1);
        mat_rand(model->bias[i]);
        mat_normalize(model->bias[i]);
    }

    return model;
}

double* forward_pass(Model* model, double input[model->layers[0]])
{
    
    Matrix* frontier = arr_to_mat(model->layers[0], input, 1);

    for(int i = 0; i < model->num_layers; i++)
    {
        if (DEBUG)
        {
            printf("--------Layer %d---------\n", i);
            printf("Frontier (%d, %d)\n", frontier->rows, frontier->columns); 
            mat_print(frontier);

            printf("\nWeights (%d, %d)\n", model->weights[i]->rows, model->weights[i]->columns);
            mat_print(model->weights[i]);

            printf("\nBias (%d, %d)\n", model->bias[i]->rows, model->bias[i]->columns);
            mat_print(model->bias[i]);
            printf("\n");
        }

        Matrix* frontier_t = mat_mul(model->weights[i], frontier); 
        mat_add(frontier_t, model->bias[i], 1);
        mat_map(frontier_t, sigmoid, 1);

        mat_free(frontier, 0);
        frontier = frontier_t;

    }
    double* res = mat_to_arr(frontier);
    mat_free(frontier, 0);
    return res;
}

double* backward_pass(
    Model* model, 
    double input[model->layers[0]], 
    double output[model->layers[model->num_layers-1]]
){    
    Matrix* O = arr_to_mat(model->layers[model->num_layers-1], output, 1);
    Matrix* frontier[model->num_layers+1];
    frontier[0] = arr_to_mat(model->layers[0], input, 1);

    for(int i = 0; i < model->num_layers; i++)
    {
        frontier[i+1] = mat_mul(model->weights[i], frontier[i]); 

        mat_add(frontier[i+1], model->bias[i], 1);
        mat_map(frontier[i+1], sigmoid, 1);
    }


    Matrix* error[model->num_layers+1];
    Matrix* neg_frontier = mat_scale(frontier[model->num_layers-1], -1, 0);
    error[model->num_layers] = mat_add(O, neg_frontier, 0); 
    free(neg_frontier);

    for(int i = model->num_layers-1; i >= 0; i--)
    {

        error[i] = mat_mul(mat_transpose(model->weights[i], 0), error[i+1]);

        Matrix* d_frontier = mat_map(frontier[i+1], dsigmoid, 0);
        
        mat_scalar_mul(d_frontier, error[i], 1);
        mat_scale(d_frontier, model->learning_rate, 1);

        Matrix* frontier_t = mat_transpose(frontier[i], 0);
        Matrix* d_weights = mat_mul(d_frontier, frontier_t);

        mat_scale(d_weights, -1, 1);
        mat_add(model->weights[i], d_weights, 1);

        free(frontier_t);
        free(d_frontier);
        free(d_weights);
    }

    double* e = mat_to_arr(error[model->num_layers]); 
    return e; 
}

int main()
{
    int layers[3] = {2, 3, 2};
    double input[2] = {0, 0};
    double output[2] = {1, 0};
    Model* model = initialize_model(2, layers); 
    backward_pass(model, input, output);
    double* result = forward_pass(model, input);
    printf("Result: "); 
    print_arr(2, result);
    return 0;
}
