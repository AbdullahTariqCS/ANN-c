#include <stdio.h>
#include <stdlib.h>
#include "matrix.c"
#include "util/sigmoid.c" 
#include "util/array.c"
#include <time.h>


#define DEBUG 0
#define GRADIENT_PRINT 0

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
    mat_normalize(frontier);
    mat_map(frontier, sigmoid, 1);

    for(int i = 0; i < model->num_layers; i++)
    {
        Matrix* frontier_t = mat_mul(model->weights[i], frontier); 
        mat_add(frontier_t, model->bias[i], 1);
        mat_map(frontier_t, sigmoid, 1);

        if (DEBUG)
        {
            printf("\ni: %d\n", i);
            printf("\nFrontier (%d, %d)\n", frontier->rows, frontier->columns); 
            mat_print(frontier);

            printf("\nWeights[%d] (%d, %d)\n", i, model->weights[i]->rows, model->weights[i]->columns);
            mat_print(model->weights[i]);

            printf("\nBias[%d] (%d, %d)\n",i, model->bias[i]->rows, model->bias[i]->columns);
            mat_print(model->bias[i]);

            printf("\nfrontier_t: sigmoid(mul(model->weights[%d], frontier) + model->bias[%d])\n", i, i);
            mat_print(frontier_t);
        }


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
    Matrix* O = arr_to_mat(model->layers[model->num_layers], output, 1);
    Matrix* frontier[model->num_layers+1];
    frontier[0] = arr_to_mat(model->layers[0], input, 1);
    mat_normalize(frontier[0]);
    mat_map(frontier[0], sigmoid, 1);

    if (DEBUG)
    {
        printf("\nFrontier[%d]: \n", 0); 
        mat_print(frontier[0]);

        printf("\noutput: \n"); 
        mat_print(O);
    }

    for(int i = 0; i < model->num_layers; i++)
    {
        frontier[i+1] = mat_mul(model->weights[i], frontier[i]); 

        mat_add(frontier[i+1], model->bias[i], 1);
        mat_map(frontier[i+1], sigmoid, 1);
        if (DEBUG)
        {
            printf("\n model->weights[%d]\n", i); 
            mat_print(model->weights[i]);

            printf("\n model->bias[%d] \n", i); 
            mat_print(model->bias[i]);

            printf("\nfrontier[%d]: sigmoid(mat_mul(model->weights[%d], frontier[%d]) + model->bias[%d])\n", i+1, i, i, i);
            mat_print(frontier[i+1]);
        }
    }


    Matrix* error[model->num_layers+1];
    Matrix* neg_frontier = mat_scale(frontier[model->num_layers], -1, 0);
    error[model->num_layers] = mat_add(O, neg_frontier, 0); 
    if (DEBUG)
    {
        printf("\n neg_frontier = -frontier[%d: model->num_layers]\n", model->num_layers);
        mat_print(neg_frontier);
        printf("error[%d: model->num_layer] = O + neg_frontier\n", model->num_layers); 
        mat_print(error[model->num_layers]);
    }

    free(neg_frontier);


    for(int i = model->num_layers-1; i >= 0; i--)
    {
        //gradient = error[i+1] * lr * dsigmoid(output: frontier[i+1])
        //d_weights[i] = gradient x input: frontier[i]
        
        Matrix* weights_t = mat_transpose(model->weights[i], 0);
        error[i] = mat_mul(weights_t, error[i+1]);

        Matrix* gradient = mat_map(frontier[i+1], dsigmoid, 0);
        mat_scalar_mul(gradient, error[i+1], 1);
        mat_scale(gradient, model->learning_rate, 1);

        // Matrix* neg_gradient = mat_scale(gradient, -1, 0);
        mat_add(model->bias[i], gradient, 1);

        Matrix* frontier_t =  mat_transpose(frontier[i], 0);
        Matrix* d_weights = mat_mul(gradient, frontier_t);

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

            printf("\nfrontier[%d]\n", i+1); 
            mat_print(frontier[i+1]);

            printf("\n gradient = map(frontier[%d], dsigmoid) * errror[%d] * lr: %f\n", i+1, i+1, model->learning_rate); 
            mat_print(gradient);

        }
        
        free(gradient);
        free(frontier_t);
        free(weights_t);
        free(d_weights);
        // free(neg_gradient);
    }

    double* e = mat_to_arr(error[model->num_layers]); 
    return e; 
}


