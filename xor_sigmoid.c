#include "model.c"

int main()
{
    int numLayers = 4; 
    int layers[] = {2, 2, 1};
    int epochs = 10000; 
    Model* model = initialize_model(numLayers-1, layers); 
    model->learning_rate = 0.1; 
    
    
    // FILE* f; 
    // f = fopen("xor.txt", "r");
    // fscanf(f, "%d", &epochs);
    srand(time(NULL));
    
    for(int i = 0; i < epochs; i++)
    {
        int a = rand() % 2, b = rand() % 2; 
        int o = a ^ b; 
        
        // fscanf(f, "%d %d %d", &a, &b, &o);
        double input[2] = {(double)a, (double)b};
        double output[1] = {(double)o};
        // double input[2] = {1, 1};
        // double output[1] = {1};
        
        // printf("%f %f %f\n", input[0], input[1], output[0]);
        backward_pass(model, input, output);
    }

    for(int i = 0; i < 2; i++)
    {
        for(int j = 0; j < 2; j++)
        {
            double input[2] = {(double)i, (double)j};
            double* result = forward_pass(model, input);
            printf("Result for %d^%d: ", i, j); 
            print_arr(1, result);
        }
    }
    return 0;
}