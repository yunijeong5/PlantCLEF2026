
#include <iostream>

// Kernel CUDA /thread code
// Ajoute les éléments de deux vecteurs
__global__ void kernel_add(int *a, int *b, int *c) {
    int index = threadIdx.x;
    c[index] = a[index] + b[index];
}



void run_hello_world_add(cudaStream_t stream) {
    std::cout << "Appel à api.cu > run_hello_world_add" << std::endl;

    const int arraySize = 5;
    const int arrayBytes = arraySize * sizeof(int);

    // Allouer de la mémoire hôte
    int h_a[arraySize] = {1, 2, 3, 4, 5};
    int h_b[arraySize] = {10, 20, 30, 40, 50};
    int h_c[arraySize];

    // Allouer de la mémoire sur le GPU
    int *d_a, *d_b, *d_c;
    cudaMalloc((void**)&d_a, arrayBytes);
    cudaMalloc((void**)&d_b, arrayBytes);
    cudaMalloc((void**)&d_c, arrayBytes);

    // Copier les données de l'hôte vers le GPU
    cudaMemcpy(d_a, h_a, arrayBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, arrayBytes, cudaMemcpyHostToDevice);

    // Lancer le kernel avec un bloc de threads
    kernel_add<<<1, arraySize>>>(d_a, d_b, d_c);

    // Copier le résultat du GPU vers l'hôte
    cudaMemcpy(h_c, d_c, arrayBytes, cudaMemcpyDeviceToHost);

    // Afficher le résultat
    std::cout << "Result: ";
    for (int i = 0; i < arraySize; i++) {
        std::cout << h_c[i] << " ";
    }
    std::cout << std::endl;

    // Libérer la mémoire sur le GPU
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return;
}