
/// Différentes implémentations du SoftMax, suivant les entrées:
// - Soit un vecteur, soit une matrice (le SoftMax est alors calculé ligne par ligne -> pour batcher les calculs)
// - 2 fonctions distinctes suivant que la taille du vecteur dépasse ou pas 1024 (i.e. nb de thread max par bloc)
//   Si le vecteur a une taille > 1024, on fait du calcul séquentiel à l'intérieur d'un bloc de threads (i.e. batch par 1024)

//#include <cuda_runtime.h>
#include <iostream>
#include <math.h>


unsigned int nextPowerOfTwo(unsigned int n) {
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n++;
    return n;
}

__global__ void kernel_softmax_vec_max1024(float *input, float *output, int N) {
    // Attention: Le nb de thread doit etre une puissance de 2 et supérieur à N
    // Maximum 1024 threads, et N <= 1024

    extern __shared__ float sdata[];
    __shared__ float input_0;
    __shared__ float max;
    __shared__ float sum;

    // Chaque thread charge un élément de l'entrée dans la mémoire partagée
    unsigned int tid = threadIdx.x;
    unsigned int i   = threadIdx.x;

    // Recherche du max au sein du vecteur
    input_0 = input[0];
    sdata[tid] = (i < N) ? input[i] : input_0;
    __syncthreads();

    // Calcul du max / Réduction parallèle / mémoire partagée
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            float a = sdata[tid];
            float b = sdata[tid + s];
            sdata[tid] = (b > a) ? b : a;
        }
        __syncthreads();
    }

    __syncthreads();
    if (tid == 0) {max = sdata[0]; }
    __syncthreads();


    // Calcul du dénominateur (somme des expo)
    // le SoftMax est invariant par translation, donc on enleve le max de tous les élements pour
    // éviter les erreurs numériques dûes à l'exponentielle: Plus grande valeur = exp(0.) = 1.
    float expo = (i < N) ? expf(input[i] - max) : 0.0;              // On applique l'exponentielle
    sdata[tid] = (i < N) ? expo : 0.0;    // sdata sera "détruit" par la réduction
    __syncthreads();

    // Calcul de la somme / Réduction parallèle / mémoire partagée
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    __syncthreads();
    if (tid == 0) {sum = sdata[0];}
    __syncthreads();

    if (i < N) {output[i] = expo / sum;}
    __syncthreads();

}


// 1024 threads par bloc ; et 1 seul bloc
// Version gérant des vecteurs de taille supérieure à 1024
// On ne fait pas plusieurs blocs CUDA, mais on itère sur des blocs fictifs
// car de toute facon, l'implémentation finale utile, prendra des tenseurs en input
// avec beaucoup de SoftMax à calculer, et on finira par itérer sur ces blocs, saturant le nb de SM.
// 1 seul calcul d'attention (1, 12, 1374, 1374) -> 12 * 1374 SoftMax de taille 1374.
// Avec cette implémentation, on limite l'overhead de la synchronisation des calculs sur plusieurs blocs..
// Limite: Si l'on calcule beaucoup de SoftMax de taille 1374, la seconde moitié des calculs
//         sur 1374 - 1024 = 350 éléments, diminue l'occupancy réelle des threads dans la réduction parrallèle
__global__ void kernel_softmax_vec(float *input, float *output, int N) {

    __shared__ float sdata[1024];     // Tableau utilisé pour les réductions parrallèles
    __shared__ float input_0;
    __shared__ float max;
    __shared__ float sum;

    // Chaque thread charge un élément de l'entrée dans la mémoire partagée
    unsigned int tid = threadIdx.x;

    unsigned int n_iter_blocs = (N + blockDim.x - 1) / blockDim.x;


    // Recherche du max au sein du vecteur
    input_0 = input[0];
    max = input[0];
    __syncthreads();
    for (int bid = 0; bid < n_iter_blocs; bid++) {
        unsigned int i = blockDim.x * bid + tid;
        sdata[tid] = (i < N) ? input[i] : input_0;
        __syncthreads();

        // Calcul du max / Réduction parallèle / mémoire partagée
        for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                float a = sdata[tid];
                float b = sdata[tid + s];
                sdata[tid] = (b > a) ? b : a;
            }
            __syncthreads();
        }

        __syncthreads();
        if (tid == 0) {
            float a = sdata[0];
            max = (a > max) ? a : max;
        }
        __syncthreads();
    }
   __syncthreads();


    // Calcul du dénominateur (somme des expo)
    sum = 0.0;
    __syncthreads();
    for (int bid = 0; bid < n_iter_blocs; bid++) {
        unsigned int i = blockDim.x * bid + tid;
        // le SoftMax est invariant par translation, donc on enleve le max de tous les élements pour
        // éviter les erreurs numériques dûes à l'exponentielle: Plus grande valeur = exp(0.) = 1.
        sdata[tid] = (i < N) ? expf(input[i] - max) : 0.0;    // sdata sera "détruit" par la réduction
        __syncthreads();

        // Calcul de la somme / Réduction parallèle / mémoire partagée
        for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                sdata[tid] += sdata[tid + s]; }
            __syncthreads(); }

        __syncthreads();
        if (tid == 0) {sum += sdata[0];}
        __syncthreads();
    }
    __syncthreads();


    // A ce point, on connait le max des éléments, et la somme des exp.
    // On a tout ce qu'il faut pour écrire la sortie dans la mémoire générale du GPU.
    // Il aurait été possible de faire plus de caching (dynamic shared)
    // Mais cela aurait induit une limite maximale sur la taille du SoftMax (limite de la taille de la shared)
    for (int bid = 0; bid < n_iter_blocs; bid++) {
        unsigned int i = blockDim.x * bid + tid;
        if (i < N) {output[i] = expf(input[i] - max) / sum;}
    }

    __syncthreads();

}

void run_softmax_vec(void *input_ptr, void *output_ptr, int n) {

    // Attention: Le nb de thread doit etre une puissance de 2 et supérieur à N
    int threadsPerBlock = nextPowerOfTwo(n);
    if (threadsPerBlock > 1024) {
        threadsPerBlock = 1024;
    }

    std::cout << "Appel à run_softmax_vec (n:" << n << ", threadsPerBlock:" << threadsPerBlock << ")" << std::endl;

    // Allouer de la mémoire sur le device
    float *d_input, *d_output;
    cudaMalloc(&d_input, n * sizeof(float));
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {fprintf(stderr, "Erreur CUDA: %s\n", cudaGetErrorString(err)); return;}

    cudaMalloc(&d_output, n * sizeof(float));
    err = cudaGetLastError();
    if (err != cudaSuccess) {fprintf(stderr, "Erreur CUDA: %s\n", cudaGetErrorString(err)); return;}

    // Copier les données de l'hôte vers le device
    cudaMemcpy(d_input, input_ptr, n * sizeof(float), cudaMemcpyHostToDevice);
    err = cudaGetLastError();
    if (err != cudaSuccess) {fprintf(stderr, "Erreur CUDA: %s\n", cudaGetErrorString(err)); return;}

    // Lancer le SoftMax (expo() + réduction parrallele)
    dim3 dimGrid(1);
    dim3 dimBlock(threadsPerBlock);
    if (n > threadsPerBlock) {
        kernel_softmax_vec<<<dimGrid, dimBlock>>>(d_input, d_output, n);
    }
    else {
        kernel_softmax_vec_max1024<<<dimGrid, dimBlock, threadsPerBlock * sizeof(float)>>>(d_input, d_output, n);
        //kernel_softmax_vec<<<1, threadsPerBlock>>>(d_input, d_output, n);
    }
    err = cudaGetLastError();
    if (err != cudaSuccess) {fprintf(stderr, "Erreur CUDA: %s\n", cudaGetErrorString(err)); return;}

    // Copier le résultat du device vers l'hôte
    cudaMemcpy(output_ptr, d_output, n * sizeof(float), cudaMemcpyDeviceToHost);
    err = cudaGetLastError();
    if (err != cudaSuccess) {fprintf(stderr, "Erreur CUDA: %s\n", cudaGetErrorString(err)); return;}

    // Libérer la mémoire
    cudaFree(d_input);
    err = cudaGetLastError();
    if (err != cudaSuccess) {fprintf(stderr, "Erreur CUDA: %s\n", cudaGetErrorString(err)); return;}

    cudaFree(d_output);
    err = cudaGetLastError();
    if (err != cudaSuccess) {fprintf(stderr, "Erreur CUDA: %s\n", cudaGetErrorString(err)); return;}

    return;
}


__global__ void kernel_softmax_max1024(float *input, float *output, int N) {
    // Attention: Le nb de thread doit etre une puissance de 2 et supérieur à N

    extern __shared__ float sdata[];
    __shared__ float input_0;
    __shared__ float sum;
    __shared__ float max;

    unsigned int tid = threadIdx.x;  // Thread index
    unsigned int i   = threadIdx.x + blockIdx.x * N; // index global tableau

    // Recherche du max au sein du vecteur
    input_0 = input[0];
    sdata[tid] = (tid < N) ? input[i] : input_0;
    __syncthreads();


    // Calcul du max / Réduction parallèle / mémoire partagée
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            float a = sdata[tid];
            float b = sdata[tid + s];
            sdata[tid] = (b > a) ? b : a;
        }
        __syncthreads();
    }

    __syncthreads();
    if (tid == 0) {max = sdata[0]; }
    //if (tid == 0) {max = 0.; }
    __syncthreads();


    // Calcul du dénominateur (somme des expo)
    float expo = (tid < N) ? expf(input[i] - max) : 0;              // On applique l'exponentiellen
    sdata[tid] = (tid < N) ? expo : 0;    // sdata sera "détruit" par la réduction
    __syncthreads();

    // Réduction parallèle dans la mémoire partagée
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s]; }
        __syncthreads(); }
    //sdata[0] = 1.0;

    // Mettre la somme des expo dans la mémoire partagée
    if (tid == 0) { sum = sdata[0]; }
    //sdata[0] = 0.0;
    __syncthreads();

    //output[i] = (tid < N) ? expo / sum : 0;
    if (tid < N) {output[i] = expo / sum;};
    //output[i] = (tid < N) ? expo / sdata[0] : 0;
    //output[i] = 100 + i + 1;
    __syncthreads();
}



__global__ void kernel_softmax(float *input, float *output, int N) {
    // Attention: Le nb de thread doit etre une puissance de 2 et supérieur à N

    __shared__ float sdata[1024];
    __shared__ float input_0;
    __shared__ float sum;
    __shared__ float max;

    unsigned int tid = threadIdx.x;  // Thread index

    unsigned int n_iter_blocs = (N + blockDim.x - 1) / blockDim.x;

    // Recherche du max au sein du vecteur
    input_0 = input[0];
    max = input[0];
    __syncthreads();
    for (int bid = 0; bid < n_iter_blocs; bid++) {
        unsigned int i = blockDim.x * bid + tid;
        unsigned int j = blockIdx.x * N + i;
        sdata[tid] = (i < N) ? input[j] : input_0;
        __syncthreads();

        // Calcul du max / Réduction parallèle / mémoire partagée
        for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                float a = sdata[tid];
                float b = sdata[tid + s];
                sdata[tid] = (b > a) ? b : a;
            }
            __syncthreads();
        }

        __syncthreads();
        if (tid == 0) {
            float a = sdata[0];
            max = (a > max) ? a : max;
        }
        __syncthreads();
    }
    __syncthreads();


    // Calcul du dénominateur (somme des expo)
    sum = 0.0;
    __syncthreads();
    for (int bid = 0; bid < n_iter_blocs; bid++) {
        unsigned int i = blockDim.x * bid + tid;
        unsigned int j = blockIdx.x * N + i;
        sdata[tid] = (i < N) ? expf(input[j] - max) : 0.0;
        __syncthreads();

        // Réduction parallèle dans la mémoire partagée
        for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                sdata[tid] += sdata[tid + s]; }
            __syncthreads(); }

        __syncthreads();
        if (tid == 0) {sum += sdata[0];}
        __syncthreads();
    }
    __syncthreads();

    // A ce point, on connait le max des éléments, et la somme des exp.
    // On a tout ce qu'il faut pour écrire la sortie dans la mémoire générale du GPU.
    // Il aurait été possible de faire plus de caching (dynamic shared)
    // Mais cela aurait induit une limite maximale sur la taille du SoftMax (limite de la taille de la shared)
    for (int bid = 0; bid < n_iter_blocs; bid++) {
        unsigned int i = blockDim.x * bid + tid;
        unsigned int j = blockIdx.x * N + i;
        //output[j] = (i < N) ? expf(input[j] - max) / sum : 0.0;
        //output[j] = (i < N) ? 1.0 : 0.0;
        if (i < N) {output[j] = expf(input[j] - max) / sum;}
        //output[j] = 1.0;
    }

    __syncthreads();
}



void run_softmax(void *input_ptr, void *output_ptr, int n, int b) {

    // Attention: Le nb de thread doit etre une puissance de 2 et supérieur à N
    int threadsPerBlock = nextPowerOfTwo(n);
    if (threadsPerBlock > 1024) {
        threadsPerBlock = 1024;
    }

    std::cout << "Appel à run_softmax (n:" << n << ", b:" << b << ", threadsPerBlock:" << threadsPerBlock << ")" << std::endl;

    if ((b < 1) || (n < 1)) {
        std::cout << "Erreur: run_softmax - bad input" << std::endl;
        return; }

    float *d_input, *d_output;
    d_input  = (float *) input_ptr;   // Tenseurs déja sur le GPU: Pas d'alloc, pas de copie des données
    d_output = (float *) output_ptr;

    // Lancer le SoftMax (expo() + réduction parrallele)
    dim3 dimGrid(b);
    dim3 dimBlock(threadsPerBlock);
    if (n > threadsPerBlock) {
        std::cout << "Appel à kernel_softmax" << std::endl;
        kernel_softmax<<<dimGrid, dimBlock>>>(d_input, d_output, n);
    }
    else {
        std::cout << "Appel à kernel_softmax_max1024" << std::endl;
        kernel_softmax_max1024<<<dimGrid, dimBlock, threadsPerBlock * sizeof(float)>>>(d_input, d_output, n);
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {fprintf(stderr, "Erreur CUDA: %s\n", cudaGetErrorString(err)); return;}

    return;
}