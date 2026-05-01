#include <iostream>
#include <math.h>



__global__ void kernel_attn_cuda_batchqkv(
    float *q_ptr,
    float *k_ptr,
    float *v_ptr,
    float *o_ptr,
    int   n,
    int   d,
    int   bq,
    int   bk,
    int   *errorFlag) {
    
    // Example of values:  n = 1374, d = 64, bq = 60, bk = 25;

    extern __shared__ float sdata[]; // Allocation dynamique

    bool b_compute_prod_q_k    = true;
    bool b_compute_prod_attn_v = true;

    // Shared memory stores, in this order:
    // *  q_batch                    : bq * d
    // *  k_batch                    : bk * d
    // *  v_batch                    : bk * d
    // *  x_batch                    : bq * d
    // *  attn_batch                 : bq * bk
    // *  max of attn_batch          : bq
    // *  sum of SM(attn_batch)      : bq

    // so = shared offsets // sdata shared memory // (1 float nb = 4 bytes)
    //int dynamic_shared_size = 4 * (2 * d * bq + 2 * d * bk + bq * bk + 2 * bq);
    unsigned int so_q           = 0;                      // [bq, d]
    unsigned int so_k           = so_q        + bq * d;   // [bk, d]
    unsigned int so_v           = so_k        + bk * d;   // [bk, d]
    unsigned int so_x           = so_v        + bk * d;   // [bq, d]
    unsigned int so_attn        = so_x        + bq * d;   // [bq, bk]
    unsigned int so_max_attn    = so_attn     + bq * bk;  // [bq]
    unsigned int so_sum_sm_attn = so_max_attn + bq;       // [bq]

    // Thread and bloc index
    /* dim3 dimGrid(b, n_batch_q); dim3 dimBlock(threadsPerBlock); */
    unsigned int tid         = threadIdx.x;  // Thread index: Computations within a groud of (bq) lines of attn matrix
                                            // Internally sequential iterations over attn sub_blocs of size [bq, bk]
    unsigned int q_index     = blockIdx.y;   // q batching within the same attention matrix - independant computations
    unsigned int attn_index  = blockIdx.x;   // attn index (different attn heads or img batchs) - independant computations

    unsigned int n_threads   = blockDim.x;   // threadsPerBlock
    unsigned int n_batch_q   = (n + bq - 1) / bq;  // Number of batch of q  
    unsigned int n_elem_q    = n * d;        // For offset with attn_index (nb of elements in q, k, v, x ; without head multiplier)

    unsigned int bqc     = (q_index < (n_batch_q - 1)) ? bq : (n - (n_batch_q - 1) * bq); // bq_cour, which is equal to bq except for last q batch.

    // go = global offsets to VRAM for accessing q and x (not k or v)
    unsigned int go_q       = n_elem_q * attn_index + q_index * bq * d;

    unsigned int n_batch_k  = (n + bk - 1) / bk;  // Number of batch of k, v

    unsigned int index_max = 0; // Local computation - Multithreading bound limit
    unsigned int n_batch   = 0; // Local computation - Multithreading threads batchs



    // Copy q_batch from VRAM to shared, using batched threads
    index_max = bqc * d;   // q_batch has size bqc * d;
    n_batch = (index_max + n_threads - 1) / n_threads;
    for (unsigned int batch_index = 0; batch_index < n_batch; batch_index++) {
        unsigned int index = batch_index * n_threads + tid;
        if (index < index_max) {
            sdata[so_q + index] = q_ptr[go_q + index];
        }

    }
    __syncthreads();


    //////////////////////////////
    // MAX and SUM computations //

    // Reset MAX and SUM of SoftMax
    // Each thread resets one element of each array
    index_max = bqc;   // max and sum sizes
    if (index_max > n_threads) {*errorFlag = (! *errorFlag) ? 52 : *errorFlag; return;}
    if (tid < index_max) {
        sdata[so_max_attn    + tid] = __int_as_float(0xFF800000);   // -Infinity
        sdata[so_sum_sm_attn + tid] = 0.0;
    }
    __syncthreads();

    // The objective of this first loop is to feed the max_attn[] and sum_sm_attn[] arrays,
    // for the SoftMax computation.
    for (unsigned int k_index = 0; k_index < n_batch_k; k_index++) {
        unsigned int go_k       = n_elem_q * attn_index + k_index * bk * d;  // Valid for k, v, x

        unsigned int bkc     = (k_index < (n_batch_k - 1)) ? bk : (n - (n_batch_k - 1) * bk); // bk_cour, which is equal to bk except for last k batch.

        // Copy k_batch from VRAM to shared using batched threads
        index_max = bkc * d;   // k, v, x number of elements
        n_batch = (index_max + n_threads - 1) / n_threads;
        for (unsigned int batch_index = 0; batch_index < n_batch; batch_index++) {
            unsigned int index = batch_index * n_threads + tid;
            if (index < index_max) {
                sdata[so_k + index] = k_ptr[go_k + index];
            }
        }

        __syncthreads();

        //////////////////////////////////////////////////////////////////////////////////////
        // Objective: Feed max_attn and sum_sm_attn arrays, for future SoftMax computation. //
        //////////////////////////////////////////////////////////////////////////////////////

        // 1) Compute attn = q * k.T matrix product
        // Input:  sdata[]:  so_q [bq, d],  so_k [bk, d]
        // Output: sdata[]:  so_attn [bq, bk]
        if (b_compute_prod_q_k) {
            index_max = bqc * bkc;   // One (or few) elements of attn matrix per thread
            n_batch = (index_max + n_threads - 1) / n_threads;
            for (unsigned int batch_index = 0; batch_index < n_batch; batch_index++) {
                unsigned int index = batch_index * n_threads + tid;
                if (index < index_max) {
                    unsigned int qi = index / bkc;
                    unsigned int ki = index % bkc;
                    float s_cour = 0.0;
                    //float mycst = 3.14;
                    //unsigned int shift_q = so_q + qi * d;
                    //unsigned int shift_k = so_k + ki * d;
                    for (unsigned di = 0; di < d; di++) {
                        //s_cour += sdata[shift_q + di] * sdata[shift_k + di];
                        s_cour += sdata[so_q + qi * d + di] * sdata[so_k + ki * d + di];
                        //s_cour += mycst * sdata[so_k + ki * d + di];
                        //s_cour += mycst * mycst;
                        //s_cour += (mycst + (float) di) * (mycst + mycst * (float) di);
                        //s_cour += 1.0;
                        //s_cour += 3.14 * 5.67;
                    }
                    sdata[so_attn + qi * bkc + ki] = s_cour;
                }
            }
        }
        __syncthreads();

        index_max = bqc;
        n_batch = (index_max + n_threads - 1) / n_threads;
        for (unsigned int batch_index = 0; batch_index < n_batch; batch_index++) {
            unsigned int index = batch_index * n_threads + tid;
            if (index < index_max) {
                unsigned int qi = index;  index = 0;

                // 2) Compute the max per line of attn.
                // Input:  sdata[]:  so_attn [bq, bk]
                // Output: sdata[]:  so_max_cour_attn [bq]
                float max_cour = sdata[so_attn + qi * bkc + 0];
                for (unsigned ki = 1; ki < bkc; ki++) {
                    float max_new = sdata[so_attn + qi * bkc + ki];
                    max_cour = (max_new > max_cour) ? max_new : max_cour;
                }

                // 3) If max_cour > max:
                //   3a) adjust sum i.e. sum = sum * exp(max_cour - max)
                //   Input:  sdata[]:  so_max_cour_attn [bq],   so_max_attn [bq],  so_sum_sm_attn [bq]
                //   Output: sdata[]:  so_sum_sm_attn [bq]
                float max = sdata[so_max_attn + qi];
                float factor = (max_cour > max) ? expf(max - max_cour) : 1.0;
                sdata[so_sum_sm_attn + qi] *= factor;

                //   3b) update max: max = MAX(max, max_cour)
                //   Input:  sdata[]:  so_max_cour_attn [bq],  so_max_attn [bq]
                //   Output: sdata[]:  so_max_attn [bq]
                max = (max_cour > max) ? max_cour : max;
                sdata[so_max_attn + qi] = max;

                // 4) Compute the current sum_cour of exp(xi - max) of elements xi in so_attn, per line of attn
                // Input:  sdata[]:  so_attn [bq, bk],  so_max_attn [bq]
                // Output: sdata[]:  so_sum_cour_sm_attn [bq]
                float sum_cour = 0.0;
                for (unsigned ki = 0; ki < bkc; ki++) {
                    sum_cour += expf(sdata[so_attn + qi * bkc + ki] - max);
                }

                // 5) Add the current sum_cour to sum
                // Input:  sdata[]:  so_sum_sm_attn [bq],  so_sum_cour_sm_attn [bq]
                // Output: sdata[]:  so_sum_sm_attn [bq]
                sdata[so_sum_sm_attn + qi] += sum_cour;
            }

        }
        __syncthreads();
    }
    __syncthreads();
    
    // Here, the max_attn[] and sum_sm_attn[] arrays have been fed, and are ready for the SoftMax computation.

    // Reset x accumulators in shared memory
    index_max = bqc * d;   // x_batch has size bqc * d;
    n_batch = (index_max + n_threads - 1) / n_threads;
    for (unsigned int batch_index = 0; batch_index < n_batch; batch_index++) {
        unsigned int index = batch_index * n_threads + tid;
        if (index < index_max) {
            unsigned int qi = index;  index = 0;
            sdata[so_x + qi] = 0.0;
        }
    }
    __syncthreads();



    //////////////////////////////////////////////////////////////////////////////////////
    // Objective of this second loop: Compute the output array x, which is cached in the shared memory.
    //////////////////////////////////////////////////////////////////////////////////////

    // x is computed sequentially by accumulation, resulting of the batching of k and v.
    // In the details, the attention matrix attn is computed again, and this time, max and sum are known
    // for SoftMax computation, sm_attn. Then matrix product between sm_attn and v is performed, to feed the output.

    for (unsigned int k_index = 0; k_index < n_batch_k; k_index++) {
        unsigned int go_k       = n_elem_q * attn_index + k_index * bk * d;  // Valid for k, v, x

        unsigned int bkc     = (k_index < (n_batch_k - 1)) ? bk : (n - (n_batch_k - 1) * bk); // bk_cour, which is equal to bk except for last k batch.

        // Copy k_batch and v_batch from VRAM to shared using batched threads
        index_max = bkc * d;   // k, v, x number of elements
        n_batch = (index_max + n_threads - 1) / n_threads;
        for (unsigned int batch_index = 0; batch_index < n_batch; batch_index++) {
            unsigned int index = batch_index * n_threads + tid;
            if (index < index_max) {
                sdata[so_k + index] = k_ptr[go_k + index];
                sdata[so_v + index] = v_ptr[go_k + index];
            }
        }

        __syncthreads();

        index_max = bqc * bkc;   // 1 element of attn matrix per thread (roughly)
        n_batch = (index_max + n_threads - 1) / n_threads;
        for (unsigned int batch_index = 0; batch_index < n_batch; batch_index++) {
            unsigned int index = batch_index * n_threads + tid;
            if (index < index_max) {
                unsigned int qi = index / bkc;
                unsigned int ki = index % bkc;

                // 1) Compute attn = q * k.T matrix product
                // Input:  sdata[]:  so_q [bq, d],  so_k [bk, d]
                // Output: sdata[]:  so_attn [bq, bk]
                float xi = 0.0;
                if (b_compute_prod_q_k) {
                    float s_cour = 0.0;
                    for (unsigned di = 0; di < d; di++) {
                        s_cour += sdata[so_q + qi * d + di] * sdata[so_k + ki * d + di];
                    }
                    xi = s_cour;
                }

                // 2) Compute sm_attn = exp(xi - max) / sum, for xi in attn, line-wise
                // Input:  sdata[]:  so_attn [bq, bk],  so_max_attn [bq],  so_sum_sm_attn [bq]
                // Output: sdata[]:  so_attn [bq, bk]
                //s_cour contient xi, un élément de attn, résultat du produit q par k
                float max = sdata[so_max_attn    + qi]; // TODO: Couteux, utiliser 2 boucles imbriquées ?!
                float sum = sdata[so_sum_sm_attn + qi]; // TODO: Couteux, utiliser 2 boucles imbriquées ?!

                sdata[so_attn + qi * bkc + ki] = expf(xi - max) / sum; // SoftMax computation, for good.
            }
        }
        __syncthreads();


        index_max = bqc * d;   // One (or few) elements of attn matrix per thread
        n_batch = (index_max + n_threads - 1) / n_threads;
        for (unsigned int batch_index = 0; batch_index < n_batch; batch_index++) {
            unsigned int index = batch_index * n_threads + tid;
            if (index < index_max) {
                unsigned int qi = index / d;
                unsigned int di = index % d;

                // 3) compute x_cour = sm_attn * v  matrix product
                // Input:  sdata[]:  so_attn [bq, bk],  so_v [bk, d]
                // Output: sdata[]:  so_x_cour [bq, d]
                float so_x_cour = 0.0;
                if (b_compute_prod_attn_v) {
                    float s_cour = 0.0;
                    for (unsigned ki = 0; ki < bkc; ki++) {
                        s_cour += sdata[so_attn + qi * bkc + ki] * sdata[so_v + ki * d + di];
                    }
                    so_x_cour = s_cour;
                }

                // 4) Accumulate x : x = x + x_cour
                // Input:  sdata[]:  so_x [bq, d],  so_x_cour [bq, d]
                // Output: sdata[]:  so_x [bq, d]
                sdata[so_x + qi * d + di] += so_x_cour;
            }
        }

        //*errorFlag = -1; return;  // Arret pour debug

        __syncthreads();
    }

    // Eventually, copy the cached x in shared to the global VRAM pointer
    index_max = bqc * d;   // x_batch has size bqc * d;
    n_batch = (index_max + n_threads - 1) / n_threads;
    for (unsigned int batch_index = 0; batch_index < n_batch; batch_index++) {
        unsigned int index = batch_index * n_threads + tid;
        if (index < index_max) {
            o_ptr[go_q + index] = sdata[so_x + index];
            //o_ptr[go_q + index] = sdata[so_x + index] + 0.0001;
            //o_ptr[go_q + index] = sdata[so_x + index] + 3.141592;
        }
    }

    __syncthreads();

}


void run_attn_cuda(
    void *q_ptr,
    void *k_ptr,
    void *v_ptr,
    void *o_ptr,
    int   n,
    int   d,
    int   b,
    int attn_type) {

    std::cout << "Appel à attn.cu > run_attn_cuda ("
        << "n=" << n << ", "
        << "d=" << d << ", "
        << "b=" << b << ", "
        << "attn_type=" << attn_type 
        << ")" << std::endl;

    float *fq_ptr, *fk_ptr, *fv_ptr, *fo_ptr;
    fq_ptr  = (float *) q_ptr;   // Tenseurs déja sur le GPU: Pas d'alloc, pas de copie des données
    fk_ptr  = (float *) k_ptr;
    fv_ptr  = (float *) v_ptr;
    fo_ptr  = (float *) o_ptr;

    int *errorFlag;
    int h_errorFlag = 0;
    cudaMalloc(&errorFlag, sizeof(int));
    cudaMemcpy(errorFlag, &h_errorFlag, sizeof(int), cudaMemcpyHostToDevice);

    if (attn_type == 3) {
        int threadsPerBlock = 1024;

        // Hard limit for the shared memory is 48ko = 48*1024 = 49152
        // but we keep some space in case we want to cache some specific variables..
        unsigned int soft_limit = 49100;
        unsigned int hard_limit = 48 * 1024;

        //int bk = 5;
        int bk = 8;
        //int bk = 9;
        //int bk = 10;
        //int bk = 15;
        //int bk = 20;

        ///////////////
        // Automatically find the biggest value for bq, below shared soft_limit
        // If the size of the main shared array changes, dss_a and dss_b formulas must to be updated too.
        int dss_a = 4 * (2 * d + bk + 2);
        int dss_b = 4 * (2 * d * bk);
        int bq = (soft_limit - dss_b) / dss_a;
        int dynamic_shared_size = 4 * (2 * d * bq + 2 * d * bk + bq * bk + 2 * bq);
        ///////////////

        // Shared memory stores, in this order:
        // *  q_batch                    : bq * d
        // *  k_batch                    : bk * d
        // *  v_batch                    : bk * d
        // *  x_batch                    : bq * d
        // *  attn_batch                 : bq * bk
        // *  max of attn_batch          : bq
        // *  sum of SM(attn_batch)      : bq

        int n_batch_q = (n + bq - 1) / bq;
        float theo_speed_up = 3.0 / (2.0 / ((float) bq) + 2. / ((float) n));
        /*
            std::cout << "bq:" << bq 
                << ", bk:" << bk 
                << ", shared_size:" << dynamic_shared_size  << " bytes"
                << ", n_threads:" << threadsPerBlock
                << ", n_batch_q:" << n_batch_q
                << ", theo_speed_up:" << theo_speed_up
                << std::endl;
        */
        if (dynamic_shared_size >= soft_limit) {
            std::cout << "Erreur: dq, dk trop greedy: Ne peut allouer suffisamment de shared memory. "
                << "Demandé:" << dynamic_shared_size << "octets, "
                << " alors que soft limit à " << soft_limit << " et hard limit à " << hard_limit << ". "
                << "Réduire dq et/ou dk. "
                << "Arrêt préventif du codu CUDA, pour éviter un 'CUDA error: invalid argument'" << std::endl;
            return;
        }

        dim3 dimGrid(b, n_batch_q);  // Un bloc de threads calcule un batch de q   (bq lignes de la matrice d'attention)
        dim3 dimBlock(threadsPerBlock);
        //std::cout << "Appel à kernel_attn_cuda_batchqkv" << std::endl;
        kernel_attn_cuda_batchqkv<<<dimGrid, dimBlock, dynamic_shared_size>>>(fq_ptr, fk_ptr, fv_ptr, fo_ptr, n, d, bq, bk, errorFlag);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {fprintf(stderr, "Erreur CUDA: %s\n", cudaGetErrorString(err)); return;}

        cudaMemcpy(&h_errorFlag, errorFlag, sizeof(int), cudaMemcpyDeviceToHost);
        if (h_errorFlag != 0) {
            if (h_errorFlag < 0) {
                std::cout << "***** Debug / Arrêt manuel du kernel (valeur:" << h_errorFlag << ") *****" << std::endl; }
            else {
                std::cout << "***** Erreur rencontrée dans le kernel:" << h_errorFlag << "*****" << std::endl; }
        }
    }
    cudaFree(errorFlag);
}
