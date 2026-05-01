#include <iostream>
#include <math.h>

typedef struct {
    int source_h;
    int source_w;
    int target_h;
    int target_w;
    float slope_h;
    float slope_w;
    float offset_h;
    float offset_w; 
} RegriddingGeometry;


// 1 bloc de 1024 threads travaillent sur 1 seule couche de l'image
__global__ void kernel_interpolate_2d_nearest(
        float *input,
        float *output,
        RegriddingGeometry geo) {

    int     source_h = geo.source_h;
    int     source_w = geo.source_w;
    int     target_h = geo.target_h;
    int     target_w = geo.target_w;
    float   slope_h  = geo.slope_h;
    float   slope_w  = geo.slope_w;
    float   offset_h = geo.offset_h;
    float   offset_w = geo.offset_w;

    float n_pixels = target_h * target_w;
    float batch_size = (n_pixels + blockDim.x - 1) / blockDim.x;

    unsigned int tid = threadIdx.x;                  // Thread index
    unsigned int bloc_target_offset = blockIdx.x * n_pixels; // index global tableau
    unsigned int bloc_source_offset = blockIdx.x * source_h * source_w; // index global tableau

    // Pixel index treated by the current thread
    unsigned int index_min = batch_size * tid;
    unsigned int index_max = batch_size * (tid + 1);
    if (index_max >= n_pixels) {index_max = n_pixels - 1;}

    int target_h_index_prec = -1;
    float f_source_h_index = -1.0;
    int source_h_index = -1;
    int source_offset = -1;
    int target_offset = -1;
    // Loop over pixels
    for (int i = index_min; i < index_max; i++) {

        // Copy nearest pixel

        // Some thread-level caching
        int target_h_index = i / target_w;
        if (target_h_index != target_h_index_prec) {
            f_source_h_index = offset_h + ((float) target_h_index) * slope_h;
            source_h_index = round(f_source_h_index);
            source_offset = bloc_source_offset + source_h_index * source_w;
            target_offset = bloc_target_offset + target_h_index * target_w;
            target_h_index_prec = target_h_index;
        }

        int target_w_index = i % target_w;
        float f_source_w_index = offset_w + ((float) target_w_index) * slope_w;
        int source_w_index = round(f_source_w_index);

        unsigned int source_index = source_offset + source_w_index;
        unsigned int target_index = target_offset + target_w_index;

        output[target_index] = input[source_index];
    }

}


// 1 bloc de 1024 threads travaillent sur 1 seule couche de l'image
__global__ void kernel_interpolate_2d_bilinear(
        float *input,
        float *output,
        RegriddingGeometry geo) {

    int     source_h = geo.source_h;
    int     source_w = geo.source_w;
    int     target_h = geo.target_h;
    int     target_w = geo.target_w;
    float   slope_h  = geo.slope_h;
    float   slope_w  = geo.slope_w;
    float   offset_h = geo.offset_h;
    float   offset_w = geo.offset_w;

    float n_pixels = target_h * target_w;
    float batch_size = (n_pixels + blockDim.x - 1) / blockDim.x;

    unsigned int tid = threadIdx.x;                  // Thread index
    unsigned int bloc_target_offset = blockIdx.x * n_pixels; // index global tableau
    unsigned int bloc_source_offset = blockIdx.x * source_h * source_w; // index global tableau

    // Pixel index treated by the current thread
    unsigned int index_min = batch_size * tid;
    unsigned int index_max = batch_size * (tid + 1);
    if (index_max >= n_pixels) {index_max = n_pixels - 1;}

    int target_h_index_prec = -1;
    float f_source_h_index = -1.0;
    int source_h1_index = -1;
    int source_h2_index = -1;
    int source_h1_offset = -1;
    int source_h2_offset = -1;
    int target_offset = -1;
    // Loop over pixels
    for (int i = index_min; i < index_max; i++) {

        // Interpolate over the 4 neighboring pixels

        // Some thread-level caching
        int target_h_index = i / target_w;
        if (target_h_index != target_h_index_prec) {
            f_source_h_index = offset_h + ((float) target_h_index) * slope_h;
            //source_h_index = round(f_source_h_index);
            source_h1_index = (int) floor(f_source_h_index);
            source_h2_index = (int) ceil(f_source_h_index);
            if (source_h1_index < 0) {
                source_h1_index = 0; source_h2_index = 1;
                f_source_h_index = (f_source_h_index > source_h1_index) ? f_source_h_index : (float) source_h1_index;}
            if (source_h2_index >= source_h) {
                source_h1_index = source_h - 2; source_h2_index = source_h - 1;
                f_source_h_index = (f_source_h_index < source_h2_index) ? f_source_h_index : (float) source_h2_index;}

            source_h1_offset = bloc_source_offset + source_h1_index * source_w;
            source_h2_offset = bloc_source_offset + source_h2_index * source_w;
            target_offset = bloc_target_offset + target_h_index * target_w;
            target_h_index_prec = target_h_index;
        }

        int target_w_index = i % target_w;
        float f_source_w_index = offset_w + ((float) target_w_index) * slope_w;
        //int source_w_index = round(f_source_w_index);
        int source_w1_index = (int) floor(f_source_w_index);
        int source_w2_index = (int) ceil(f_source_w_index);
        if (source_w1_index < 0) {
            source_w1_index = 0; source_w2_index = 1;
            f_source_w_index = (f_source_w_index > (float) source_w1_index) ? f_source_w_index : (float) source_w1_index;}
        if (source_w2_index >= source_w) {
            source_w1_index = source_w - 2; source_w2_index = source_w - 1;
            f_source_w_index = (f_source_w_index < (float) source_w2_index) ? f_source_w_index : (float) source_w2_index;}

        unsigned int source_h1w1_index = source_h1_offset + source_w1_index;
        unsigned int source_h2w1_index = source_h2_offset + source_w1_index;
        unsigned int source_h1w2_index = source_h1_offset + source_w2_index;
        unsigned int source_h2w2_index = source_h2_offset + source_w2_index;
        float f_h1w1 = input[source_h1w1_index];
        float f_h2w1 = input[source_h2w1_index];
        float f_h1w2 = input[source_h1w2_index];
        float f_h2w2 = input[source_h2w2_index];

        float weight_h = f_source_h_index - (float) source_h1_index;
        float weight_w = f_source_w_index - (float) source_w1_index;

        float output_val = f_h1w1
                + (f_h1w2 - f_h1w1) * weight_w
                + (f_h2w1 - f_h1w1) * weight_h
                + (f_h1w1 + f_h2w2 - f_h1w2 - f_h2w1) * weight_w * weight_h;

        unsigned int target_index = target_offset + target_w_index;

        output[target_index] = output_val;
    }

}


// 1 bloc de 1024 threads travaillent sur 1 seule couche de l'image
__global__ void kernel_interpolate_2d_box(
        float *input,
        float *output,
        RegriddingGeometry geo) {

    int     source_h = geo.source_h;
    int     source_w = geo.source_w;
    int     target_h = geo.target_h;
    int     target_w = geo.target_w;
    float   slope_h  = geo.slope_h;
    float   slope_w  = geo.slope_w;
    float   offset_h = geo.offset_h;
    float   offset_w = geo.offset_w;

    float n_pixels = target_h * target_w;
    float batch_size = (n_pixels + blockDim.x - 1) / blockDim.x;

    unsigned int tid = threadIdx.x;                  // Thread index
    unsigned int bloc_target_offset = blockIdx.x * n_pixels; // index global tableau
    unsigned int bloc_source_offset = blockIdx.x * source_h * source_w; // index global tableau

    // Pixel index treated by the current thread
    unsigned int index_min = batch_size * tid;
    unsigned int index_max = batch_size * (tid + 1);
    if (index_max >= n_pixels) {index_max = n_pixels - 1;}

    // Loop over pixels
    for (int i = index_min; i < index_max; i++) {

        // Apply the Box filter, on both axis (See: https://legacy.imagemagick.org/Usage/filter/#triangle)

        int   target_h_index   = i / target_w;
        int   target_w_index   = i % target_w;
        float f_source_h_index = offset_h + ((float) target_h_index) * slope_h;
        float f_source_w_index = offset_w + ((float) target_w_index) * slope_w;

        int   target_offset = bloc_target_offset + target_h_index * target_w;


        float f_min_h = f_source_h_index - slope_h * 0.5;
        if (f_min_h < 0.0) {f_min_h = 0.0;}
        float f_max_h = f_source_h_index + slope_h * 0.5;
        if (f_max_h > source_h - 1) {f_max_h = source_h - 1;}
        float f_min_w = f_source_w_index - slope_w * 0.5;
        if (f_min_w < 0.0) {f_min_w = 0.0;}
        float f_max_w = f_source_w_index + slope_w * 0.5;
        if (f_max_w > source_w - 1) {f_max_w = source_w - 1;}
        int min_h = round(f_min_h);
        int max_h = round(f_max_h);
        int min_w = round(f_min_w);
        int max_w = round(f_max_w);

        float sum_weights = 0.0;
        float sum_values  = 0.0;
        for (int j = min_h; j <= max_h; j++) {
            for (int k = min_w; k <= max_w; k++) {
                int source_h_index = j;
                int source_w_index = k;
                int source_h_offset = bloc_source_offset + source_h_index * source_w;
                unsigned int source_hw_index = source_h_offset + source_w_index;

                float weight_cour = 1.0;
                float value_cour = weight_cour * input[source_hw_index];
                sum_weights += weight_cour;
                sum_values  += value_cour;

            }
        }

        unsigned int target_index = target_offset + target_w_index;
        output[target_index] = sum_values / sum_weights;
    }

}


// 1 bloc de 1024 threads travaillent sur 1 seule couche de l'image
__global__ void kernel_interpolate_2d_triangle(
        float *input,
        float *output,
        RegriddingGeometry geo) {

    int     source_h = geo.source_h;
    int     source_w = geo.source_w;
    int     target_h = geo.target_h;
    int     target_w = geo.target_w;
    float   slope_h  = geo.slope_h;
    float   slope_w  = geo.slope_w;
    float   offset_h = geo.offset_h;
    float   offset_w = geo.offset_w;

    float n_pixels = target_h * target_w;
    float batch_size = (n_pixels + blockDim.x - 1) / blockDim.x;

    unsigned int tid = threadIdx.x;                  // Thread index
    unsigned int bloc_target_offset = blockIdx.x * n_pixels; // index global tableau
    unsigned int bloc_source_offset = blockIdx.x * source_h * source_w; // index global tableau

    // Pixel index treated by the current thread
    unsigned int index_min = batch_size * tid;
    unsigned int index_max = batch_size * (tid + 1);
    if (index_max >= n_pixels) {index_max = n_pixels - 1;}

    // Loop over pixels
    for (int i = index_min; i < index_max; i++) {

        // Apply the triangle filter, on both axis (See: https://legacy.imagemagick.org/Usage/filter/#triangle)

        int   target_h_index   = i / target_w;
        int   target_w_index   = i % target_w;
        float f_source_h_index = offset_h + ((float) target_h_index) * slope_h;
        float f_source_w_index = offset_w + ((float) target_w_index) * slope_w;

        int   target_offset = bloc_target_offset + target_h_index * target_w;


        float f_min_h = f_source_h_index - slope_h;
        if (f_min_h < 0.0) {f_min_h = 0.0;}
        float f_max_h = f_source_h_index + slope_h;
        if (f_max_h > source_h - 1) {f_max_h = source_h - 1;}
        float f_min_w = f_source_w_index - slope_w;
        if (f_min_w < 0.0) {f_min_w = 0.0;}
        float f_max_w = f_source_w_index + slope_w;
        if (f_max_w > source_w - 1) {f_max_w = source_w - 1;}
        int min_h = round(f_min_h);
        int max_h = round(f_max_h);
        int min_w = round(f_min_w);
        int max_w = round(f_max_w);

        float sum_weights = 0.0;
        float sum_values  = 0.0;
        for (int j = min_h; j <= max_h; j++) {
            int source_h_index = j;
            float dh = 1.0 - fabs((float) source_h_index - f_source_h_index) / slope_h;
            //float dh = fabs((float) source_h_index - f_source_h_index) / slope_h;
            for (int k = min_w; k <= max_w; k++) {
                int source_w_index = k;
                float dw = 1.0 - fabs((float) source_w_index - f_source_w_index) / slope_w;
                //float dw = fabs((float) source_w_index - f_source_w_index) / slope_w;
                int source_h_offset = bloc_source_offset + source_h_index * source_w;
                unsigned int source_hw_index = source_h_offset + source_w_index;

                float weight_cour = dh * dw;
                //float weight_cour = sqrt(dh * dh + dw * dw);
                //float weight_cour = 1.0 - sqrt(dh * dh + dw * dw);
                if (weight_cour < 0.0) {weight_cour = 0.0;}
                float value_cour = weight_cour * input[source_hw_index];
                sum_weights += weight_cour;
                sum_values  += value_cour;
            }
        }

        unsigned int target_index = target_offset + target_w_index;
        output[target_index] = sum_values / sum_weights;
    }

}

#define PI 3.14159265358979323846


// Fonction pour calculer le polynôme cubique
__device__ double cubic_polynomial(double x) {
    double abs_x = fabs(x);
    if (abs_x <= 1.0) {
        return 1.0 - 2.0 * abs_x * abs_x + abs_x * abs_x * abs_x;
    } else if (abs_x < 2.0) {
        return 4.0 - 8.0 * abs_x + 5.0 * abs_x * abs_x - abs_x * abs_x * abs_x;
    } else {
        return 0.0;
    }
}

// 1 bloc de 1024 threads travaillent sur 1 seule couche de l'image
__global__ void kernel_interpolate_2d_bicubic(
        float *input,
        float *output,
        RegriddingGeometry geo) {

    int     source_h = geo.source_h;
    int     source_w = geo.source_w;
    int     target_h = geo.target_h;
    int     target_w = geo.target_w;
    float   slope_h  = geo.slope_h;
    float   slope_w  = geo.slope_w;
    float   offset_h = geo.offset_h;
    float   offset_w = geo.offset_w;

    float n_pixels = target_h * target_w;
    float batch_size = (n_pixels + blockDim.x - 1) / blockDim.x;

    unsigned int tid = threadIdx.x;                  // Thread index
    unsigned int bloc_target_offset = blockIdx.x * n_pixels; // index global tableau
    unsigned int bloc_source_offset = blockIdx.x * source_h * source_w; // index global tableau

    // Pixel index treated by the current thread
    unsigned int index_min = batch_size * tid;
    unsigned int index_max = batch_size * (tid + 1);
    if (index_max >= n_pixels) {index_max = n_pixels - 1;}

    // Loop over pixels
    for (int p = index_min; p < index_max; p++) {

        // Apply the bicubic filter, on both axis (See: https://legacy.imagemagick.org/Usage/filter/)
        int   y   = p / target_w;
        int   x   = p % target_w;
        int target_offset = bloc_target_offset + y * target_w;

        double src_x = (x + 0.5) * slope_w - 0.5;
        double src_y = (y + 0.5) * slope_h - 0.5;

        int x0 = (int)floor(src_x);
        int y0 = (int)floor(src_y);

        double sum_values = 0.0;
        double sum_weights = 0.0;

        for (int j = -1; j <= 2; j++) {
            for (int i = -1; i <= 2; i++) {
                int x_sample = x0 + i;
                int y_sample = y0 + j;

                if (x_sample >= 0 && x_sample < source_w && y_sample >= 0 && y_sample < source_h) {
                    double weight_x = cubic_polynomial(src_x - (x_sample + 0.5));
                    double weight_y = cubic_polynomial(src_y - (y_sample + 0.5));
                    double weight = weight_x * weight_y;

                    int source_h_offset = bloc_source_offset + y_sample * source_w;
                    unsigned int source_hw_index = source_h_offset + x_sample;

                    sum_values += input[source_hw_index] * weight;
                    sum_weights += weight;
                }
            }
        }

        unsigned int target_index = target_offset + x;
        output[target_index] = sum_values / sum_weights;
    }

}


__device__ double sinc(double x) {
    if (x == 0.0) {
        return 1.0;
    } else {
        return sin(PI * x) / (PI * x);
    }
}

__device__ double lanczos(double x, double a) {
    if (x == 0.0) {
        return 1.0;
    } else if (fabs(x) < a) {
        return sinc(x) * sinc(x / a);
    } else {
        return 0.0;
    }
}



// 1 bloc de 1024 threads travaillent sur 1 seule couche de l'image
__global__ void kernel_interpolate_2d_lanczos(
        float *input,
        float *output,
        RegriddingGeometry geo) {

    const int a = 3; // Lanczos3

    int     source_h = geo.source_h;
    int     source_w = geo.source_w;
    int     target_h = geo.target_h;
    int     target_w = geo.target_w;
    float   slope_h  = geo.slope_h;
    float   slope_w  = geo.slope_w;
    float   offset_h = geo.offset_h;
    float   offset_w = geo.offset_w;

    float n_pixels = target_h * target_w;
    float batch_size = (n_pixels + blockDim.x - 1) / blockDim.x;

    unsigned int tid = threadIdx.x;                  // Thread index
    unsigned int bloc_target_offset = blockIdx.x * n_pixels; // index global tableau
    unsigned int bloc_source_offset = blockIdx.x * source_h * source_w; // index global tableau

    // Pixel index treated by the current thread
    unsigned int index_min = batch_size * tid;
    unsigned int index_max = batch_size * (tid + 1);
    if (index_max >= n_pixels) {index_max = n_pixels - 1;}

    // Loop over pixels
    for (int p = index_min; p < index_max; p++) {

        // Apply the Lanczos filter, on both axis (See: https://legacy.imagemagick.org/Usage/filter/)

        int   y   = p / target_w;
        int   x   = p % target_w;
        int target_offset = bloc_target_offset + y * target_w;

        double center_x = (x + 0.5) * slope_w;
        double center_y = (y + 0.5) * slope_h;

        double sum_weights = 0.0;
        double sum_values = 0.0;

        for (int i = -((int)a); i <= (int)a; i++) {
            for (int j = -((int)a); j <= (int)a; j++) {
                double px = center_x + i;
                double py = center_y + j;

                if (px >= 0.0 && px < source_w && py >= 0.0 && py < source_h) {
                    double weight_x = lanczos(center_x - px, a);
                    double weight_y = lanczos(center_y - py, a);
                    double weight = weight_x * weight_y;

                    int source_h_offset = bloc_source_offset + py * source_w;
                    unsigned int source_hw_index = source_h_offset + py;

                    sum_values += input[source_hw_index] * weight;
                    sum_weights += weight;
                }
            }
        }

        unsigned int target_index = target_offset + x;
        output[target_index] = sum_values / sum_weights;
    }

}






// TODO: Implémenter le triangle
// https://legacy.imagemagick.org/Usage/filter/#triangle

void run_interpolate_2d(
        void *input_ptr,
        void *output_ptr,
        int source_h,
        int source_w,
        int target_h,
        int target_w,
        int interp_type,
        int b)
{
    //std::cout << "Appel à run_interpolate_2d" << std::endl;

    if (target_h != target_w) {
        std::cout << "Erreur dans run_interpolate_2d: Pas implémenté pour resize vers images non carrées" << std::endl;
        return;
    }

    float *d_input, *d_output;
    d_input  = (float *) input_ptr;   // Tenseurs déja sur le GPU: Pas d'alloc, pas de copie des données
    d_output = (float *) output_ptr;

    // Calculs géométriques pour le regridding
    bool min_is_h = source_h < source_w;
    int source_min = min_is_h ? source_h : source_w;

    float slope = (((float) source_min) - 1.) / (((float) target_h) - 1.); // Même ratio pour height et width -> preserve image ratio
    float offset = fabs(0.5 * (float) (source_h - source_w));
    float offset_h = min_is_h ?      0 : offset ;
    float offset_w = min_is_h ? offset :      0 ;

    RegriddingGeometry geo;
    geo.source_h = source_h;
    geo.source_w = source_w;
    geo.target_h = target_h;
    geo.target_w = target_w;
    geo.slope_h = slope;
    geo.slope_w = slope;
    geo.offset_h = offset_h;
    geo.offset_w = offset_w;

    int threadsPerBlock = 1024;
    dim3 dimGrid(b);
    dim3 dimBlock(threadsPerBlock);
    cudaError_t err;

    switch (interp_type) {
        case 0:
            // Valeur spéciale, pas de calcul d'interpolation, renvoie une image noire
            return;


        case 1:
            // Resize / crop / nearest
            //std::cout << "Appel à kernel_interpolate_2d_nearest" << std::endl;
            kernel_interpolate_2d_nearest<<<dimGrid, dimBlock>>>(d_input, d_output, geo);
            err = cudaGetLastError();
            if (err != cudaSuccess) {fprintf(stderr, "Erreur CUDA: %s\n", cudaGetErrorString(err)); return;}
            return;

        case 2:
            // Resize / crop / bilinear
            // Interpolation bilinéaire avec crop des bords de l'image, du côté long
            //std::cout << "Appel à kernel_interpolate_2d_bilinear" << std::endl;
            kernel_interpolate_2d_bilinear<<<dimGrid, dimBlock>>>(d_input, d_output, geo);
            err = cudaGetLastError();
            if (err != cudaSuccess) {fprintf(stderr, "Erreur CUDA: %s\n", cudaGetErrorString(err)); return;}
            return;

        case 3:
            // Resize / crop / Box averaging filter
            //std::cout << "Appel à kernel_interpolate_2d_box" << std::endl;
            kernel_interpolate_2d_box<<<dimGrid, dimBlock>>>(d_input, d_output, geo);
            err = cudaGetLastError();
            if (err != cudaSuccess) {fprintf(stderr, "Erreur CUDA: %s\n", cudaGetErrorString(err)); return;}
            return;

        case 4:
            // Resize / crop / bilinear with Triangle
            // Interpolation bilinéaire méthode Triangle avec crop des bords de l'image, du côté long
            //std::cout << "Appel à kernel_interpolate_2d_triangle" << std::endl;
            kernel_interpolate_2d_triangle<<<dimGrid, dimBlock>>>(d_input, d_output, geo);
            err = cudaGetLastError();
            if (err != cudaSuccess) {fprintf(stderr, "Erreur CUDA: %s\n", cudaGetErrorString(err)); return;}
            return;

        case 5:
            // Resize / crop / bilinear with Lanczos
            kernel_interpolate_2d_bicubic<<<dimGrid, dimBlock>>>(d_input, d_output, geo);
            err = cudaGetLastError();
            if (err != cudaSuccess) {fprintf(stderr, "Erreur CUDA: %s\n", cudaGetErrorString(err)); return;}
            return;


        case 6:
            // Resize / crop / bilinear with Lanczos
            kernel_interpolate_2d_lanczos<<<dimGrid, dimBlock>>>(d_input, d_output, geo);
            err = cudaGetLastError();
            if (err != cudaSuccess) {fprintf(stderr, "Erreur CUDA: %s\n", cudaGetErrorString(err)); return;}
            return;



        default:
            std::cout << "Erreur dans run_interpolate_2d: Type d'interpolation non reconnu." << std::endl;
    }


}
