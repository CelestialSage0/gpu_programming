#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/binary_search.h>
#include <thrust/copy.h>

#define INF 2147483647 // simply INT_MAX
#define BLOCK_SIZE 256 // you can change it.

#define STATE_NONE 0
#define STATE_NEAR 1
#define STATE_FAR 2

// ============================================================================
// CUDA KERNELS
// ============================================================================

// ============================================================================
// SINGLE SOURCE DELTA-STEPPING DRIVER
// ============================================================================

void run_delta_stepping_single_source(
    /*




    */
)
{
    /*
    for adaptive delta, print from within this function. That would be easy.


    */
}

// ============================================================================
// MAIN FUNCTION
// ============================================================================

int main(int argc, char **argv)
{
    if (argc < 3)
    {
        std::cerr << "Usage: " << argv[0] << " <input_file> <output_file>\n";
        return 1;
    }

    std::ifstream infile(argv[1]);
    if (!infile.is_open())
    {
        std::cerr << "Error: Unable to open input file " << argv[1] << "\n";
        return 1;
    }

    std::ofstream outfile(argv[2]);
    if (!outfile.is_open())
    {
        std::cerr << "Error: Unable to open output file " << argv[2] << "\n";
        return 1;
    }

    int delta_mode, K, N, E, S_count;
    infile >> delta_mode >> K;
    infile >> N >> E >> S_count;

    std::vector<int> sources(S_count);
    for (int i = 0; i < S_count; ++i)
    {
        infile >> sources[i];
    }

    int *offsets = new int[N + 1]{0};
    for (int i = 0; i <= N; ++i)
    {
        infile >> offsets[i];
    }

    int *neighs = new int[E]{0};
    for (int i = 0; i < E; ++i)
    {
        infile >> neighs[i];
    }

    int *weights = new int[E]{0};
    for (int i = 0; i < E; ++i)
    {
        infile >> weights[i];
    }

    infile.close();

    int *d_offsets = nullptr, *d_neighs = nullptr, *d_weights = nullptr;
    cudaMalloc(&d_offsets, (N + 1) * sizeof(int));
    cudaMalloc(&d_neighs, E * sizeof(int));
    cudaMalloc(&d_weights, E * sizeof(int));

    cudaMemcpy(d_offsets, offsets, (N + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_neighs, neighs, E * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, weights, E * sizeof(int), cudaMemcpyHostToDevice);

    int *h_tent = new int[N]{0}; // sssp distance arry. tent means tentative distance
    int *d_tent = nullptr;
    cudaMalloc(&d_tent, N * sizeof(int));

    /* ToDo : BEGIN */

    int *d_state = nullptr;      // STATE_NONE / STATE_NEAR / STATE_FAR
    int *d_in_far = nullptr;     // 1 if vertex has an entry in the far array
    int *d_in_settled = nullptr; // 1 if vertex is already in the settled list
    cudaMalloc(&d_state, N * sizeof(int));
    cudaMalloc(&d_in_far, N * sizeof(int));
    cudaMalloc(&d_in_settled, N * sizeof(int));

    int *d_near_r = nullptr, *d_near_w = nullptr; // near queue: read side / write side (swapped each iteration)
    int *d_far_r = nullptr, *d_far_w = nullptr;   // far queue: read side / write side (swapped at each filter)
    int *d_settled = nullptr;                     // vertices processed in the current window
    cudaMalloc(&d_near_r, N * sizeof(int));
    cudaMalloc(&d_near_w, N * sizeof(int));
    cudaMalloc(&d_far_r, N * sizeof(int));
    cudaMalloc(&d_far_w, N * sizeof(int));
    cudaMalloc(&d_settled, N * sizeof(int));

    int *d_near_cnt = nullptr; // next-near count / promoted count
    int *d_far_cnt = nullptr;
    int *d_settled_cnt = nullptr;
    int *d_dmin = nullptr;
    cudaMalloc(&d_near_cnt, sizeof(int));
    cudaMalloc(&d_far_cnt, sizeof(int));
    cudaMalloc(&d_settled_cnt, sizeof(int));
    cudaMalloc(&d_dmin, sizeof(int));

    /* ToDo : END */

    for (int i = 0; i < S_count; ++i)
    {
        int source = sources[i];

        outfile << source << "\n";

        run_delta_stepping_single_source(
            /*
            ToDo






            */
        );

        cudaMemcpy(h_tent, d_tent, N * sizeof(int), cudaMemcpyDeviceToHost);

        for (int v = 0; v < N; ++v)
        {
            outfile << h_tent[v] << "\n";
        }
    }

    delete[] offsets;
    delete[] neighs;
    delete[] weights;

    // Added : BEGIN
    cudaFree(d_offsets);
    cudaFree(d_neighs);
    cudaFree(d_weights);
    cudaFree(d_tent);
    cudaFree(d_state);
    cudaFree(d_in_far);
    cudaFree(d_in_settled);
    cudaFree(d_near_r);
    cudaFree(d_near_w);
    cudaFree(d_far_r);
    cudaFree(d_far_w);
    cudaFree(d_settled);
    cudaFree(d_settled);
    cudaFree(d_near_cnt);
    cudaFree(d_far_cnt);
    cudaFree(d_settled_cnt);
    cudaFree(d_dmin);
    delete[] h_tent;
    // ADDED : END

    outfile.close();
    return 0;
}