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
__global__ void init_kernel(int *tent, int *state, int *in_far, int *in_settled, int N, int source)
{
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= N)
        return;

    tent[v] = (v == source) ? 0 : INF;
    state[v] = (v == source) ? STATE_NEAR : STATE_NONE;
    in_far[v] = 0;
    in_settled[v] = 0;
}

__global__ void light_kernel(const int *offsets, const int *neighs, const int *weights,
                             int *tent, int *state, int *in_far, int *in_settled,
                             const int *near_r, int near_cnt,
                             int *near_w, int *near_w_cnt,
                             int *far_q, int *far_cnt,
                             int *settled, int *settled_cnt,
                             int delta, int cutoff)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= near_cnt)
        return;

    int u = near_r[t];
    atomicExch(&state[u], STATE_NONE);
    int du = __ldcg(&tent[u]);

    if (atomicExch(&in_settled[u], 1) == 0)
        settled[atomicAdd(settled_cnt, 1)] = u;

    for (int e = offsets[u]; e < offsets[u + 1]; ++e)
    {
        int w = weights[e];
        if (w > delta)
            continue;

        int v = neighs[e];
        int nd = du + w;

        if (atomicMin(&tent[v], nd) > nd)
        {
            if (nd <= cutoff)
            {
                if (atomicExch(&state[v], STATE_NEAR) != STATE_NEAR)
                    near_w[atomicAdd(near_w_cnt, 1)] = v;
            }
            else
            {
                if (atomicCAS(&state[v], STATE_NONE, STATE_FAR) == STATE_NONE)
                    if (atomicExch(&in_far[v], 1) == 0)
                        far_q[atomicAdd(far_cnt, 1)] = v;
            }
        }
    }
}

// ============================================================================
// SINGLE SOURCE DELTA-STEPPING DRIVER
// ============================================================================

void run_delta_stepping_single_source(
    const int *d_offsets, const int *d_neighs, const int *d_weights,
    int N, int source, int delta,
    int *d_tent, int *d_state, int *d_in_far, int *d_in_settled,
    int *d_near_r, int *d_near_w, int *d_far_r, int *d_far_w, int *d_settled,
    int *d_near_w_cnt, int *d_far_cnt, int *d_settled_cnt, int *d_dmin)
{
    /*
    for adaptive delta, print from within this function. That would be easy.*/
    int grid_n = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    init_kernel<<<grid_n, BLOCK_SIZE>>>(d_tent, d_state, d_in_far, d_in_settled, N, source);

    cudaMemset(d_near_w_cnt, 0, sizeof(int));
    cudaMemset(d_far_cnt, 0, sizeof(int));
    cudaMemset(d_settled_cnt, 0, sizeof(int));

    cudaMemcpy(d_near_r, &source, sizeof(int), cudaMemcpyHostToDevice);
    int near_cnt = 1;

    int cutoff = (delta >= INF) ? INF - 1 : delta;

    while (near_cnt > 0)
    {
        cudaMemset(d_near_w_cnt, 0, sizeof(int));

        int grid = (near_cnt + BLOCK_SIZE - 1) / BLOCK_SIZE;
        light_kernel<<<grid, BLOCK_SIZE>>>(
            d_offsets, d_neighs, d_weights,
            d_tent, d_state, d_in_far, d_in_settled,
            d_near_r, near_cnt,
            d_near_w, d_near_w_cnt,
            d_far_r, d_far_cnt,
            d_settled, d_settled_cnt,
            delta, cutoff);

        cudaMemcpy(&near_cnt, d_near_w_cnt, sizeof(int), cudaMemcpyDeviceToHost);
        std::swap(d_near_r, d_near_w); // what we just wrote is what we read next
    }
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

    int *d_state = nullptr; // STATE_NONE / STATE_NEAR / STATE_FAR
    int *d_in_far = nullptr;
    int *d_in_settled = nullptr;
    cudaMalloc(&d_state, N * sizeof(int));
    cudaMalloc(&d_in_far, N * sizeof(int));
    cudaMalloc(&d_in_settled, N * sizeof(int));

    int *d_near_r = nullptr, *d_near_w = nullptr;
    int *d_far_r = nullptr, *d_far_w = nullptr;
    int *d_settled = nullptr;
    cudaMalloc(&d_near_r, N * sizeof(int));
    cudaMalloc(&d_near_w, N * sizeof(int));
    cudaMalloc(&d_far_r, N * sizeof(int));
    cudaMalloc(&d_far_w, N * sizeof(int));
    cudaMalloc(&d_settled, N * sizeof(int));

    int *d_near_cnt = nullptr;
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
            d_offsets, d_neighs, d_weights,
            N, source, delta_mode,
            d_tent, d_state, d_in_far, d_in_settled,
            d_near_r, d_near_w, d_far_r, d_far_w, d_settled,
            d_near_cnt, d_far_cnt, d_settled_cnt, d_dmin);

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
    cudaFree(d_near_cnt);
    cudaFree(d_far_cnt);
    cudaFree(d_settled_cnt);
    cudaFree(d_dmin);
    delete[] h_tent;
    // ADDED : END

    outfile.close();
    return 0;
}