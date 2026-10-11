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

__global__ void heavy_kernel(const int *offsets, const int *neighs, const int *weights,
                             int *tent, int *state, int *in_far, int *in_settled,
                             const int *settled, int settled_cnt,
                             int *far_q, int *far_cnt, int delta)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= settled_cnt)
        return;

    int u = settled[t];
    int du = tent[u];
    in_settled[u] = 0;

    for (int e = offsets[u]; e < offsets[u + 1]; ++e)
    {
        int w = weights[e];
        if (w <= delta)
            continue;

        int v = neighs[e];
        int nd = du + w;

        if (atomicMin(&tent[v], nd) > nd)
        {
            if (atomicCAS(&state[v], STATE_NONE, STATE_FAR) == STATE_NONE)
                if (atomicExch(&in_far[v], 1) == 0)
                    far_q[atomicAdd(far_cnt, 1)] = v;
        }
    }
}

__global__ void min_far_kernel(const int *far_q, int far_cnt, const int *tent, const int *state, int *dmin)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= far_cnt)
        return;

    int v = far_q[t];
    if (state[v] == STATE_FAR)
        atomicMin(dmin, tent[v]);
}

__global__ void filter_kernel(const int *far_in, int far_cnt,
                              int *far_out, int *far_out_cnt,
                              int *near_out, int *near_out_cnt,
                              const int *tent, int *state, int *in_far, int cutoff)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= far_cnt)
        return;

    int v = far_in[t];

    if (state[v] == STATE_FAR)
    {
        if (tent[v] <= cutoff)
        {
            state[v] = STATE_NEAR;
            in_far[v] = 0;
            near_out[atomicAdd(near_out_cnt, 1)] = v;
        }
        else
        {
            far_out[atomicAdd(far_out_cnt, 1)] = v;
        }
    }
    else
    {
        in_far[v] = 0;
    }
}

__global__ void compact_far_kernel(const int *far_in, int far_cnt,
                                   int *far_out, int *keys_out, int *far_out_cnt,
                                   const int *tent, const int *state, int *in_far)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= far_cnt)
        return;

    int v = far_in[t];
    if (state[v] == STATE_FAR)
    {
        int p = atomicAdd(far_out_cnt, 1);
        far_out[p] = v;
        keys_out[p] = tent[v];
    }
    else
    {
        in_far[v] = 0;
    }
}

__global__ void promote_kernel(const int *sorted_far, int I, int *near_out, int *state, int *in_far)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= I)
        return;

    int v = sorted_far[t];
    state[v] = STATE_NEAR;
    in_far[v] = 0;
    near_out[t] = v;
}

// ============================================================================
// SINGLE SOURCE DELTA-STEPPING DRIVER
// ============================================================================

void run_delta_stepping_single_source(
    const int *d_offsets, const int *d_neighs, const int *d_weights,
    int N, int source, int delta,
    int *d_tent, int *d_state, int *d_in_far, int *d_in_settled,
    int *d_near_r, int *d_near_w, int *d_far_r, int *d_far_w, int *d_settled,
    int *d_near_w_cnt, int *d_far_cnt, int *d_settled_cnt, int *d_dmin,
    int K, std::ofstream &outfile, int *d_far_keys)
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
    int cur_delta = (delta < 0) ? 0 : delta;

    int cutoff = (cur_delta >= INF) ? INF - 1 : cur_delta;

    while (true)
    {
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
                cur_delta, cutoff);

            cudaMemcpy(&near_cnt, d_near_w_cnt, sizeof(int), cudaMemcpyDeviceToHost);
            std::swap(d_near_r, d_near_w);
        }

        int settled_cnt;
        cudaMemcpy(&settled_cnt, d_settled_cnt, sizeof(int), cudaMemcpyDeviceToHost);
        if (settled_cnt > 0)
        {
            int grid = (settled_cnt + BLOCK_SIZE - 1) / BLOCK_SIZE;
            heavy_kernel<<<grid, BLOCK_SIZE>>>(
                d_offsets, d_neighs, d_weights,
                d_tent, d_state, d_in_far, d_in_settled,
                d_settled, settled_cnt,
                d_far_r, d_far_cnt, cur_delta);
            cudaMemset(d_settled_cnt, 0, sizeof(int));
        }

        int far_cnt;
        cudaMemcpy(&far_cnt, d_far_cnt, sizeof(int), cudaMemcpyDeviceToHost);
        if (far_cnt == 0)
            break;

        if (delta >= 0)
        {
            int dmin = INF;
            cudaMemcpy(d_dmin, &dmin, sizeof(int), cudaMemcpyHostToDevice);
            int grid = (far_cnt + BLOCK_SIZE - 1) / BLOCK_SIZE;
            min_far_kernel<<<grid, BLOCK_SIZE>>>(d_far_r, far_cnt, d_tent, d_state, d_dmin);
            cudaMemcpy(&dmin, d_dmin, sizeof(int), cudaMemcpyDeviceToHost);
            if (dmin == INF)
                break;

            long long c = (long long)dmin + delta;
            cutoff = (c >= INF) ? INF - 1 : (int)c;

            cudaMemset(d_near_w_cnt, 0, sizeof(int));
            cudaMemset(d_far_cnt, 0, sizeof(int));
            filter_kernel<<<grid, BLOCK_SIZE>>>(
                d_far_r, far_cnt,
                d_far_w, d_far_cnt,
                d_near_r, d_near_w_cnt,
                d_tent, d_state, d_in_far, cutoff);

            std::swap(d_far_r, d_far_w);
            cudaMemcpy(&near_cnt, d_near_w_cnt, sizeof(int), cudaMemcpyDeviceToHost);
        }
        else
        {
            cudaMemset(d_far_cnt, 0, sizeof(int));
            int grid = (far_cnt + BLOCK_SIZE - 1) / BLOCK_SIZE;

            compact_far_kernel<<<grid, BLOCK_SIZE>>>(
                d_far_r, far_cnt, d_far_w, d_far_keys, d_far_cnt,
                d_tent, d_state, d_in_far);

            std::swap(d_far_r, d_far_w);

            cudaMemcpy(&far_cnt, d_far_cnt, sizeof(int),
                       cudaMemcpyDeviceToHost);

            if (far_cnt == 0)
                break;

            thrust::device_ptr<int> keys(d_far_keys);
            thrust::device_ptr<int> verts(d_far_r);

            thrust::sort_by_key(keys, keys + far_cnt, verts);

            int dmin;
            cudaMemcpy(&dmin, d_far_keys, sizeof(int), cudaMemcpyDeviceToHost);
            bool dup_fallback = false;

            int I;
            if (far_cnt <= K)
            {
                I = far_cnt;
            }
            else
            {
                int V;
                cudaMemcpy(&V, d_far_keys + K - 1, sizeof(int), cudaMemcpyDeviceToHost);
                I = thrust::lower_bound(keys, keys + far_cnt, V) - keys;
                if (I == 0)
                {
                    I = thrust::upper_bound(keys, keys + far_cnt, dmin) - keys;
                    dup_fallback = true;
                }
            }

            int last;
            cudaMemcpy(&last, d_far_keys + (I - 1), sizeof(int), cudaMemcpyDeviceToHost);
            cur_delta = last - dmin;
            if (cur_delta == 0 && !dup_fallback)
                cur_delta = 1;
            cutoff = dmin + cur_delta;
            outfile << cur_delta << "\n";

            int grid_p = (I + BLOCK_SIZE - 1) / BLOCK_SIZE;
            promote_kernel<<<grid_p, BLOCK_SIZE>>>(d_far_r, I, d_near_r, d_state, d_in_far);
            near_cnt = I;

            int rest = far_cnt - I;
            if (rest > 0)
                thrust::copy(verts + I, verts + far_cnt, thrust::device_ptr<int>(d_far_w));
            std::swap(d_far_r, d_far_w);
            cudaMemcpy(d_far_cnt, &rest, sizeof(int), cudaMemcpyHostToDevice);
        }
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
    int *d_far_keys = nullptr;
    cudaMalloc(&d_far_keys, N * sizeof(int));
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
            d_near_cnt, d_far_cnt, d_settled_cnt, d_dmin,
            K, outfile, d_far_keys);

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
    cudaFree(d_far_keys);
    delete[] h_tent;
    // ADDED : END

    outfile.close();
    return 0;
}