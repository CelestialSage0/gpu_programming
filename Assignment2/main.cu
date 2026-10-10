#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <sys/time.h>
#include <cuda.h>
using namespace std;

// __global__ void addKernel(int *matrixA, int *matrixB, int *matrixC, int size)
// {
// 	int id = blockIdx.x * blockDim.x + threadIdx.x;
// 	if (id < size)
// 	{
// 		matrixC[id] = matrixA[id] + matrixB[id];
// 	}
// }

__global__ void transposeMultiplyKernel(int *matrixA, int *matrixB, int *matrixC, int p, int q, int r, bool transposeA, bool transposeB)
{
	__shared__ int matA[32][33];
	__shared__ int matB[32][33];

	int row = blockIdx.y * 32 + threadIdx.y;
	int col = blockIdx.x * 32 + threadIdx.x;

	int sum = 0;

	for (int t = 0; t < q; t += 32)
	{
		if (transposeA)
		{
			if (row < p && (t + threadIdx.x) < q)
			{
				matA[threadIdx.y][threadIdx.x] =
					matrixA[(t + threadIdx.x) * p + row];
			}
			else
			{
				matA[threadIdx.y][threadIdx.x] = 0;
			}
		}
		else
		{
			if (row < p && (t + threadIdx.x) < q)
			{
				matA[threadIdx.y][threadIdx.x] =
					matrixA[row * q + (t + threadIdx.x)];
			}
			else
			{
				matA[threadIdx.y][threadIdx.x] = 0;
			}
		}

		if (transposeB)
		{
			if (col < r && (t + threadIdx.y) < q)
			{
				matB[threadIdx.y][threadIdx.x] =
					matrixB[col * q + (t + threadIdx.y)];
			}
			else
			{
				matB[threadIdx.y][threadIdx.x] = 0;
			}
		}
		else
		{
			if (col < r && (t + threadIdx.y) < q)
			{
				matB[threadIdx.y][threadIdx.x] =
					matrixB[(t + threadIdx.y) * r + col];
			}
			else
			{
				matB[threadIdx.y][threadIdx.x] = 0;
			}
		}

		__syncthreads();

		for (int k = 0; k < 32; k++)
		{
			sum += matA[threadIdx.y][k] *
				   matB[k][threadIdx.x];
		}

		__syncthreads();
	}

	if (row < p && col < r)
	{
		matrixC[row * r + col] += sum;
	}
}

// function to compute the output matrix
void compute(int p, int q, int r, int *h_matrixA, int *h_matrixB,
			 int *h_matrixC, int *h_matrixD, int *h_matrixE)
{
	// Device variables declarations...
	int *d_matrixA, *d_matrixB, *d_matrixC, *d_matrixD, *d_matrixE;

	// allocate memory...
	cudaMalloc(&d_matrixA, q * p * sizeof(int));
	cudaMalloc(&d_matrixB, q * r * sizeof(int));
	cudaMalloc(&d_matrixC, p * q * sizeof(int));
	cudaMalloc(&d_matrixD, r * q * sizeof(int));
	cudaMalloc(&d_matrixE, p * r * sizeof(int));

	// copy the values...
	cudaMemcpy(d_matrixA, h_matrixA, q * p * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_matrixB, h_matrixB, q * r * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_matrixC, h_matrixC, p * q * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_matrixD, h_matrixD, r * q * sizeof(int), cudaMemcpyHostToDevice);

	/* ****************************************************************** */
	/* Write your code here */
	/* Configure and launch kernels */
	dim3 block(32, 32);
	dim3 grid((r + 32 - 1) / 32, (p + 32 - 1) / 32);

	int *temp1, *temp2;
	cudaMalloc(&temp1, p * r * sizeof(int));
	cudaMalloc(&temp2, p * r * sizeof(int));

	transposeMultiplyKernel<<<grid, block>>>(d_matrixA, d_matrixB, temp1, p, q, r, true, false);
	transposeMultiplyKernel<<<grid, block>>>(d_matrixC, d_matrixD, temp2, p, q, r, false, true);

	// addKernel<<<(p * r + 256 - 1) / 256, 256>>>(temp1, temp2, d_matrixE, p * r);

	cudaFree(temp1);
	cudaFree(temp2);
	/* ****************************************************************** */

	// copy the result back...
	cudaMemcpy(h_matrixE, d_matrixE, p * r * sizeof(int), cudaMemcpyDeviceToHost);

	// deallocate the memory...
	cudaFree(d_matrixA);
	cudaFree(d_matrixB);
	cudaFree(d_matrixC);
	cudaFree(d_matrixD);
	cudaFree(d_matrixE);
}

// function to read the input matrices from the input file
void readMatrix(FILE *inputFilePtr, int *matrix, int rows, int cols)
{
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			fscanf(inputFilePtr, "%d", &matrix[i * cols + j]);
		}
	}
}

// function to write the output matrix into the output file
void writeMatrix(FILE *outputFilePtr, int *matrix, int rows, int cols)
{
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			fprintf(outputFilePtr, "%d ", matrix[i * cols + j]);
		}
		fprintf(outputFilePtr, "\n");
	}
}

int main(int argc, char **argv)
{
	// variable declarations
	int p, q, r;
	int *matrixA, *matrixB, *matrixC, *matrixD, *matrixE;
	struct timeval t1, t2;
	double seconds, microSeconds;

	// get file names from command line
	char *inputFileName = argv[1];
	char *outputFileName = argv[2];

	// file pointers
	FILE *inputFilePtr, *outputFilePtr;

	inputFilePtr = fopen(inputFileName, "r");
	if (inputFilePtr == NULL)
	{
		printf("Failed to open the input file.!!\n");
		return 0;
	}

	// read input values
	fscanf(inputFilePtr, "%d %d %d", &p, &q, &r);

	// allocate memory and read input matrices
	matrixA = (int *)malloc(q * p * sizeof(int));
	matrixB = (int *)malloc(q * r * sizeof(int));
	matrixC = (int *)malloc(p * q * sizeof(int));
	matrixD = (int *)malloc(r * q * sizeof(int));
	readMatrix(inputFilePtr, matrixA, q, p);
	readMatrix(inputFilePtr, matrixB, q, r);
	readMatrix(inputFilePtr, matrixC, p, q);
	readMatrix(inputFilePtr, matrixD, r, q);

	// allocate memory for output matrix
	matrixE = (int *)malloc(p * r * sizeof(int));

	// call the compute function
	gettimeofday(&t1, NULL);
	compute(p, q, r, matrixA, matrixB, matrixC, matrixD, matrixE);
	cudaDeviceSynchronize();
	gettimeofday(&t2, NULL);

	// print the time taken by the compute function
	seconds = t2.tv_sec - t1.tv_sec;
	microSeconds = t2.tv_usec - t1.tv_usec;
	printf("Time taken (ms): %.3f\n", 1000 * seconds + microSeconds / 1000);

	// store the result into the output file
	outputFilePtr = fopen(outputFileName, "w");
	writeMatrix(outputFilePtr, matrixE, p, r);

	// close files
	fclose(inputFilePtr);
	fclose(outputFilePtr);

	// deallocate memory
	free(matrixA);
	free(matrixB);
	free(matrixC);
	free(matrixD);
	free(matrixE);

	return 0;
}
