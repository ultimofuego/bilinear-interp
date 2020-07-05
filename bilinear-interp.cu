#include <cuda_runtime.h>
#include <iostream>
#include <ctime>

#include "EBMP/EasyBMP.h"


// 2D float texture
texture<float, cudaTextureType2D, cudaReadModeElementType> texRefR;
texture<float, cudaTextureType2D, cudaReadModeElementType> texRefG;
texture<float, cudaTextureType2D, cudaReadModeElementType> texRefB;

using namespace std;

#define checkCudaErrors(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

void saveImage(float* imageR,float* imageG,float* imageB, int height, int width, bool method) {
	BMP Output;
	Output.SetSize(width, height);

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			RGBApixel pixel;
			pixel.Red = imageR[i * width + j];
			pixel.Green = imageG[i * width + j];
			pixel.Blue = imageB[i * width + j];
			Output.SetPixel(j, i, pixel);
		}
	}
	if (method)
		Output.WriteToFile("CatGPUout.bmp");
	else
		Output.WriteToFile("CatCPUout.bmp");

}

//Unit square
//we choose a coordinate system in which the four points where f is known are(0, 0), (1, 0), (0, 1), and (1, 1)
// wiki

void bilinearCPU(float* imageR, float* imageG, float* imageB, float* resaultR,float* resaultG,float* resaultB, int height, int width)
{
	for (int j = 0; j < height-1; j++) {
		for (int i = 0; i < width-1; i++) {


			float f01R = imageR[j * width + i];
			float f11R = imageR[j * width + i + 1];
			float f00R = imageR[j * width + width + i];
			float f10R = imageR[j * width + width + i + 1];


			float n11R = f01R * 0.5 + f11R * 0.5;
			float n00R = f00R * 0.5 + f01R * 0.5;
			float n10R = f00R * 0.5 * 0.5 + f10R * 0.5 * 0.5 + f01R * 0.5 * 0.5 + f11R * 0.5 * 0.5;

			resaultR[j* width * 4 + i * 2] = f01R;
			resaultR[j * width * 4 + i * 2 + 1] = n11R;
			resaultR[j * width * 4 + i * 2 + width * 2] = n00R;
			resaultR[j * width * 4 + i * 2 + width * 2 + 1] = n10R;

			float f01G = imageG[j * width + i];
			float f11G = imageG[j * width + i + 1];
			float f00G = imageG[j * width + width + i];
			float f10G = imageG[j * width + width + i + 1];


			float n11G = f01G * 0.5 + f11G * 0.5;
			float n00G = f00G * 0.5 + f01G * 0.5;
			float n10G = f00G * 0.5 * 0.5 + f10G * 0.5 * 0.5 + f01G * 0.5 * 0.5 + f11G * 0.5 * 0.5;

			resaultG[j * width * 4 + i * 2] = f01G;
			resaultG[j * width * 4 + i * 2 + 1] = n11G;
			resaultG[j * width * 4 + i * 2 + width * 2] = n00G;
			resaultG[j * width * 4 + i * 2 + width * 2 + 1] = n10G;

			float f01B = imageB[j * width + i];
			float f11B = imageB[j * width + i + 1];
			float f00B = imageB[j * width + width + i];
			float f10B = imageB[j * width + width + i + 1];


			float n11B = f01B * 0.5 + f11B * 0.5;
			float n00B = f00B * 0.5 + f01B * 0.5;
			float n10B = f00B * 0.5 * 0.5 + f10B * 0.5 * 0.5 + f01B * 0.5 * 0.5 + f11B * 0.5 * 0.5;

			resaultB[j * width * 4 + i * 2] = f01B;
			resaultB[j * width * 4 + i * 2 + 1] = n11B;
			resaultB[j * width * 4 + i * 2 + width * 2] = n00B;
			resaultB[j * width * 4 + i * 2 + width * 2 + 1] = n10B;

		}
	}

}


// A good example to demonstrate the difference between a CPU and a GPU is because the algorithms are almost ideal.



//Unit square
//we choose a coordinate system in which the four points where f is known are(0, 0), (1, 0), (0, 1), and (1, 1)
// wiki

__global__ void myFilter(float* outputR, float* outputG, float* outputB, int imageWidth, int imageHeight) {

	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	float f01R = tex2D(texRefR, col, row);
	float f11R = tex2D(texRefR, col+1, row);
	float f00R = tex2D(texRefR, col, row+1);
	float f10R = tex2D(texRefR, col + 1, row+1);

	float n11R = f01R*0.5+f11R*0.5;
	float n00R = f00R*0.5+f01R*0.5;
	float n10R = f00R * 0.5 * 0.5 + f10R * 0.5*0.5 + f01R * 0.5 * 0.5 + f11R * 0.5 * 0.5;

	outputR[row * imageWidth * 4 + col*2] = f01R;
	outputR[row * imageWidth * 4 + col * 2 + 1] = n11R;
	outputR[row * imageWidth * 4 + col * 2 + imageWidth * 2] = n00R;
	outputR[row * imageWidth * 4 + col * 2 + imageWidth * 2 + 1] = n10R;

	float f01G = tex2D(texRefG, col, row);
	float f11G = tex2D(texRefG, col + 1, row);
	float f00G = tex2D(texRefG, col, row + 1);
	float f10G = tex2D(texRefG, col + 1, row + 1);

	float n11G = f01G * 0.5 + f11G * 0.5;
	float n00G = f00G * 0.5 + f01G * 0.5;
	float n10G = f00G * 0.5 * 0.5 + f10G * 0.5 * 0.5 + f01G * 0.5 * 0.5 + f11G * 0.5 * 0.5;

	outputG[row * imageWidth * 4 + col * 2] = f01G;
	outputG[row * imageWidth * 4 + col * 2 + 1] = n11G;
	outputG[row * imageWidth * 4 + col * 2 + imageWidth * 2] = n00G;
	outputG[row * imageWidth * 4 + col * 2 + imageWidth * 2 + 1] = n10G;

	float f01B = tex2D(texRefB, col, row);
	float f11B = tex2D(texRefB, col + 1, row);
	float f00B = tex2D(texRefB, col, row + 1);
	float f10B = tex2D(texRefB, col + 1, row + 1);

	float n11B = f01B * 0.5 + f11B * 0.5;
	float n00B = f00B * 0.5 + f01B * 0.5;
	float n10B = f00B * 0.5 * 0.5 + f10B * 0.5 * 0.5 + f01B * 0.5 * 0.5 + f11B * 0.5 * 0.5;

	outputB[row * imageWidth * 4 + col * 2] = f01B;
	outputB[row * imageWidth * 4 + col * 2 + 1] = n11B;
	outputB[row * imageWidth * 4 + col * 2 + imageWidth * 2] = n00B;
	outputB[row * imageWidth * 4 + col * 2 + imageWidth * 2 + 1] = n10B;

}


int main(void)
{
	int nIter = 100;
	BMP Image;
	Image.ReadFromFile("cat250x188.bmp");
	int height = Image.TellHeight();
	int width = Image.TellWidth();

	float* imageArrayR = (float*)calloc(height * width, sizeof(float));
	float* imageArrayG = (float*)calloc(height * width, sizeof(float));
	float* imageArrayB = (float*)calloc(height * width, sizeof(float));
	float* outputCPUr = (float*)calloc(height * width*4, sizeof(float));
	float* outputCPUg = (float*)calloc(height * width * 4, sizeof(float));
	float* outputCPUb = (float*)calloc(height * width * 4, sizeof(float));

	float* outputGPUr = (float*)calloc(height * width*4, sizeof(float));
	float* outputDeviceR;
	float* outputGPUg = (float*)calloc(height * width * 4, sizeof(float));
	float* outputDeviceG;
	float* outputGPUb = (float*)calloc(height * width * 4, sizeof(float));
	float* outputDeviceB;


	for (int j = 0; j < Image.TellHeight(); j++) {
		for (int i = 0; i < Image.TellWidth(); i++) {
			imageArrayR[j * width + i] = Image(i, j)->Red;
			imageArrayG[j * width + i] = Image(i, j)->Green;
			imageArrayB[j * width + i] = Image(i, j)->Blue;
		}
	}

	unsigned int start_time = clock();

	for (int j = 0; j < nIter; j++) {
		bilinearCPU(imageArrayR,imageArrayG,imageArrayB, outputCPUr,outputCPUg,outputCPUb, height, width);
	}

	unsigned int elapsedTime = clock() - start_time;
	float msecPerMatrixMulCpu = elapsedTime / nIter;

	cout << "CPU time: " << msecPerMatrixMulCpu << endl;

	int device_count = 0;
	cudaGetDeviceCount(&device_count);


		// Allocate CUDA array in device memory

		//Returns a channel descriptor with format f and number of bits of each component x, y, z, and w
		cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
		cudaArray* cu_arr;

		checkCudaErrors(cudaMallocArray(&cu_arr, &channelDesc, width, height));
		checkCudaErrors(cudaMemcpyToArray(cu_arr, 0, 0, imageArrayR, height * width * sizeof(float), cudaMemcpyHostToDevice));		// set texture parameters
		texRefR.addressMode[0] = cudaAddressModeClamp;
		texRefR.addressMode[1] = cudaAddressModeClamp;
		texRefR.filterMode = cudaFilterModePoint;


		// Bind the array to the texture
		cudaBindTextureToArray(texRefR, cu_arr, channelDesc);

		cudaArray* cu_arrG;

		checkCudaErrors(cudaMallocArray(&cu_arrG, &channelDesc, width, height));
		checkCudaErrors(cudaMemcpyToArray(cu_arrG, 0, 0, imageArrayG, height * width * sizeof(float), cudaMemcpyHostToDevice));		// set texture parameters
		texRefG.addressMode[0] = cudaAddressModeClamp;
		texRefG.addressMode[1] = cudaAddressModeClamp;
		texRefG.filterMode = cudaFilterModePoint;


		// Bind the array to the texture
		cudaBindTextureToArray(texRefG, cu_arrG, channelDesc);

		cudaArray* cu_arrB;

		checkCudaErrors(cudaMallocArray(&cu_arrB, &channelDesc, width, height));
		checkCudaErrors(cudaMemcpyToArray(cu_arrB, 0, 0, imageArrayB, height * width * sizeof(float), cudaMemcpyHostToDevice));		// set texture parameters
		texRefB.addressMode[0] = cudaAddressModeClamp;
		texRefB.addressMode[1] = cudaAddressModeClamp;
		texRefB.filterMode = cudaFilterModePoint;


		// Bind the array to the texture
		cudaBindTextureToArray(texRefB, cu_arrB, channelDesc);

		checkCudaErrors(cudaMalloc(&outputDeviceR, height * width * 4* sizeof(float)));
		checkCudaErrors(cudaMalloc(&outputDeviceG, height * width * 4 * sizeof(float)));
		checkCudaErrors(cudaMalloc(&outputDeviceB, height * width * 4 * sizeof(float)));

		dim3 threadsPerBlock(32, 32);
		dim3 blocksPerGrid((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
			(height + threadsPerBlock.y - 1) / threadsPerBlock.y);

		cudaEvent_t start;
		cudaEvent_t stop;
		checkCudaErrors(cudaEventCreate(&start));
		checkCudaErrors(cudaEventCreate(&stop));

		// start record
		checkCudaErrors(cudaEventRecord(start, 0));

		for (int j = 0; j < nIter; j++) {
			myFilter << <blocksPerGrid, threadsPerBlock >> > (outputDeviceR, outputDeviceG, outputDeviceB, width, height);
		}

		// stop record
		checkCudaErrors(cudaEventRecord(stop, 0));

		// wait end of event
		checkCudaErrors(cudaEventSynchronize(stop));

		float msecTotal = 0.0f;
		checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));

		float msecPerMatrixMul = msecTotal / nIter;

		cout << "GPU time: " << msecPerMatrixMul << endl;

		cudaDeviceSynchronize();

		checkCudaErrors(cudaMemcpy(outputGPUr, outputDeviceR, height * width * 4 * sizeof(float), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(outputGPUg, outputDeviceG, height* width * 4 * sizeof(float), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(outputGPUb, outputDeviceB, height* width * 4 * sizeof(float), cudaMemcpyDeviceToHost));

		cudaDeviceSynchronize();

		saveImage(outputGPUr, outputGPUg, outputGPUb, height*2, width*2, true);
		saveImage(outputCPUr,outputCPUg,outputCPUb, height*2, width*2, false);

		checkCudaErrors(cudaFreeArray(cu_arr));
		checkCudaErrors(cudaFree(outputDeviceR));
		checkCudaErrors(cudaFreeArray(cu_arrG));
		checkCudaErrors(cudaFree(outputDeviceG));
		checkCudaErrors(cudaFreeArray(cu_arrB));
		checkCudaErrors(cudaFree(outputDeviceB));
	
	return 0;
}

