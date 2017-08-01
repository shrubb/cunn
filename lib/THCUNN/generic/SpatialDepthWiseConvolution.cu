#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/SpatialDepthWiseConvolution.cu"
#else

#include "common.h"
#include <vector>

// for updateOutput
__global__ void fillOutputWithBiasKernel(
  real *output, int batchSize, int elementsPerPlane,
  real *bias, int nInputPlane, int nOutputPlane) {

  int index = blockIdx.x * blockDim.x + threadIdx.x;

  if (index < batchSize * nInputPlane * nOutputPlane * elementsPerPlane) {
    real *outputPixel = &output[index];

    // index of a pixel in the output sample
    index %= (elementsPerPlane * nInputPlane * nOutputPlane);
    int inPlaneIdx = index / (elementsPerPlane * nOutputPlane);
    // index of a pixel in the output for a single input
    index %= (elementsPerPlane * nOutputPlane);
    int outPlaneIdx = index / elementsPerPlane;

    // bias is of size (nOutputPlane) x (nInputPlane)
    *outputPixel = bias[outPlaneIdx*nInputPlane + inPlaneIdx];
  }
}

// Helper function. Transposes `tensor` pseudo-in-place using `buffer`.
// `buffer` is enlarged if needed.
// This function REQUIRES `buffer` to be non-NULL.
void transposeWithBuffer(
  THCState *state, THCTensor *tensor, THCStorage *buffer, int dim1, int dim2) {

  THAssert(buffer != NULL);

  THCTensor *tensor_viewT = THCTensor_(newTranspose)(state, tensor, dim1, dim2);
  
  // Size of `buffer` (== `columns`, for example) should be
  // enough in general, but who knows, so let `setStorageNd` ensure
  THCTensor *bufferTensorT = THCTensor_(new)(state);
  THCTensor_(setStorageNd)(state, bufferTensorT, buffer, 0, 
    tensor_viewT->nDimension, tensor_viewT->size, NULL);

  // This makes a contiguous tensor from `tensor_viewT`, i.e. does the actual transpose
  THCTensor_(copy)(state, bufferTensorT, tensor_viewT);
  // Copy the transposed data back
  THCTensor_(copy)(state, tensor, bufferTensorT);

  // Now reshape `tensor` to match the transposed size
  std::vector<long> newSize(tensor->size, tensor->size + tensor->nDimension);
  std::swap(newSize[dim1], newSize[dim2]);
  THCTensor_(setStorageNd)(state, tensor, tensor->storage,
    tensor->storageOffset, tensor->nDimension, newSize.data(), NULL);

  THCTensor_(free)(state, tensor_viewT);
  THCTensor_(free)(state, bufferTensorT);
}

static inline void THNN_(SpatialDepthWiseConvolution_shapeCheck)(
                         THCState *state,
                         THCTensor *input, THCTensor *gradOutput,
                         THCTensor *weight, THCTensor *bias,
                         int kH, int kW, int dH, int dW, int padH, int padW) {
  THArgCheck(kW > 0 && kH > 0, 9,
             "kernel size should be greater than zero, but got kH: %d kW: %d", kH, kW);
  THArgCheck(dW > 0 && dH > 0, 11,
             "stride should be greater than zero, but got dH: %d dW: %d", dH, dW);
  THCUNN_argCheck(state, weight->nDimension == 4, 5, weight,
                  "2D or 4D weight tensor expected, but got: %s");

  if (bias != NULL) {
    THCUNN_check_dim_size(state, bias, 2, 0, weight->size[0]);
    THCUNN_check_dim_size(state, bias, 2, 1, weight->size[1]);
  }

  int ndim = input->nDimension;
  int dimf = 0;
  int dimh = 1;
  int dimw = 2;

  if (ndim == 4) {
    dimf++;
    dimh++;
    dimw++;
  }

  THCUNN_argCheck(state, ndim == 3 || ndim == 4, 2, input,
                  "3D or 4D input tensor expected but got: %s");

  long nInputPlane  = weight->size[1];
  long inputHeight  = input->size[dimh];
  long inputWidth   = input->size[dimw];
  long nOutputPlane = weight->size[0];
  long outputHeight = (inputHeight + 2*padH - kH) / dH + 1;
  long outputWidth  = (inputWidth + 2*padW - kW) / dW + 1;

  if (outputWidth < 1 || outputHeight < 1)
      THError("Given input size: (%d x %d x %d). "
              "Calculated output size: (%d x %d x %d). Output size is too small",
              nInputPlane,inputHeight,inputWidth,nOutputPlane*nInputPlane,outputHeight,outputWidth);

  THCUNN_check_dim_size(state, input, ndim, dimf, nInputPlane);

  if (gradOutput != NULL) {
    THCUNN_check_dim_size(state, gradOutput, ndim + 1, dimf, nInputPlane);
    THCUNN_check_dim_size(state, gradOutput, ndim + 1, dimh, nOutputPlane);
    THCUNN_check_dim_size(state, gradOutput, ndim + 1, dimw, outputHeight);
    THCUNN_check_dim_size(state, gradOutput, ndim + 1, dimw + 1, outputWidth);
  }
}

void THNN_(SpatialDepthWiseConvolution_updateOutput)(
           THCState *state,
           THCTensor *input,
           THCTensor *output,
           THCTensor *weight,
           THCTensor *bias,
           THCTensor *columns,
           THCTensor *ones,
           int kW, int kH,
           int dW, int dH,
           int padW, int padH) {

  THCUNN_assertSameGPU(state, 5, input, output, weight, columns, ones);
  if (bias) {
    THCUNN_assertSameGPU(state, 2, weight, bias);
  }

  // Params:
  int nInputPlane = weight->nDimension == 2 ? weight->size[1]/(kH*kW) : weight->size[1];
  int nOutputPlane = weight->size[0];
  if (weight->nDimension == 2) {
    THCTensor_(resize4d)(state, weight, nOutputPlane, nInputPlane, kH, kW);
  }

  THNN_(SpatialDepthWiseConvolution_shapeCheck)
       (state, input, NULL, weight, bias, kH, kW, dH, dW, padH, padW);

  long inputWidth   = input->size[3];
  long inputHeight  = input->size[2];
  long outputWidth  = (inputWidth + 2*padW - kW) / dW + 1;
  long outputHeight = (inputHeight + 2*padH - kH) / dH + 1;

  input = THCTensor_(newContiguous)(state, input);

  int batch = 1;
  if (input->nDimension == 3) {
    // Force batch
    batch = 0;
    THCTensor_(resize4d)(state, input, 1, input->size[0], input->size[1], input->size[2]);
  }

  // Batch size + input planes
  long batchSize = input->size[0];

  // Resize output
  THCTensor_(resize5d)(state, output, batchSize, nInputPlane, nOutputPlane, outputHeight, outputWidth);
  // Resize temporary columns
  THCTensor_(resize3d)(state, columns, nInputPlane, kW*kH, outputHeight*outputWidth);

  transposeWithBuffer(state, weight, columns->storage, 0, 1);

  // Helpers
  THCTensor *input_n = THCTensor_(new)(state);
  THCTensor *output_n = THCTensor_(new)(state);

  // For cublas<t>gemmBatched

  // I'd love to use nullptr but someone might be using a medieval compiler
  const real **columns_batches = NULL;
  const real **weight_batches  = NULL;
  real **output_batches        = NULL;
  THCudaCheck(cudaMalloc(&columns_batches,  sizeof(real*) * nInputPlane));
  THCudaCheck(cudaMalloc(&weight_batches,  sizeof(real*) * nInputPlane));
  THCudaCheck(cudaMalloc(&output_batches, sizeof(real*) * batchSize*nInputPlane));
  std::vector<real*> columns_batches_host (nInputPlane);
  std::vector<real*> weight_batches_host (nInputPlane);
  std::vector<real*> output_n_batches_host(batchSize*nInputPlane);
  for (int k = 0; k < nInputPlane; ++k) {
    columns_batches_host[k] = THCTensor_(data)(state, columns) + k * columns->stride[0];
    weight_batches_host[k] = THCTensor_(data)(state, weight) + k * weight->stride[0];
  }
  for (int k = 0; k < output_n_batches_host.size(); ++k) {
    output_n_batches_host[k] = THCTensor_(data)(state, output) + k * output->stride[1];
  }
  THCudaCheck(cudaMemcpy(columns_batches, columns_batches_host.data(), 
    sizeof(real*)*nInputPlane, cudaMemcpyHostToDevice));
  THCudaCheck(cudaMemcpy(weight_batches, weight_batches_host.data(), 
    sizeof(real*)*nInputPlane, cudaMemcpyHostToDevice));
  THCudaCheck(cudaMemcpy(output_batches, output_n_batches_host.data(), 
      sizeof(real*)*batchSize*nInputPlane, cudaMemcpyHostToDevice));

  // Do bias first (fill the output)
  if (bias) {
    fillOutputWithBiasKernel 
      <<<GET_BLOCKS(batchSize*outputHeight*outputWidth*nInputPlane*nOutputPlane), 
      CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state)>>> (
        THCTensor_(data)(state, output), batchSize, outputHeight*outputWidth,
        THCTensor_(data)(state, bias), nInputPlane, nOutputPlane);
    THCudaCheck(cudaGetLastError());
  } else {
    THCTensor_(zero)(state, output);
  }

  // For each elt in batch, do:
  for (int elt = 0; elt < batchSize; elt++) {
    // Matrix mulitply per output:
    THCTensor_(select)(state, input_n, input, 0, elt);
    THCTensor_(select)(state, output_n, output, 0, elt);

    // Extract columns:
    im2col(
      THCState_getCurrentStream(state),
      THCTensor_(data)(state, input_n),
      nInputPlane, inputHeight, inputWidth, kH, kW, padH, padW, dH, dW,
      1, 1, THCTensor_(data)(state, columns)
    );

    #ifndef THC_REAL_IS_HALF
      // M,N,K are dims of matrix A and B
      // (see http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm)
      long m = outputHeight * outputWidth;
      long n = nOutputPlane;
      long k = kH * kW;
      real **output_n_batches = output_batches + elt*nInputPlane;

      // Do GEMM (note: this is a bit confusing because gemm assumes column-major matrices)
      // (m x k) * (k x n) = (m x n)
      #ifdef THC_REAL_IS_FLOAT
      THCudaBlas_SgemmBatched(
      #elif defined(THC_REAL_IS_DOUBLE)
      THCudaBlas_DgemmBatched(
      #endif
        state,
        'n', 'n',
        m, n, k,
        ScalarConvert<int, real>::to(1),
        columns_batches, m,
        weight_batches, k,
        ScalarConvert<int, real>::to(1),
        output_n_batches, m,
        nInputPlane
      );
    #endif
  }

  // transpose back
  transposeWithBuffer(state, weight, columns->storage, 0, 1);

  // Free
  THCTensor_(free)(state, input_n);
  THCTensor_(free)(state, output_n);

  THCudaCheck(cudaFree(columns_batches));
  THCudaCheck(cudaFree(weight_batches));
  THCudaCheck(cudaFree(output_batches));

  // Transpose output
  THCTensor_(resize4d)(state, output, batchSize, nInputPlane * nOutputPlane, outputHeight, outputWidth);

  // Make a contiguous copy of output (OPTIONAL)
  // THCTensor *_output = THCTensor_(newContiguous)(state, output);

  // Resize output
  if (batch == 0) {
    THCTensor_(select)(state, output, NULL, 0, 0);
    THCTensor_(select)(state, input, NULL, 0, 0);
  }
  //else
    //THCTensor_(resize5d)(state, output, batchSize, nOutputPlane, nInputPlane, outputHeight, outputWidth);

  // Copy output back
  // THCTensor_(freeCopyTo)(state, _output, output);

  THCTensor_(free)(state, input);
}

void THNN_(SpatialDepthWiseConvolution_updateGradInput)(
           THCState *state,
           THCTensor *input,
           THCTensor *gradOutput,
           THCTensor *gradInput,
           THCTensor *weight,
           THCTensor *gradColumns,
           THCTensor *ones,
           int kW, int kH,
           int dW, int dH,
           int padW, int padH) {

  THCUNN_assertSameGPU(state, 5, input, gradOutput, weight,
                       gradColumns, gradInput);

  // Params:
  int nInputPlane = weight->nDimension == 2 ? weight->size[1]/(kH*kW) : weight->size[1];
  int nOutputPlane = weight->size[0];
  if (weight->nDimension == 2) {
    THCTensor_(resize4d)(state, weight, nOutputPlane, nInputPlane, kH, kW);
  }

  gradOutput = THCTensor_(newWithTensor)(state, gradOutput);

  if (input->nDimension == 3) {
    if (gradOutput->nDimension == 3) {
      THCTensor_(resize4d)(state, gradOutput, nInputPlane, nOutputPlane, gradOutput->size[1], gradOutput->size[2]);
    }
  }
  else
  {
    if (gradOutput->nDimension == 4) {
      THCTensor_(resize5d)(state, gradOutput, gradOutput->size[0], nInputPlane, nOutputPlane, gradOutput->size[2], gradOutput->size[3]);
    }
  }

  THNN_(SpatialDepthWiseConvolution_shapeCheck)
       (state, input, gradOutput, weight, NULL, kH, kW, dH, dW, padH, padW);

  transposeWithBuffer(state, weight, gradColumns->storage, 0, 1);

  input = THCTensor_(newContiguous)(state, input);

  int batch = 1;
  if (input->nDimension == 3) {
    // Force batch
    batch = 0;
    THCTensor_(resize4d)(state, input, 1, input->size[0], input->size[1], input->size[2]);
    THCTensor_(resize5d)(state, gradOutput, 1, gradOutput->size[0], gradOutput->size[1], gradOutput->size[2], gradOutput->size[3]);
  }

  long inputWidth   = input->size[3];
  long inputHeight  = input->size[2];
  long outputWidth  = (inputWidth + 2*padW - kW) / dW + 1;
  long outputHeight = (inputHeight + 2*padH - kH) / dH + 1;

  // Batch size + input planes
  long batchSize = input->size[0];

  // Resize output
  THCTensor_(resize4d)(state, gradInput, batchSize, nInputPlane, inputHeight, inputWidth);

  // Resize temporary columns
  THCTensor_(resize3d)(state, gradColumns, nInputPlane, kW*kH, outputHeight*outputWidth);

  // Helpers
  THCTensor *gradInput_n  = THCTensor_(new)(state);
  THCTensor *gradOutput_n = THCTensor_(new)(state);

  // Helpers for DepthWiseConvolution
  THCTensor *gradOutput_i  = THCTensor_(new)(state);
  THCTensor *gradInput_i   = THCTensor_(new)(state);
  THCTensor *weight_i      = THCTensor_(new)(state);
  THCTensor *gradColumns_i = THCTensor_(new)(state);

  // For cublas<t>gemmBatched

  // I'd love to use nullptr but someone might be using a medieval compiler
  const real **weight_batches     = NULL;
  const real **gradOutput_batches = NULL;
  real **columns_batches          = NULL;
  THCudaCheck(cudaMalloc(&columns_batches,    sizeof(real*) * nInputPlane));
  THCudaCheck(cudaMalloc(&weight_batches,     sizeof(real*) * nInputPlane));
  THCudaCheck(cudaMalloc(&gradOutput_batches, sizeof(real*) * batchSize*nInputPlane));
  std::vector<real*>      columns_batches_host(nInputPlane);
  std::vector<real*>       weight_batches_host(nInputPlane);
  std::vector<real*> gradOutput_n_batches_host(batchSize*nInputPlane);
  for (int k = 0; k < nInputPlane; ++k) {
    columns_batches_host[k] = THCTensor_(data)(state, gradColumns) + k * gradColumns->stride[0];
    weight_batches_host[k] = THCTensor_(data)(state, weight) + k * weight->stride[0];
  }
  for (int k = 0; k < gradOutput_n_batches_host.size(); ++k) {
    gradOutput_n_batches_host[k] = THCTensor_(data)(state, gradOutput) + k * gradOutput->stride[1];
  }
  THCudaCheck(cudaMemcpy(columns_batches, columns_batches_host.data(), 
    sizeof(real*)*nInputPlane, cudaMemcpyHostToDevice));
  THCudaCheck(cudaMemcpy(weight_batches, weight_batches_host.data(), 
    sizeof(real*)*nInputPlane, cudaMemcpyHostToDevice));
  THCudaCheck(cudaMemcpy(gradOutput_batches, gradOutput_n_batches_host.data(), 
    sizeof(real*)*batchSize*nInputPlane, cudaMemcpyHostToDevice));

  // For each elt in batch, do:
  for (int elt = 0; elt < batchSize; elt ++) {
    // Matrix mulitply per sample:
    THCTensor_(select)(state, gradInput_n, gradInput, 0, elt);
    THCTensor_(select)(state, gradOutput_n, gradOutput, 0, elt);

    #ifndef THC_REAL_IS_HALF
    long m = kW * kH;
    long n = outputHeight * outputWidth; // this is gradColumns's last dimension
    long k = nOutputPlane;
    const real **gradOutput_n_batches = gradOutput_batches + elt*nInputPlane;

    // Do GEMM (note: this is a bit confusing because gemm assumes column-major matrices)
      #ifdef THC_REAL_IS_FLOAT
      THCudaBlas_SgemmBatched(
      #elif defined(THC_REAL_IS_DOUBLE)
      THCudaBlas_DgemmBatched(
      #endif
          state,
          'n', 't',
          n, m, k,
          ScalarConvert<int, real>::to(1),
          gradOutput_n_batches, n,
          weight_batches, m,
          ScalarConvert<int, real>::to(0),
          columns_batches, n,
          nInputPlane
      );
    #endif
    
    // Unpack columns back into input:
    col2im<real, accreal>(
      THCState_getCurrentStream(state),
      THCTensor_(data)(state, gradColumns),
      nInputPlane, inputHeight, inputWidth, kH, kW, padH, padW, dH, dW,
      1, 1, THCTensor_(data)(state, gradInput_n)
    );
  }

  transposeWithBuffer(state, weight, gradColumns->storage, 0, 1);

  // Free
  THCTensor_(free)(state, gradInput_n);
  THCTensor_(free)(state, gradOutput_n);

  THCudaCheck(cudaFree(columns_batches));
  THCudaCheck(cudaFree(weight_batches));
  THCudaCheck(cudaFree(gradOutput_batches));

  // Resize output
  if (batch == 0) {
    THCTensor_(select)(state, gradOutput, NULL, 0, 0);
    THCTensor_(select)(state, input, NULL, 0, 0);
    THCTensor_(select)(state, gradInput, NULL, 0, 0);
  }

  THCTensor_(free)(state, input);
  THCTensor_(free)(state, gradOutput);
}

__global__ void transposeGradOutput(
  real *gradOutputTransposed, real *gradOutput, int batchSize, 
  int nInputPlane, int nOutputPlane, int elementsPerPlane) {

  // gradOutput           is (batchSize) x (nInputPlane) x (nOutputPlane) x (h) x (w)
  // gradOutputTransposed is (nInputPlane) x (nOutputPlane) x (batchSize) x (h) x (w)

  int index = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (index < batchSize * nInputPlane * nOutputPlane * elementsPerPlane) { 
    int pixelIdx = index % elementsPerPlane; // index of a pixel in the plane
    index /= elementsPerPlane;
    int outPlaneIdx = index % nOutputPlane;
    index /= nOutputPlane;
    int inPlaneIdx = index % nInputPlane;
    index /= nInputPlane; // `index` is now the index of a sample in batch

    gradOutputTransposed[
      elementsPerPlane * batchSize * nOutputPlane * inPlaneIdx +
      elementsPerPlane * batchSize * outPlaneIdx +
      elementsPerPlane * index +
      pixelIdx] =

    gradOutput[
      elementsPerPlane * nOutputPlane * nInputPlane * index +
      elementsPerPlane * nOutputPlane * inPlaneIdx +
      elementsPerPlane * outPlaneIdx +
      pixelIdx];
  }
}

void THNN_(SpatialDepthWiseConvolution_accGradParameters)(
           THCState *state,
           THCTensor *input,
           THCTensor *gradOutput,
           THCTensor *gradWeight,
           THCTensor *gradBias,
           THCTensor *columns,
           THCTensor *ones,
           int kW, int kH,
           int dW, int dH,
           int padW, int padH,
           accreal scale_) {

  real scale = ScalarConvert<accreal, real>::to(scale_);

  THCUNN_assertSameGPU(state, 5, input, gradOutput, gradWeight, columns, ones);
  if (gradBias) {
    THCUNN_assertSameGPU(state, 2, gradWeight, gradBias);
  }

  input = THCTensor_(newContiguous)(state, input);

  int batch = 1;
  if (input->nDimension == 3) {
    // Force batch
    batch = 0;
    THCTensor_(resize4d)(state, input, 1, input->size[0], input->size[1], input->size[2]);
    THCTensor_(resize5d)(state, gradOutput, 1, gradOutput->size[0], gradOutput->size[1], gradOutput->size[2], gradOutput->size[3]);
  }

  // Batch size + input planes
  long batchSize = input->size[0];

  long inputWidth   = input->size[3];
  long inputHeight  = input->size[2];
  long outputWidth  = (inputWidth + 2*padW - kW) / dW + 1;
  long outputHeight = (inputHeight + 2*padH - kH) / dH + 1;

  // Params
  int nInputPlane = gradWeight->nDimension == 2 ? gradWeight->size[1]/(kW*kH) : gradWeight->size[1];
  int nOutputPlane = gradWeight->size[0];
  if (gradWeight->nDimension == 2) {
    THCTensor_(resize4d)(state, gradWeight, nOutputPlane, nInputPlane, kH, kW);
  }

  gradOutput = THCTensor_(newWithTensor)(state, gradOutput);
  if (input->nDimension == 3) {
    if (gradOutput->nDimension == 3) {
      THCTensor_(resize4d)(state, gradOutput, nInputPlane, nOutputPlane, gradOutput->size[1], gradOutput->size[2]);
    }
  }
  else
  {
    if (gradOutput->nDimension == 4) {
      THCTensor_(resize5d)(state, gradOutput, gradOutput->size[0], nInputPlane, nOutputPlane, gradOutput->size[2], gradOutput->size[3]);
    }
  }

  THNN_(SpatialDepthWiseConvolution_shapeCheck)
       (state, input, gradOutput, gradWeight, gradBias, kH, kW, dH, dW, padH, padW);
  
  // Resize temporary columns
  THCTensor_(resize3d)(state, columns, kW*kH, batchSize, outputHeight*outputWidth);

  // merge two last (spatial) dimensions
  THCTensor_(setStorage3d)(state, gradWeight, 
    gradWeight->storage, gradWeight->storageOffset,
    nOutputPlane, -1, nInputPlane, -1, kH*kW, -1);

  transposeWithBuffer(state, gradWeight, columns->storage, 0, 1);
  transposeWithBuffer(state, gradBias  , columns->storage, 0, 1);

  // transpose for proper accumulation in GEMM
  transposeWithBuffer(state, gradWeight, columns->storage, 1, 2);

  // Define a buffer of ones, for bias accumulation
  if (ones->nDimension != 2 || ones->size[0]*ones->size[1] < outputHeight*outputWidth) {
    // Resize plane and fill with ones...
    THCTensor_(resize2d)(state, ones, outputHeight, outputWidth);
    THCTensor_(fill)(state, ones, ScalarConvert<int, real>::to(1));
  }

  // Helpers
  THCTensor *input_n = THCTensor_(new)(state);
  THCTensor *gradOutput_n = THCTensor_(new)(state);

  // For each elt in batch, do:
  for (int elt = 0; elt < batchSize; elt ++) {
    // Matrix mulitply per output:
    THCTensor_(select)(state, input_n, input, 0, elt);
    THCTensor_(select)(state, gradOutput_n, gradOutput, 0, elt);

    // Do Bias:
    // (see http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm)
    long m_ = nInputPlane * nOutputPlane;
    long k_ = outputHeight * outputWidth;

    // Do GEMV (note: this is a bit confusing because gemv assumes column-major matrices)
    if (gradBias) {
      #if defined(THC_REAL_IS_FLOAT) || defined(THC_REAL_IS_DOUBLE)
      #ifdef THC_REAL_IS_FLOAT
      THCudaBlas_Sgemv(
      #elif defined(THC_REAL_IS_DOUBLE)
      THCudaBlas_Dgemv(
      #endif
          state,
          't',
          k_, m_,
          scale,
          THCTensor_(data)(state, gradOutput_n), k_,
          THCTensor_(data)(state, ones), 1,
          ScalarConvert<int, real>::to(1),
          THCTensor_(data)(state, gradBias), 1
      );
      #endif
      #ifdef THC_REAL_IS_HALF
      THCudaBlas_Hgemm(
          state,
          't', 'n',
          m_, 1, k_,
          scale,
          THCTensor_(data)(state, gradOutput_n), k_,
          THCTensor_(data)(state, ones), k_,
          ScalarConvert<int, real>::to(1),
          THCTensor_(data)(state, gradBias), m_
      );
      #endif
    }
  }

  THCTensor_(resize4d)(state, ones, nInputPlane, nOutputPlane, batchSize, outputHeight*outputWidth);
  THCTensor *gradOutputTransposed = ones;

  transposeGradOutput
    <<<GET_BLOCKS(THCTensor_(nElement)(state, gradOutput)), 
    CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state)>>>(
        THCTensor_(data)(state, gradOutputTransposed), THCTensor_(data)(state, gradOutput),
        batchSize, nInputPlane, nOutputPlane, outputHeight*outputHeight);
  THCudaCheck(cudaGetLastError());

  THCTensor *gradOutputTransposed_i = THCTensor_(new)(state);
  THCTensor *gradWeight_i = THCTensor_(new)(state);

  for (int inPlaneIdx = 0; inPlaneIdx < nInputPlane; ++inPlaneIdx) {

    // columns: (kW*kH) x (batchSize) x (outputHeight*outputWidth)
    // gradOutputTransposed: (nInputPlane) x (nOutputPlane) x (batchSize) x (outputHeight*outputWidth)
    // gradWeight: (nInputPlane) x (nOutputPlane) x (kH*kW)
    THCTensor_(select)(state, gradWeight_i, gradWeight, 0, inPlaneIdx);
    THCTensor_(select)(state, gradOutputTransposed_i, gradOutputTransposed, 0, inPlaneIdx);
    
    // Extract columns:
    im2col_depthwise(
      THCState_getCurrentStream(state),
      THCTensor_(data)(state, input), batchSize, nInputPlane,
      inPlaneIdx, inputHeight, inputWidth, kH, kW, padH, padW, dH, dW,
      1, 1, THCTensor_(data)(state, columns)
    );

    // M,N,K are dims of matrix A and B
    // (see http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm)
    long m = nOutputPlane;
    long n = kH * kW;
    long k = batchSize * outputHeight * outputWidth;

    // Do GEMM (note: this is a bit confusing because gemm assumes column-major matrices)
    // (m x k) * (n x k)^T = (m x n)
    #ifdef THC_REAL_IS_FLOAT
    THCudaBlas_Sgemm(
    #elif defined(THC_REAL_IS_HALF)
    THCudaBlas_Hgemm(
    #elif defined(THC_REAL_IS_DOUBLE)
    THCudaBlas_Dgemm(
    #endif
        state,
        't', 'n',
        m, n, k,
        scale,
        THCTensor_(data)(state, gradOutputTransposed_i), k,
        THCTensor_(data)(state, columns), k,
        ScalarConvert<int, real>::to(1),
        THCTensor_(data)(state, gradWeight_i), m
    );
  }

  // transpose back
  transposeWithBuffer(state, gradWeight, columns->storage, 1, 2);

  transposeWithBuffer(state, gradWeight, columns->storage, 0, 1);
  transposeWithBuffer(state, gradBias  , columns->storage, 0, 1);

  // un-merge two last (spatial) dimensions
  THCTensor_(setStorage4d)(state, gradWeight,
    gradWeight->storage, gradWeight->storageOffset,
    nOutputPlane, -1, nInputPlane, -1, kH, -1, kW, -1);

  // Free
  THCTensor_(free)(state, input_n);
  THCTensor_(free)(state, gradOutput_n);
  // THCTensor_(free)(state, gradOutputTransposed);

  THCTensor_(free)(state, gradOutputTransposed_i);
  THCTensor_(free)(state, gradWeight_i);

  // Resize
  if (batch == 0) {
    THCTensor_(select)(state, gradOutput, NULL, 0, 0);
    THCTensor_(select)(state, input, NULL, 0, 0);
  }

  THCTensor_(free)(state, input);
  THCTensor_(free)(state, gradOutput);
}

#endif
