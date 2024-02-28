#include <stdint.h>
#include <stdlib.h>
#include <torch/torch.h>
#include "torchneuron/register.h"




torch::Tensor forward_cpu_shape(int B, int T, int C, int H, torch::Tensor &s, torch::Tensor &r, torch::Tensor &k, torch::Tensor &v, torch::Tensor &w, torch::Tensor &u) {
    
   torch::Tensor t_out = torch::zeros({B, H, T + (C/H), C/H},torch::kFloat);

   return t_out;
}



NEURON_LIBRARY(my_ops, m) {
   m.def("forward_cpu", &forward_cpu_shape, "forward_cpu_compute");
}
