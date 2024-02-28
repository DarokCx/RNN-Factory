#include <torch/extension.h>
#include "ATen/ATen.h"

#include "./justaws.cpp"



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward_cpu", &forward_cpu, "CPU forward");
}

TORCH_LIBRARY(wkv5, m) {
    m.def("forward_cpu", forward_cpu);
}

