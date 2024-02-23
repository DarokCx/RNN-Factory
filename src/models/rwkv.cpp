
#include "ATen/ATen.h"
#include <torch/extension.h>
#include "rwkv.h"
RWKV* rwkv = nullptr;


torch::Tensor  forward_cpu(torch::Tensor& inps) {
    
   // get unsigned ints from tensor
   size_t n = inps.size(0);
    size_t d = inps.size(1);
    if (rwkv->blocks[0].att.state.shape[0] < n) {
        rwkv->set_state(rwkv->new_state(n));
        }
    std::vector<std::vector<size_t>> vecs(n, std::vector<size_t>(d));
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < d; j++) {
            vecs[i][j] = inps[i][j].item().toInt();
        }
    }
   auto oout = rwkv->operator()(vecs); 

    // copy output to tensor
    auto shape = oout.shape;

    std::vector<int64_t> nshape;
    for (int i = 0; i < shape.size(); i++) {
        nshape.push_back(shape[i]);
    }

    at::IntArrayRef shape_ref(nshape);

    torch::Tensor out = torch::from_blob(oout.data, shape_ref, at::kFloat);

    return out;

}

void resetState() {
    rwkv->set_state(rwkv->new_state(rwkv->blocks[0].att.state.shape[0]));
}

torch::Tensor getStateKey(std::string key){

            // split string to get layer 
            auto substr = key.substr(7);
            auto layer = substr.substr(0, substr.find("."));
            auto layerint = std::stoi(layer);

            if (substr.find("att") != std::string::npos) {
                if (substr.find("timeshift") != std::string::npos) {
                    auto data = rwkv->blocks[layerint].att.timeshift.state;

                    auto shape = data.shape;

                    std::vector<int64_t> nshape;
                    for (int i = 0; i < shape.size(); i++) {
                        nshape.push_back(shape[i]);
                    }

                    at::IntArrayRef shape_ref(nshape);

                    torch::Tensor out = torch::from_blob(data.data, shape_ref, at::kFloat);
                    return out;

                   
                } else {
                    auto data = rwkv->blocks[layerint].att.state;

                    auto shape = data.shape;

                    std::vector<int64_t> nshape;
                    for (int i = 0; i < shape.size(); i++) {
                        nshape.push_back(shape[i]);
                    }

                    at::IntArrayRef shape_ref(nshape);

                    torch::Tensor out = torch::from_blob(data.data, shape_ref, at::kFloat);
                    return out;


                }
            } else {
                auto data = rwkv->blocks[layerint].ffn.timeshift.state;

                auto shape = data.shape;

                std::vector<int64_t> nshape;
                for (int i = 0; i < shape.size(); i++) {
                    nshape.push_back(shape[i]);
                }

                at::IntArrayRef shape_ref(nshape);

                torch::Tensor out = torch::from_blob(data.data, shape_ref, at::kFloat);

                return out;

            }
     }

void setStateKey(std::string key, torch::Tensor& input, int64_t batchid = 0){

            if(batchid >= rwkv->blocks[0].att.state.shape[0]){
                rwkv->set_state(rwkv->new_state(batchid+1));
                }
            // split string to get layer 
            auto substr = key.substr(7);
            auto layer = substr.substr(0, substr.find("."));
            auto layerint = std::stoi(layer);

            if (substr.find("att") != std::string::npos) {
                if (substr.find("timeshift") != std::string::npos) {
                    auto data = rwkv->blocks[layerint].att.timeshift.state[batchid];

                    auto inptr = input.data_ptr<float>();
                    memcpy(data.data, inptr, data.data_size_in_bytes);

                   
                } else {
                    auto data = rwkv->blocks[layerint].att.state[batchid];

                    auto inptr = input.data_ptr<float>();
                    memcpy(data.data, inptr, data.data_size_in_bytes);

                }
            } else {
                auto data = rwkv->blocks[layerint].ffn.timeshift.state[batchid];

                auto inptr = input.data_ptr<float>();
                memcpy(data.data, inptr, data.data_size_in_bytes);
            }
     }

const std::string getModelInfo(){

    const std::string out = R"""(
    {
        "model": "rwkv",
        "layers": )""" + std::to_string(rwkv->layers) + R"""(,
        "wkv_state_size": )""" + std::to_string(rwkv->blocks[0].att.state[0].get_element_count()) + R"""(,
        "timeshift_state_size": )""" + std::to_string(rwkv->blocks[0].att.timeshift.state[0].get_element_count()) + R"""(
    }
    )""" ;

    return out;

}

void init(std::string model_path) {
    rwkv = new RWKV(model_path);
}




PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward_cpu", &forward_cpu, "CPU forward");
    m.def("init", &init, "Initialize model");
    m.def("resetState", &resetState, "Reset model state");
    m.def("getStateKey", &getStateKey, "Get state key");
    m.def("setStateKey", &setStateKey, "Set state key");
    m.def("getModelInfo", &getModelInfo, "Get model info");
}

TORCH_LIBRARY(wkv5, m) {
    m.def("forward_cpu", forward_cpu);
    m.def("init", init);
    m.def("resetState", resetState);
    m.def("getStateKey", getStateKey);
    m.def("setStateKey", setStateKey);
    m.def("getModelInfo", getModelInfo);
}


