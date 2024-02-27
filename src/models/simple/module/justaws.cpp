#include <torch/extension.h>
#include "ATen/ATen.h"

#include "torchneuron/register.h"

typedef at::BFloat16 bf16;
typedef at::Half fp16;
typedef float fp32;



torch::Tensor forward_cpu(int64_t B, int64_t T, int64_t C, int64_t H, torch::Tensor &s, torch::Tensor &r, torch::Tensor &k, torch::Tensor &v, torch::Tensor &w, torch::Tensor &u) {
    
    auto rr = r.accessor<float, 4>();
    auto kk = k.accessor<float, 4>();
    auto vv = v.accessor<float, 4>();
    auto ww = w.accessor<float, 2>();
    auto uu = u.accessor<float, 2>();
    auto ss = s.accessor<float, 4>();
    
    auto y = torch::zeros({B, H, T + (C/H), C/H});

    auto out = y.accessor<float, 4>();

    // B,Z,H,Z
    // B,T,H,Z
    
    for (int64_t t = 0; t < T; t++) {

        for (int64_t bb = 0; bb < B; bb++) {

            for (int64_t hh = 0; hh < H; hh++) {


                for (int64_t i = 0; i < C/H; i++) {

                    
                    auto kkk = kk[bb][t][hh][i];  
                    auto uuu = uu[hh][i]; 
                    auto rrr = rr[bb][t][hh][i];
                    auto www = ww[hh][i];

                    for(int64_t j = 0; j < C/H; j++){

                    auto vvv = vv[bb][t][hh][j];

                    auto atu = vvv * kkk;

                    if(t == 0){
                        auto sss = ss[bb][hh][i][j];

                        auto sssatuuuu = ((atu*uuu)+sss);

                        out[bb][hh][t][j] = out[bb][hh][t][j] + (sssatuuuu*rrr);

                        out[bb][hh][T+i][j] = ((sss*www)+atu);
                    }
                    else{
                        auto sss = out[bb][hh][T+i][j];

                        auto sssatuuuu = ((atu*uuu)+sss);

                        out[bb][hh][t][j] = out[bb][hh][t][j] + (sssatuuuu*rrr);

                        out[bb][hh][T+i][j] = ((sss*www)+atu);
                    }

                    }

                }
            }
        }
    }
    return y;
}



NEURON_LIBRARY(my_ops, m) {
   m.def("forward_cpu", &forward_cpu, "forward_cpu");
}
