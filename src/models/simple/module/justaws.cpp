#include <stdint.h>
#include <stdlib.h>
#include <torch/torch.h>

torch::Tensor forward_cpu(const torch::Tensor &s, const torch::Tensor &r, const torch::Tensor &k, const torch::Tensor &v, const torch::Tensor &w, const torch::Tensor &u) {
    
    // printf("forward_cpu\n");
    // printf("s: %d\n", int(s.dim()));
    // printf("r: %d\n", int(r.dim()));
    // printf("k: %d\n", int(k.dim()));
    // printf("v: %d\n", int(v.dim()));
    // printf("w: %d\n", int(w.dim()));
    // printf("u: %d\n", int(u.dim()));
    auto rr = r.accessor<float, 4>();
    auto kk = k.accessor<float, 4>();
    auto vv = v.accessor<float, 4>();
    auto ww = w.accessor<float, 2>();
    auto uu = u.accessor<float, 2>();
    auto ss = s.accessor<float, 4>();

    int64_t B = r.size(0);
    int64_t T = r.size(1);
    int64_t H = w.size(0);
    int64_t Z = w.size(1);
    
    auto y = torch::zeros({B, H, T + (Z), Z}, torch::kFloat);

    auto out = y.accessor<float, 4>();

    // B,Z,H,Z
    // B,T,H,Z
    
    for (int64_t t = 0; t < T; t++) {

        for (int64_t bb = 0; bb < B; bb++) {

            for (int64_t hh = 0; hh < H; hh++) {


                for (int64_t i = 0; i < Z; i++) {

                    
                    auto kkk = kk[bb][t][hh][i];  
                    auto uuu = uu[hh][i]; 
                    auto rrr = rr[bb][t][hh][i];
                    auto www = ww[hh][i];

                    for(int64_t j = 0; j < Z; j++){

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


