
#[cfg(test)]
mod test{
    use tch::{Tensor, Kind, Device, nn::Module, CModule};

    pub use tch_utils_macros::to_jit;
    
    #[to_jit(tensor_size = [200_000])]
    fn sigmoid(x:&Tensor)->Tensor{
        let mut t = (x * -1.0).exp_();
        t += 1.0;
        t.divide_scalar_(1)
    }

    #[test]
    fn test_sigmoid_jit(){
        let input = Tensor::randn(&[2_000_000], (Kind::Float, Device::Cpu));
        
        let jit_out = sigmoid(&input);
        let start = std::time::Instant::now();
        for _ in 0..10_000{
            let _ = sigmoid(&input);
        }
        println!("JIT : {:?}", start.elapsed());
        let jitless_out = sigmoid_jit::sigmoid_jitless(&input);
        let start = std::time::Instant::now();
        for _ in 0..10_000{
            let _ = sigmoid_jit::sigmoid_jitless(&input);
        }
        println!("No JIT : {:?}", start.elapsed());

        assert!(jit_out.equal(&jitless_out));
    }

}