
#[cfg(test)]
mod test{
    use std::time::Instant;

    use tch::{Tensor, Kind, Device, nn::Module};

    fn sigmoid(x:&Tensor)->Tensor{
        let mut t = (x * -1.0).exp_();
        t += 1.0;
        t.divide_scalar_(1)
    }

    fn sigmoid_non_reuse(x:&Tensor)->Tensor{
        ((x * -1.0).exp() + 1.0).divide_scalar(1.0)
    }

    #[test]
    fn bidouillage(){
        let input = Tensor::rand(&[1_000_000_000], (Kind::Float, Device::Cpu));
        let start = Instant::now();
        let _ = sigmoid_non_reuse(&input);
        println!("Bad opti {:?}", start.elapsed());
        let start = Instant::now();
        let nj = sigmoid(&input);
        println!("Not Jit {:?}", start.elapsed());
        let sigm = tch::jit::CModule::create_by_tracing("sigmoid_", "forward", &[Tensor::rand(&[10_000_000], (Kind::Float, Device::Cpu))], &mut |x|x.iter().map(sigmoid).collect::<Vec<_>>());
        let sigm = sigm.unwrap(); 
        let start = Instant::now();
        let j = sigm.forward(&input);
        println!("Jit {:?}", start.elapsed());
        println!("matches {:?}", bool::from((nj.eq_tensor(&j)).all()))
    }

}