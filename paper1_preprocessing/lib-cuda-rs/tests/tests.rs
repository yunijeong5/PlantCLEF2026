
extern crate anyhow;
extern crate candle;

//use cuda_rs::lib;

use anyhow::Result;
use candle::{DType, Device, IndexOp, Tensor}; //, D};
//use candle::IndexOp;


fn to_vec3_round(t: Tensor, digits: i32) -> Result<Vec<Vec<Vec<f32>>>> {
    let b = 10f32.powi(digits);
    let t = t.to_vec3::<f32>()?;
    let t = t
        .iter()
        .map(|t| {
            t.iter()
                .map(|t| t.iter().map(|t| f32::round(t * b) / b).collect())
                .collect()
        })
        .collect();
    Ok(t)
}


#[test]
fn test_hello_world_add() -> Result<()> {
    let _device = Device::new_cuda(0)?;
    //Ok(lib_cuda_rs::lib_hello_world_add()?)
    Ok(lib_cuda_rs::cuda_hello_world_add()?)
}


//fn test_attention_internal(attn_type : lib_cuda_rs::AttnType) -> Result<()> {
fn test_attention_internal(attn_type : i32) -> Result<()> {
    let device = Device::new_cuda(0)?;

    let q = Tensor::arange(0u32, 48, &device)?
        .to_dtype(DType::F16)?
        .reshape((1, 3, 2, 8))?;
    let k = (&q / 40.)?;
    let v = (&q / 50.)?;
    let q = (&q / 30.)?;

    let q = q.transpose(1, 2)?;
    let k = k.transpose(1, 2)?;
    let v = v.transpose(1, 2)?;
    //let ys2 = lib_cuda_rs::attn_generic(&q, &k, &v, attn_type)?.transpose(1, 2)?;
    let ys2 = lib_cuda_rs::cuda_attn_generic(&q, &k, &v, attn_type)?.transpose(1, 2)?;
    let ys2 = ys2.i(0)?.to_dtype(DType::F32)?;

    println!("ys2.dims(): {:?}", ys2.dims());
    println!("ys2: {:?}", ys2);
    assert_eq!(ys2.dims(), &[3, 2, 8]);


    let t_good = 
        [
            [
                [0.0837, 0.1038, 0.1238, 0.1438, 0.1637, 0.1837, 0.2037, 0.2238],
                [0.0922, 0.1122, 0.1322, 0.1522, 0.1721, 0.1921, 0.2122, 0.2322]
            ],
            [
                [0.4204, 0.4404, 0.4604, 0.4805, 0.5005, 0.5205, 0.5405, 0.5605],
                [0.428, 0.448, 0.468, 0.488, 0.5083, 0.5283, 0.5483, 0.5684]
            ],
            [
                [0.7554, 0.7754, 0.7954, 0.8154, 0.8354, 0.8555, 0.8755, 0.8955],
                [0.7622, 0.7822, 0.8022, 0.8223, 0.8423, 0.8623, 0.8823, 0.9023]
            ]
        ];

    assert_eq!(to_vec3_round(ys2, 4)?, t_good);

    Ok(())
}

/*
#[test]
fn test_attention_candle_basic() -> Result<()> {
    test_attention_internal(cuda_rs::AttnType::CandleLineByLine)
}

*/