mod kernel;

#[cfg(test)]
mod tests {
    use cudarc::driver::{DriverError, LaunchConfig, PushKernelArg};

    #[test]
    fn test_simple() -> Result<(), DriverError> {
        // Get a stream for GPU 0
        let ctx = cudarc::driver::CudaContext::new(0)?;
        let stream = ctx.default_stream();

        // copy a rust slice to the device
        let inp = stream.memcpy_stod(&[1.0f32; 100])?;

        // or allocate directly
        let mut out = stream.alloc_zeros::<f32>(100)?;
        // Dynamically load it into the device
        let ptx = cudarc::nvrtc::Ptx::from_src(crate::kernel::CUDA);
        let module = ctx.load_module(ptx)?;
        let sin_kernel = module.load_function("sin_kernel")?;
        let mut builder = stream.launch_builder(&sin_kernel);
        builder.arg(&mut out);
        builder.arg(&inp);
        builder.arg(&100usize);
        unsafe { builder.launch(LaunchConfig::for_num_elems(100)) }?;
        Ok(())
    }
}
