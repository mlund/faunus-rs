//! GPU-accelerated compute backend using wgpu.
//!
//! Feature-gated behind the `gpu` feature flag. Provides device initialization,
//! buffer management, and spline table upload for GPU-accelerated pairwise
//! force/energy computation and rigid-body Langevin dynamics.

pub mod buffers;
pub mod langevin;

use interatomic::gpu::{GpuGridType, GpuSplineData};
use std::sync::Arc;

/// Owns the wgpu device and queue. Created once at simulation startup.
pub struct GpuContext {
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
}

impl GpuContext {
    /// Initialize GPU device with high-performance preference.
    pub fn new() -> anyhow::Result<Self> {
        pollster::block_on(Self::new_async())
    }

    async fn new_async() -> anyhow::Result<Self> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                ..Default::default()
            })
            .await
            .ok_or_else(|| anyhow::anyhow!("No GPU adapter found"))?;

        log::info!("GPU adapter: {}", adapter.get_info().name);

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("faunus_gpu"),
                    required_features: wgpu::Features::PUSH_CONSTANTS,
                    required_limits: wgpu::Limits {
                        max_push_constant_size: 64,
                        ..Default::default()
                    },
                    memory_hints: Default::default(),
                },
                None,
            )
            .await
            .map_err(|e| anyhow::anyhow!("Failed to create GPU device: {e}"))?;

        Ok(Self {
            device: Arc::new(device),
            queue: Arc::new(queue),
        })
    }

    /// Upload immutable spline data to GPU storage buffers.
    pub fn upload_spline_data<G: GpuGridType>(&self, data: &GpuSplineData<G>) -> SplineBuffers {
        let params = buffers::storage_buffer_init_readonly(
            &self.device,
            "spline_params",
            data.params_as_bytes(),
        );
        let coeffs = buffers::storage_buffer_init_readonly(
            &self.device,
            "spline_coeffs",
            data.coefficients_as_bytes(),
        );
        SplineBuffers { params, coeffs }
    }
}

impl std::fmt::Debug for GpuContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuContext").finish_non_exhaustive()
    }
}

/// Immutable GPU buffers holding spline coefficient tables.
pub struct SplineBuffers {
    pub params: wgpu::Buffer,
    pub coeffs: wgpu::Buffer,
}

/// Read back f32 data from a GPU buffer via a staging buffer.
pub fn readback_f32(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    source: &wgpu::Buffer,
    count: usize,
) -> Vec<f32> {
    let size = (count * 4) as u64;
    let staging = buffers::staging_buffer(device, "readback_staging", size);

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("readback_encoder"),
    });
    encoder.copy_buffer_to_buffer(source, 0, &staging, 0, size);
    queue.submit(Some(encoder.finish()));

    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = tx.send(result);
    });
    device.poll(wgpu::Maintain::Wait);
    rx.recv()
        .expect("Channel closed")
        .expect("Buffer mapping failed");

    let data = slice.get_mapped_range();
    let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    staging.unmap();
    result
}
