use crate::handles::lock_handles;
use crate::parallel_aqlm::ParallelAQLMLinear;
use log::info;
use nn::linear::Module;
use std::cell::Cell;
use std::collections::HashMap;
use tensorlib::functional::argmin;
use tokio::sync::Semaphore;
use web_time::Instant;

static CALIB_SEMAPHORE: Semaphore = Semaphore::const_new(1);
thread_local! {
    static AQLM_N_WORKERS_CACHE: Cell<HashMap<(usize, usize), usize>> = Cell::new(HashMap::new());
}

pub(crate) async fn get_optimal_aqlm_n_workers(
    codebooks: &[f32],
    scales: &[f32],
    codes: &[u8],
    out_dim: usize,
    in_group_dim: usize,
) -> anyhow::Result<usize> {
    let _guard = CALIB_SEMAPHORE.acquire().await.unwrap();

    let cache_entry = (out_dim, in_group_dim);
    let mut cache = AQLM_N_WORKERS_CACHE.take();

    let output = cache.get(&cache_entry).cloned();
    if let Some(output) = output {
        AQLM_N_WORKERS_CACHE.set(cache);
        return Ok(output);
    }

    let output =
        do_get_optimal_aqlm_n_workers(codebooks, scales, codes, out_dim, in_group_dim).await?;

    cache.insert(cache_entry, output);
    AQLM_N_WORKERS_CACHE.set(cache);

    Ok(output)
}

pub(crate) async fn do_get_optimal_aqlm_n_workers(
    codebooks: &[f32],
    scales: &[f32],
    codes: &[u8],
    out_dim: usize,
    in_group_dim: usize,
) -> anyhow::Result<usize> {
    let test_data = vec![3f32; in_group_dim * 8];

    let n_workers = {
        let handles_guard = lock_handles().await;
        handles_guard.borrow().len()
    };

    assert!(n_workers > 0);

    let mut timings: Vec<f32> = Vec::new();

    for layer_n_workers in 1..=n_workers {
        let mut aqlm = ParallelAQLMLinear::new(
            format!("calib_{layer_n_workers}"),
            codebooks,
            scales,
            codes,
            out_dim,
            in_group_dim,
            layer_n_workers,
        )
        .await;

        aqlm.forward(&test_data).await;

        let begin = Instant::now();
        for _ in 0..10 {
            aqlm.forward(&test_data).await;
        }

        timings.push(begin.elapsed().as_secs_f64() as f32);

        aqlm.async_drop().await;
        drop(aqlm);
    }

    let output = argmin(&timings) + 1;

    info!(
        "AQLM Calibration({}x{}): {} {:?}",
        out_dim,
        in_group_dim * 8,
        output,
        timings
    );

    Ok(output)
}
