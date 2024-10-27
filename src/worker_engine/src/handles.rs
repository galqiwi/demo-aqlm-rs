use crate::registry_rpc_handle::RPCLinearRegistryHandle;
use std::cell::RefCell;
use std::mem;
use tokio::sync::{Semaphore, SemaphorePermit};

thread_local! {
    static HANDLES_G: RefCell<Vec<RPCLinearRegistryHandle>> = const { RefCell::new(Vec::new()) };
}

static HANDLES_SEMAPHORE_G: Semaphore = Semaphore::const_new(1);

pub(crate) async fn lock_handles() -> HandlesGuard {
    let guard = HANDLES_SEMAPHORE_G.acquire().await.unwrap();
    let inner = HANDLES_G.take();

    HandlesGuard {
        _guard: guard,
        inner,
    }
}

pub struct HandlesGuard {
    _guard: SemaphorePermit<'static>,
    inner: Vec<RPCLinearRegistryHandle>,
}

impl HandlesGuard {
    pub fn borrow(&self) -> &Vec<RPCLinearRegistryHandle> {
        &self.inner
    }

    pub fn borrow_mut(&mut self) -> &mut Vec<RPCLinearRegistryHandle> {
        &mut self.inner
    }
}

impl Drop for HandlesGuard {
    fn drop(&mut self) {
        HANDLES_G.set(mem::take(&mut self.inner))
    }
}
