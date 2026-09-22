use std::collections::HashMap;
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc, Condvar, Mutex, OnceLock,
    mpsc::{self, Receiver, Sender},
};
use std::time::Duration;

use crate::Value;

#[derive(Clone, Debug)]
pub(crate) enum SyncValue {
    Number(f64),
    Text(String),
    Boolean(bool),
    List(Vec<SyncValue>),
    Object(HashMap<String, SyncValue>),
    Null,
}

impl SyncValue {
    pub(crate) fn from_value(value: &Value) -> Result<Self, String> {
        Ok(match value {
            Value::Number(v) => Self::Number(*v),
            Value::Text(v) => Self::Text(v.clone()),
            Value::Boolean(v) => Self::Boolean(*v),
            Value::Null => Self::Null,
            Value::List(values) => Self::List(values.iter().map(Self::from_value).collect::<Result<Vec<_>, _>>()?),
            Value::Object(values) => {
                Self::Object(values.iter()
                    .map(|(k, v)| Ok((k.clone(), Self::from_value(v)?)))
                    .collect::<Result<HashMap<_, _>, String>>()?)
            }
            Value::Function(_) | Value::Tensor(_) => {
                return Err("Nano sync: Tensor/Function não podem ser compartilhados".into());
            }
        })
    }

    pub(crate) fn into_value(self) -> Value {
        match self {
            Self::Number(v) => Value::Number(v),
            Self::Text(v) => Value::Text(v),
            Self::Boolean(v) => Value::Boolean(v),
            Self::Null => Value::Null,
            Self::List(values) => Value::List(values.into_iter().map(Self::into_value).collect()),
            Self::Object(values) => Value::Object(values.into_iter().map(|(k, v)| (k, v.into_value())).collect()),
        }
    }
}

struct Semaphore {
    count: Mutex<i64>,
    wake: Condvar,
}

impl Semaphore {
    fn new(count: i64) -> Self {
        Self { count: Mutex::new(count), wake: Condvar::new() }
    }

    fn acquire(&self, timeout_ms: Option<u64>) -> bool {
        let mut count = self.count.lock().expect("semaphore lock poisoned");
        if *count > 0 {
            *count -= 1;
            return true;
        }
        if let Some(ms) = timeout_ms {
            let (guard, result) = self.wake.wait_timeout_while(
                count,
                Duration::from_millis(ms),
                |value| *value <= 0,
            ).expect("semaphore wait poisoned");
            count = guard;
            if result.timed_out() && *count <= 0 {
                return false;
            }
        } else {
            count = self.wake.wait_while(count, |value| *value <= 0)
                .expect("semaphore wait poisoned");
        }
        if *count > 0 {
            *count -= 1;
            true
        } else {
            false
        }
    }

    fn release(&self) {
        let mut count = self.count.lock().expect("semaphore lock poisoned");
        *count += 1;
        self.wake.notify_one();
    }
}

struct Registry {
    values: HashMap<u64, Arc<Mutex<SyncValue>>>,
    semaphores: HashMap<u64, Arc<Semaphore>>,
    senders: HashMap<u64, Sender<SyncValue>>,
    receivers: HashMap<u64, Arc<Mutex<Receiver<SyncValue>>>>,
}

static REGISTRY: OnceLock<Mutex<Registry>> = OnceLock::new();
static NEXT_HANDLE: AtomicU64 = AtomicU64::new(1);

fn registry() -> &'static Mutex<Registry> {
    REGISTRY.get_or_init(|| Mutex::new(Registry {
        values: HashMap::new(),
        semaphores: HashMap::new(),
        senders: HashMap::new(),
        receivers: HashMap::new(),
    }))
}

fn handle() -> u64 {
    NEXT_HANDLE.fetch_add(1, Ordering::Relaxed)
}

pub(crate) fn mutex_new(value: &Value) -> Result<u64, String> {
    let value = SyncValue::from_value(value)?;
    let id = handle();
    registry().lock().map_err(|_| "Nano sync: registry bloqueado".to_string())?
        .values.insert(id, Arc::new(Mutex::new(value)));
    Ok(id)
}

pub(crate) fn mutex_get(id: u64) -> Result<Value, String> {
    let cell = registry().lock().map_err(|_| "Nano sync: registry bloqueado".to_string())?
        .values.get(&id).cloned().ok_or_else(|| format!("Nano sync: mutex {id} não existe"))?;
    Ok(cell.lock().map_err(|_| format!("Nano sync: mutex {id} envenenado"))?.clone().into_value())
}

pub(crate) fn mutex_set(id: u64, value: &Value) -> Result<(), String> {
    let value = SyncValue::from_value(value)?;
    let cell = registry().lock().map_err(|_| "Nano sync: registry bloqueado".to_string())?
        .values.get(&id).cloned().ok_or_else(|| format!("Nano sync: mutex {id} não existe"))?;
    *cell.lock().map_err(|_| format!("Nano sync: mutex {id} envenenado"))? = value;
    Ok(())
}

pub(crate) fn mutex_swap(id: u64, value: &Value) -> Result<Value, String> {
    let value = SyncValue::from_value(value)?;
    let cell = registry().lock().map_err(|_| "Nano sync: registry bloqueado".to_string())?
        .values.get(&id).cloned().ok_or_else(|| format!("Nano sync: mutex {id} não existe"))?;
    let mut guard = cell.lock().map_err(|_| format!("Nano sync: mutex {id} envenenado"))?;
    let old = std::mem::replace(&mut *guard, value);
    Ok(old.into_value())
}

pub(crate) fn sync_remove(id: u64) -> bool {
    registry().lock().ok().map(|mut r| r.values.remove(&id).is_some()).unwrap_or(false)
}

pub(crate) fn semaphore_new(count: u64) -> u64 {
    let id = handle();
    if let Ok(mut r) = registry().lock() {
        r.semaphores.insert(id, Arc::new(Semaphore::new(count as i64)));
    }
    id
}

pub(crate) fn semaphore_acquire(id: u64, timeout_ms: Option<u64>) -> Result<bool, String> {
    let semaphore = registry().lock().map_err(|_| "Nano sync: registry bloqueado".to_string())?
        .semaphores.get(&id).cloned().ok_or_else(|| format!("Nano sync: semáforo {id} não existe"))?;
    Ok(semaphore.acquire(timeout_ms))
}

pub(crate) fn semaphore_release(id: u64) -> Result<(), String> {
    let semaphore = registry().lock().map_err(|_| "Nano sync: registry bloqueado".to_string())?
        .semaphores.get(&id).cloned().ok_or_else(|| format!("Nano sync: semáforo {id} não existe"))?;
    semaphore.release();
    Ok(())
}

pub(crate) fn semaphore_remove(id: u64) -> bool {
    registry().lock().ok().map(|mut r| r.semaphores.remove(&id).is_some()).unwrap_or(false)
}

pub(crate) fn channel_new() -> u64 {
    let (tx, rx) = mpsc::channel::<SyncValue>();
    let id = handle();
    if let Ok(mut r) = registry().lock() {
        r.senders.insert(id, tx);
        r.receivers.insert(id, Arc::new(Mutex::new(rx)));
    }
    id
}

pub(crate) fn channel_send(id: u64, value: &Value) -> Result<(), String> {
    let value = SyncValue::from_value(value)?;
    let sender = registry().lock().map_err(|_| "Nano sync: registry bloqueado".to_string())?
        .senders.get(&id).cloned().ok_or_else(|| format!("Nano sync: canal {id} não existe"))?;
    sender.send(value).map_err(|_| format!("Nano sync: canal {id} foi fechado"))
}

pub(crate) fn channel_recv(id: u64, timeout_ms: Option<u64>) -> Result<Value, String> {
    let receiver = registry().lock().map_err(|_| "Nano sync: registry bloqueado".to_string())?
        .receivers.get(&id).cloned().ok_or_else(|| format!("Nano sync: canal {id} não existe"))?;
    let value = receiver.lock().map_err(|_| format!("Nano sync: canal {id} envenenado"))?;
    let item = match timeout_ms {
        Some(ms) => value.recv_timeout(Duration::from_millis(ms)).map_err(|e| format!("Nano sync: recv_timeout: {e}"))?,
        None => value.recv().map_err(|e| format!("Nano sync: recv: {e}"))?,
    };
    Ok(item.into_value())
}

pub(crate) fn channel_try_recv(id: u64) -> Result<Option<Value>, String> {
    let receiver = registry().lock().map_err(|_| "Nano sync: registry bloqueado".to_string())?
        .receivers.get(&id).cloned().ok_or_else(|| format!("Nano sync: canal {id} não existe"))?;
    let value = receiver.lock().map_err(|_| format!("Nano sync: canal {id} envenenado"))?;
    match value.try_recv() {
        Ok(item) => Ok(Some(item.into_value())),
        Err(mpsc::TryRecvError::Empty) => Ok(None),
        Err(mpsc::TryRecvError::Disconnected) => Err(format!("Nano sync: canal {id} foi fechado")),
    }
}

pub(crate) fn channel_close(id: u64) {
    if let Ok(mut r) = registry().lock() {
        r.senders.remove(&id);
        r.receivers.remove(&id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shared_mutex_round_trip() {
        let id = mutex_new(&Value::Number(1.0)).unwrap();
        assert_eq!(mutex_get(id).unwrap(), Value::Number(1.0));
        assert_eq!(mutex_swap(id, &Value::Text("ok".into())).unwrap(), Value::Number(1.0));
        assert_eq!(mutex_get(id).unwrap(), Value::Text("ok".into()));
        assert!(sync_remove(id));
    }

    #[test]
    fn semaphore_timeout_and_release() {
        let id = semaphore_new(0);
        assert!(!semaphore_acquire(id, Some(1)).unwrap());
        semaphore_release(id).unwrap();
        assert!(semaphore_acquire(id, Some(1)).unwrap());
        assert!(semaphore_remove(id));
    }

    #[test]
    fn shared_channel_round_trip() {
        let id = channel_new();
        channel_send(id, &Value::Text("hello".into())).unwrap();
        assert_eq!(channel_recv(id, Some(10)).unwrap(), Value::Text("hello".into()));
        channel_close(id);
    }
}
