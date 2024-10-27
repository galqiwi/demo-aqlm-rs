use log::error;
use std::panic;

fn log_panic(info: &panic::PanicInfo) {
    let mut msg = info.to_string();
    msg.push_str("\n\n");

    error!("{}", msg);
}

pub fn setup_logger() {
    panic::set_hook(Box::new(log_panic));
    console_log::init_with_level(log::Level::Debug).unwrap();
}
