use std::sync::mpsc::{self, Receiver, Sender};
use std::thread;

use winit::{
    application::ApplicationHandler,
    dpi::LogicalSize,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, EventLoop},
    window::{Window, WindowId},
};

pub(crate) enum UiCommand {
    SetTitle(String),
    Close,
}

pub(crate) struct UiHandle {
    pub(crate) command: Sender<UiCommand>,
    pub(crate) events: Receiver<String>,
}

struct UiApp {
    title: String,
    width: f64,
    height: f64,
    commands: Receiver<UiCommand>,
    events: Sender<String>,
    window: Option<Window>,
    window_id: Option<WindowId>,
}

impl ApplicationHandler for UiApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }

        let attributes = Window::default_attributes()
            .with_title(self.title.clone())
            .with_inner_size(LogicalSize::new(self.width, self.height));

        match event_loop.create_window(attributes) {
            Ok(window) => {
                self.window_id = Some(window.id());
                let _ = self.events.send(format!("created:{}", format!("{:?}", window.id())));
                self.window = Some(window);
            }
            Err(error) => {
                let _ = self.events.send(format!("error:{error}"));
                event_loop.exit();
            }
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        if Some(window_id) != self.window_id {
            return;
        }

        match event {
            WindowEvent::CloseRequested => {
                let _ = self.events.send("close_requested".into());
                event_loop.exit();
            }
            WindowEvent::Resized(size) => {
                let _ = self.events.send(format!("resized:{}:{}", size.width, size.height));
            }
            WindowEvent::Focused(focused) => {
                let _ = self.events.send(format!("focused:{focused}"));
            }
            WindowEvent::CursorMoved { position, .. } => {
                let _ = self.events.send(format!("mouse_move:{:.3}:{:.3}", position.x, position.y));
            }
            WindowEvent::MouseInput { state, button, .. } => {
                let _ = self.events.send(format!("mouse_button:{button:?}:{state:?}"));
            }
            WindowEvent::KeyboardInput { event, .. } => {
                let _ = self.events.send(format!("key:{:?}:{:?}", event.logical_key, event.state));
            }
            WindowEvent::RedrawRequested => {
                let _ = self.events.send("redraw".into());
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        while let Ok(command) = self.commands.try_recv() {
            match command {
                UiCommand::SetTitle(title) => {
                    self.title = title.clone();
                    if let Some(window) = &self.window {
                        window.set_title(&title);
                    }
                }
                UiCommand::Close => {
                    let _ = self.events.send("closed".into());
                    event_loop.exit();
                    break;
                }
            }
        }
    }
}

pub(crate) fn spawn(title: String, width: f64, height: f64) -> Result<UiHandle, String> {
    if width <= 0.0 || height <= 0.0 {
        return Err("Nano UI: largura e altura devem ser positivas".into());
    }

    let (command_tx, command_rx) = mpsc::channel();
    let (event_tx, event_rx) = mpsc::channel();

    thread::Builder::new()
        .name("nano-ui".into())
        .spawn(move || {
            let Ok(event_loop) = EventLoop::new() else {
                let _ = event_tx.send("error:event-loop".into());
                return;
            };

            let mut app = UiApp {
                title,
                width,
                height,
                commands: command_rx,
                events: event_tx,
                window: None,
                window_id: None,
            };

            let _ = event_loop.run_app(&mut app);
        })
        .map_err(|e| format!("Nano UI: não foi possível criar thread: {e}"))?;

    Ok(UiHandle {
        command: command_tx,
        events: event_rx,
    })
}
