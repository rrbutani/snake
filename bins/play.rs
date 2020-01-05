#![feature(duration_constants)]

use snake::{SnakeGame, SnekWithVecDeque, Direction, Status};

use std::io::Read;
use std::sync::mpsc::{channel, Sender, Receiver};

// fn play() {
//     let mut s = SnekWithVecDeque::<8, true>::new(20);
//     let mut dir = Direction::Right;

//     let stdin = std::io::stdin();
//     let mut handle = stdin.lock();

//     let mut buf = [0; 256];

//     while let Status::Alive = s.step(dir) {
//         println!("{}\nScore: {:3}\nPosition: {:?}", s, s.len(), s.pos());

//         if let Ok(num) = handle.read(&mut buf) {
//             for b in buf[0..num].iter() {
//                 match <char as From<u8>>::from(*b).to_ascii_lowercase() {
//                     'w' => dir = Direction::Up,
//                     'a' => dir = Direction::Left,
//                     's' => dir = Direction::Down,
//                     'd' => dir = Direction::Right,
//                     _ => { },
//                 }
//             }
//         }

//         std::thread::sleep(std::time::Duration::SECOND);
//     }

//     println!("{:?}", s.status());
// }

fn play(rx: &Receiver<Direction>) {
    let mut s = SnekWithVecDeque::<8, true>::new(20);
    let mut dir = Direction::Right;

    while let Status::Alive = s.step(dir) {
        println!("{}\nScore: {:3}\nPosition: {:?}", s, s.len(), s.pos());

        if let Ok(d) = rx.try_recv() {
            dir = d;
        }

        std::thread::sleep(std::time::Duration::SECOND);
    }

    println!("{:?}", s.status());
}

fn main() {
    let (tx, rx) = channel();

    std::thread::spawn(move || {
        let stdin = std::io::stdin();
        let mut handle = stdin.lock();

        let mut buf = [0; 256];

        loop {
            if let Ok(num) = handle.read(&mut buf) {
                for b in buf[0..num].iter() {
                    match <char as From<u8>>::from(*b).to_ascii_lowercase() {
                        'w' => tx.send(Direction::Up).unwrap(),
                        'a' => tx.send(Direction::Left).unwrap(),
                        's' => tx.send(Direction::Down).unwrap(),
                        'd' => tx.send(Direction::Right).unwrap(),
                        _ => { },
                    }
                }
            }
        }
    });

    loop {
        play(&rx)
    }
}
