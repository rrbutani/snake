#![allow(incomplete_features)]

#![feature(non_ascii_idents)]
#![feature(const_generics)]
#![feature(const_fn)]
#![feature(const_if_match)]
#![feature(impl_trait_in_bindings)]
#![feature(bindings_after_at)]

#![deny(unsafe_code)]

use std::ops::Add;
use std::convert::TryInto;
use std::fmt::Display;

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Direction {
    Up,
    Down,
    Left,
    Right,
}

impl Direction {
    fn into_pair(self) -> (i32, i32) {
        use Direction::*;

        match self {
            Up => (0, -1),
            Down => (0, 1),
            Left => (-1, 0),
            Right => (1, 0),
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Coord {
    x: u16,
    y: u16,
}

impl From<(u16, u16)> for Coord {
    fn from((x, y): (u16, u16)) -> Self {
        Coord { x, y }
    }
}

impl Add<Direction> for Coord {
    type Output = Result<Coord, std::num::TryFromIntError>;

    fn add(self, rhs: Direction) -> Result<Coord, std::num::TryFromIntError> {
        let Coord { x, y } = self;
        let (Δx, Δy) = rhs.into_pair();

        Ok(Coord {
            x: ((x as i32) + Δx).try_into()?,
            y: ((y as i32) + Δy).try_into()?,
        })
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Status {
    Alive,
    Dead,
    Won,
}

pub trait SnakeGame<const SIZE: u16>: Display {
    fn new(seed: u64) -> Self;

    fn status(&self) -> Status;
    fn pos(&self) -> Coord;
    fn len(&self) -> usize;

    fn step(&mut self, dir: Direction) -> Status;
}


pub mod snake;
pub use snake::Game as SnekWithVecDeque;


