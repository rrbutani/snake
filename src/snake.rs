//! Snake, but faster!

use super::{Direction, Coord, SnakeGame, Status};

use std::mem::{self, MaybeUninit};
use std::fmt::{self, Display};
use std::ops::{Deref, DerefMut, Index, IndexMut};
use std::collections::VecDeque;

use rand::{rngs::StdRng, Rng, SeedableRng};

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Cell {
    Empty,
    Food,
    Snake,
    Head(Direction),
}

impl Cell {
    pub fn is_empty(&self) -> bool {
        if let Cell::Empty = self {
            true
        } else {
            false
        }
    }
}

impl Display for Cell {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        use Cell::*;
        use Direction::*;

        write!(fmt, "{}", match self {
            Empty => " ",
            Food => "🍎",
            Snake => "⬛",
            Head(Up) => "🔺",
            Head(Right) => "▶️",
            Head(Down) => "🔻",
            Head(Left) => "◀️",
        })
    }
}

// Bad hack to let us do this:
// ```
// impl<const SIZE: usize> Display for [Cell; SIZE] {}
// ```
//
// We want to do the above so that we can avoid sticking const generic params in
// method signatures.
trait DisplayProxy {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result;
}

impl<const SIZE: usize> DisplayProxy for &[Cell; SIZE] {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        write!(fmt, "┃")?;
        for cell in self.iter() {
            write!(fmt, "{}", cell)?
        }
        writeln!(fmt, "┃")
    }
}

// Same deal as above with this trait.
//
// This one actually removes the need for the above trait but w/e.
trait IntoCellIterator<'a>: 'a {
    type Iter: Iterator<Item = &'a Cell>;

    fn cell_iter(&self) -> Self::Iter;
}

impl<'a, const SIZE: usize> IntoCellIterator<'a> for &'a [Cell; SIZE] {
    type Iter = std::slice::Iter<'a, Cell>;

    fn cell_iter(&self) -> Self::Iter {
        self.iter()
    }
}


// type Map<const SIZE: u16> = [[Cell; SIZE as usize]; SIZE as usize];
// #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct Arr2D<T, const SIZE: usize>([[T; SIZE]; SIZE]);

impl<T: Default, const SIZE: usize> Arr2D<T, SIZE> {
    #[allow(unsafe_code)]
    pub /*const*/ fn new() -> Self {
        // This is safe because our array is of type `MaybeUninit` -- a type
        // that requires no initialization.
        //
        // See [this](https://doc.rust-lang.org/std/mem/union.MaybeUninit.html#initializing-an-array-element-by-element)
        // for more.
        let mut inner: [[MaybeUninit<T>; SIZE]; SIZE] = unsafe { MaybeUninit::uninit().assume_init() };

        // We now go initialize every element, guaranteeing that the array is
        // properly initialized.
        for row in &mut inner[..] {
            for cell in &mut row[..] {
                *cell = MaybeUninit::new(T::default());
            }
        }

        // In a nicer world (where const generics are more complete), we could
        // just go turn this array of `MaybeUninit`s into the real with with
        // mem::transmute, like this:

        /*```
        Self(unsafe {
            mem::transmute::<[[MaybeUninit<T>; SIZE]; SIZE], [[T; SIZE]; SIZE]>(inner)
        })
        ```*/

        // If we try to we get this lovely error:
        // ```
        // error[E0512]: cannot transmute between types of different sizes, or dependently-sized types
        //   --> src/snake.rs:59:13
        //    |
        // 58 |             mem::transmute::<[[MaybeUninit<T>; SIZE]; SIZE], [[T; SIZE]; SIZE]>(inner)
        //    |             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        //    |
        //    = note: source type: `[[std::mem::MaybeUninit<T>; _]; _]` (this type does not have a fixed size)
        //    = note: target type: `[[T; _]; _]` (this type does not have a fixed size)
        // ```
        //
        // So for now we use `transmute_copy` instead. It's precondition (that
        // the sizes we're _transmuting_ between are of the same size) is
        // guaranteed to be true since `MaybeUninit<T>` is guaranteed to be have
        // the same layout as `T` but for completeness we have an assert for
        // this anyways.

        assert_eq!(
            mem::size_of::<[[MaybeUninit<T>; SIZE]; SIZE]>(),
            mem::size_of::<[[T; SIZE]; SIZE]>()
        );

        Self(unsafe {
            mem::transmute_copy::<[[MaybeUninit<T>; SIZE]; SIZE], _>(&inner)
        })

        // Self([[T::default(); SIZE]; SIZE])
    }
}

impl<T: Copy, const SIZE: usize> Arr2D<T, SIZE> {
    #[allow(unsafe_code)]
    pub /*const*/ fn new_with(val: T) -> Self {
        // See the [`new`](Self::new) function's comments.

        // Note: we need all of this duplicated code because T could implement
        // [`Default`] but not [`Copy`].
        let mut inner: [[MaybeUninit<T>; SIZE]; SIZE] = unsafe { MaybeUninit::uninit().assume_init() };

        for row in &mut inner[..] {
            for cell in &mut row[..] {
                *cell = MaybeUninit::new(val);
            }
        }

        assert_eq!(
            mem::size_of::<[[MaybeUninit<T>; SIZE]; SIZE]>(),
            mem::size_of::<[[T; SIZE]; SIZE]>()
        );

        Self(unsafe {
            mem::transmute_copy::<[[MaybeUninit<T>; SIZE]; SIZE], _>(&inner)
        })

        // Self([[val; SIZE]; SIZE])
    }
}

impl<T, const SIZE: usize> Deref for Arr2D<T, SIZE> {
    type Target = [[T; SIZE]; SIZE];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T, const SIZE: usize> DerefMut for Arr2D<T, SIZE> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

// Ordinarily we wouldn't need these functions since we've implemented Deref,
// but the state of const generics limits where we can actually take advantage
// of Deref.
//
// Essentially, the compiler doesn't seem to be able to deal with traits (or
// bare functions?) that involve const generics in their signatures. So, we're
// implementing the functions that we actually need. And then implementing them
// again on Map. In other words, proxying the functions twice. It's dumb, but
// alas.
impl<T, const SIZE: usize> Index<Coord> for Arr2D<T, SIZE> {
    type Output = T;

    fn index(&self, idx: Coord) -> &T {
        let Coord { x, y }  = idx;

        &(self.0)[y as usize][x as usize]
    }
}

impl<T, const SIZE: usize> IndexMut<Coord> for Arr2D<T, SIZE> {
    fn index_mut(&mut self, idx: Coord) -> &mut T {
        let Coord { x, y }  = idx;

        &mut (self.0)[y as usize][x as usize]
    }
}

impl<T: Copy, const SIZE: usize> Arr2D<T, SIZE> {
    fn iter(&self) -> impl Iterator<Item = &[T; SIZE]> + '_ {
        self.0.iter()
    }
}

// #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct Map<const SIZE: u16>(Arr2D<Cell, {SIZE as usize}>);

//// Unfortunately, const generics don't let us do this yet! ////
// impl<const SIZE: u16> Deref for Map<SIZE> {
//     type Target = Arr2D<Cell, {SIZE as usize}>;

//     fn deref(&self) -> &Self::Target {
//         &self.0
//     }
// }

// impl<const SIZE: u16> DerefMut for Map<SIZE> {
//     fn deref_mut(&mut self) -> &mut Self::Target {
//         &mut self.0
//     }
// }

impl<const SIZE: u16> Map<SIZE> {
    fn new() -> Self {
        Map(Arr2D::new_with(Cell::Empty))
    }
}

// Second round of proxying (see above).
impl<const SIZE: u16> Index<Coord> for Map<SIZE> {
    type Output = Cell;

    fn index(&self, idx: Coord) -> &Self::Output {
        let Coord { x, y }  = idx;

        self.0.index(idx)
    }
}

impl<const SIZE: u16> IndexMut<Coord> for Map<SIZE> {
    fn index_mut(&mut self, idx: Coord) -> &mut Self::Output {
        let Coord { x, y }  = idx;

        self.0.index_mut(idx)
        // &mut (*self.0)[y as usize][x as usize]
        // &mut self.0[y as usize][x as usize]
    }
}

impl<const SIZE: u16> Map<SIZE> {
    // fn iter(&self) -> impl Iterator<Item = impl DisplayProxy + '_> + '_ {
    // fn iter<'a>(&'a self) -> impl Iterator<Item = impl IntoIterator<Item = Cell> + Sized + 'a> + '_ {
    fn iter<'a>(&'a self) -> impl Iterator<Item = impl DisplayProxy + IntoCellIterator<'a> + 'a> + '_ {
        self.0.iter()
    }
}

pub struct Game<const SIZE: u16, const MAP: bool> {
    // map: Map<SIZE>,
    // map: [[Cell; {SIZE as usize}]; {SIZE as usize}],
    map: Option<Map<SIZE>>, // If you want size savings, Box this.

    snake: VecDeque<Coord>,
    food: Option<Coord>,

    current_direction: Direction,
    rng: StdRng,

    state: Status,
}

impl<const SIZE: u16, const MAP: bool> SnakeGame<SIZE> for Game<SIZE, MAP> {
    fn new(seed: u64) -> Self {
        Self::new(seed)
    }

    fn status(&self) -> Status {
        self.state
    }

    fn pos(&self) -> Coord {
        *self.snake.front().unwrap()
    }

    fn len(&self) -> usize {
        self.snake.len()
    }

    fn step(&mut self, dir: Direction) -> Status {
        Game::step(self, dir)
    }
}

impl<const SIZE: u16, const MAP: bool> Game<SIZE, MAP> {
    const CELLS: usize = (SIZE as usize) * (SIZE as usize);

    fn new(seed: u64) -> Self {
        let mut rng: StdRng = SeedableRng::seed_from_u64(seed);

        let mut game = Self {
            map: None,
            snake: VecDeque::with_capacity(Self::CELLS),
            food: None,
            current_direction: Direction::Right,
            rng,
            state: Status::Alive,
        };

        let snake_pos = (SIZE / 2, SIZE / 2).into();
        game.snake.push_back(snake_pos);

        if MAP {
            game.map = Some(Map::new());
            game.map.as_mut().unwrap()[snake_pos] = Cell::Head(game.current_direction);
        }

        let food_pos = game.rand_unfilled_pos();
        game.food = Some(food_pos);

        if MAP {
            game.map.as_mut().unwrap()[food_pos] = Cell::Food;
        }

        game
    }

    fn step(&mut self, dir: Direction) -> Status {
        if let Status::Dead | Status::Won = self.state {
            return self.state;
        }

        self.current_direction = dir;

        let head = *self.snake.front().unwrap();
        let new_pos = head + dir;

        let new_pos = match new_pos {
            Ok(p @ Coord { x, y }) if x < SIZE && y < SIZE => p,
            _ => {
                self.state = Status::Dead;
                return self.state;
            }
        };

        if new_pos == self.food.unwrap() {
            // Grow by growing up front:
            self.snake.push_front(new_pos);

            if MAP {
                let map = self.map.as_mut().unwrap();
                map[head] = Cell::Snake;
                map[new_pos] = Cell::Head(dir);
            }

            // If we haven't yet won the game, add new food:
            if self.snake.len() == Self::CELLS {
                self.state = Status::Won;
                return self.state;
            } else {
                let food_pos = self.rand_unfilled_pos();

                self.food = Some(food_pos);

                if MAP {
                    self.map.as_mut().unwrap()[food_pos] = Cell::Food;
                }
            }
        } else if self.is_occupied(new_pos) {
            // If we didn't hit food but the cell is occupied, we've hit
            // ourselves!
            self.state = Status::Dead;
            return self.state;
        } else {
            // In the normal case, pop from the back and add to the front.
            self.snake.push_front(new_pos);
            let old_tail = self.snake.pop_back().unwrap();

            if MAP {
                let map = self.map.as_mut().unwrap();
                map[new_pos] = Cell::Head(dir);
                map[head] = Cell::Snake;
                map[old_tail] = Cell::Empty;
                // In the case that the snake is only 1 cell long, old_tail will
                // be equal to head and everything will be fine.
            }
        }

        return self.state;
    }

    fn is_occupied(&self, pos: Coord) -> bool {
        if MAP {
            !self.map.as_ref().unwrap()[pos].is_empty()
        } else {
            // If we aren't maintaining a map, to check that a position is free
            // we need to scan through the snake's cells and check the food's
            // position.
            if let Some(p) = self.food {
                if p == pos { return true; }
            }

            self.snake.iter().all(|p| *p != pos)
        }
    }

    fn rand_unfilled_pos(&mut self) -> Coord {
        let rand_pos: impl Fn(&mut StdRng) -> Coord =
            |r| (r.gen_range(0, SIZE as usize) as u16, r.gen_range(0, SIZE as usize) as u16).into();

        // There are two main approaches here and a few factors to consider.
        //
        // The first approach is to randomly pick a point, check if said point
        // is unoccupied, and repeat until we have an unoccupied point.
        //
        // The second approach is pick a number (N) from [0, <number of empty
        // points>) and then to traverse the grid (top to bottom, left to right)
        // until we reach the the Nth empty cell.
        //
        // For the factors to consider we've got:
        //   - Number of open cells (F) -- free
        //   - Number of occupied cells (N) -- not free
        //   - Size of the grid (S) -- size
        //   - Whether or not we've got a map*
        //
        // *Whether or not we maintain a map as we go is currently decided by
        // the MAP const generic parameter. If this is set to false, we don't
        // currently construct a map in this function, though that is a
        // potentially interesting idea: is it more expensive to amortize the
        // cost of having an updated map or to construct the map as needed?
        // I think the answer is that it depends on how the user is playing: if
        // the user is playing 'badly' (lots of turns to get to the next food
        // pellet) then we could do more work maintaining the map than it would
        // take to construct the map anew.
        //
        // Also note that the cost of checking if a cell is occupied is O(N) if
        // there isn't a map.
        //
        // We'll start with the case where we have a map. The second approach is
        // guaranteed to get us a position in 1 go. But, to do so it must -- in
        // the worst case -- check the entire grid: an O(S) operation.
        //
        // If we're lucky, the first approach works right away. With a map
        // checking if a cell is occupied is an O(1) operation which means in
        // the best case, the first approach is O(1) -- since we're assuming
        // picking the random cell in the first place is also O(1). However,
        // as the grid fills up, it becomes increasingly unlikely that we'll
        // find an unoccupied cell in just one guess.
        //
        // The odds of a single guess being good are F / S.
        // The odds that it takes K guesses to find an unoccupied cell are
        // (N / S)^(max(0, K - 1)) * (F / S).
        //
        // The expected value for the number of guesses we need to take (K) is:
        //
        //   E[K] = 1 + (N / S) * E[K]
        //          |    \   /     \
        //          |     \ /       \-- Number of guesses we'll need once we try
        //          |      |            again.
        //          |       \- Odds the we picked an occupied cell.
        //           \- The current guess.
        //
        //   E[K] - (N / S) * E[K] = 1
        //   E[K](1 - (N / S)) = 1
        //   E[K] = 1 / (1 - (N / S))
        //   E[K] = 1 / (F / S)
        //   E[K] = (S / F)
        //
        // Calculating the odds for the amount of work the second approach will
        // have to do is a little more complicated since there are a few factors
        // that affect this:
        //   - the number (index) that's picked (X) in [0, F)
        //   - the number of occupied cells
        //   - the _distribution_ of the occupied cells
        //
        // To find an unoccupied cell with this approach we need to process N
        // _unoccupied_ cells *and* the _occupied_ cells that are between the
        // cell that's chosen and the top left corner in left to right, top to
        // bottom order. This will depend on where the snake is on the grid
        // (it won't depend on the food pellet's locations since we're only
        // asking for an unoccupied cell if there isn't a food pellet currently
        // on the map -- but that doesn't really matter anyways).
        //
        // As a first-order estimate, we'll assume that the occupied cells are
        // distributed evenly relative to a left to right, top to bottom order.
        //
        // For example, if there are 36 cells with 8 occupied (and 28 empty)
        // and if X is 12, in the best case we have to go through just 12 cells.
        // In the worst case we have to go through 20 cells (12 + all of the 8
        // occupied cells). Our estimate predicts that we go through the 12
        // cells and (12 / 28) * 8 or 3 occupied cells for a total of 15 cells.
        //
        //   E[K] = X + N * (X / F)
        //
        //
        // So, we've got all this math. But it's not immediately clear how this
        // translates to actually making the guesses.
        //
        // 'Guessing' (the first approach) some number of times and then
        // switching to the second approach is likely the right strategy but
        // it's not clear _how many_ times to guess. It's also worth noting that
        // without profiling we don't actually know how long one guess takes
        // relative to one step (the unit of work for the first and second
        // approaches, respectively).
        //
        // One (illegal) strategy would be to pick X before deciding between
        // the approaches (i.e. if X is high the second approach is expensive
        // so don't use it). We cannot do this since it biases towards 'indexed'
        // cells.
        //
        // It's not entirely clear to me how we even use this math. For example,
        // does a low expected number of guesses for the first approach mean
        // that we spend more attempts on it before cutting our losses? Is there
        // a threshold at which we skip guessing altogether?
        //
        // What is clear is that the N / F part of the second approach's formula
        // is the only part that's useful to us since (as mentioned) we can't
        // know X before we decide.
        //
        // I think we can compute an optimal strategy. Let's say we've got a
        // grid with 36 cells (6 x 6). For now, we'll say the cost the unit of
        // work in each approach is the same.
        //
        // For our first try let's go with a strategy where we guess 5 times and
        // then, if those guesses didn't work, we run the second approach.
        //
        // For this strategy, for a particular (N, F, X) tuple we can calculate
        // how many units of work we expect to take.
        //
        //   1 guesses: (N / S) ^ 0 * (F / S) * 1
        //   2 guesses: (N / S) ^ 1 * (F / S) * 2
        //   3 guesses: (N / S) ^ 2 * (F / S) * 3
        //   4 guesses: (N / S) ^ 3 * (F / S) * 4
        //   5 guesses: (N / S) ^ 4 * (F / S) * 5
        //   6 guesses: (N / S) ^ 5 * (X + N * (X / F))
        //   ...summed up. It's essentially a partial summation of the formula
        //   in the 9th paragraph + the estimate for the 6th approach.
        //
        // We can then sum up all this expected work values for each (N, F, X)
        // tuple such that:
        //   - N + F = S
        //   - X ∈ [0, F)
        //
        // Let's do it!
        // ```python
        // from typing import Callable
        //
        // def sum_across_all_values(size: int, score: Callable[[int, int, int], float]) -> float:
        //   sum: float = 0
        //   count: int = 0
        //
        //   for F in range(size):
        //     N: int = size - F  # All cells that aren't free are occupied
        //
        //     inner_sum: float = 0
        //     # This will actually be a 0-based idx, but for work units we
        //     # want it to start at 1.
        //     for X in range(1, F + 1):
        //       inner_sum += score(N, F, X)
        //     sum += (inner_sum / max(1, F))
        //
        //   return (sum / size)
        //
        // def guess_K_times(K: int, N: int, F: int, X: int) -> float:
        //   guesses = sum(map(lambda x: ((N / S) ** x) * (F / S) * (x + 1), range(K)))
        //   return guesses + ((N / S) ** K) * (X + N * (X / F))
        //
        // print([(num_guesses, sum_across_all_values(36, lambda N, F, X: guess_K_times(num_guesses, N, F, X))) for num_guesses in range(10)])
        // ```
        //
        // The above compares the amount of work you have to do, averaged across
        // all the possible (N, F, X) pairs for strategies where you guess a
        // particular amount of times (K) before switching for all the possible
        // K values for a given size (K ∈ [0, S)). We seem to reach a minimum
        // around 78% of S.
        //
        // Here are the results for S = 36 and S = 100, rounded.
        //
        // 36:
        // ```
        // [(0, 31), (1, 17), (2, 12), (3, 9), (4, 8), (5, 7), (6, 6), (7, 6),
        //  (8, 5), (9, 5), (10, 5), (11, 5), (12, 5), (13, 4), (14, 4),
        //  (15, 4), (16, 4), (17, 4), (18, 4), (19, 4), (20, 4), (21, 4),
        //  (22, 4), (23, 4), (24, 4), (25, 4), (26, 4), (27, 4), (28, 4),
        //  (29, 4), (30, 4), (31, 4), (32, 4), (33, 4), (34, 4), (35, 4)]
        // ```
        //
        // 100:
        // ```
        // [(0, 52), (1, 27), (2, 19), (3, 15), (4, 12), (5, 10), (6, 9),
        //  (7, 9), (8, 8), (9, 7), (10, 7), (11, 7), (12, 6), (13, 6), (14, 6),
        //  (15, 6), (16, 6), (17, 5), (18, 5), (19, 5), (20, 5), (21, 5),
        //  (22, 5), (23, 5), (24, 5), (25, 5), (26, 5), (27, 5), (28, 5),
        //  (29, 5), (30, 5), (31, 5), (32, 4), (33, 4), (34, 4), (35, 4),
        //  (36, 4), (37, 4), (38, 4), (39, 4), (40, 4), (41, 4), (42, 4),
        //  (43, 4), (44, 4), (45, 4), (46, 4), (47, 4), (48, 4), (49, 4),
        //  (50, 4), (51, 4), (52, 4), (53, 4), (54, 4), (55, 4), (56, 4),
        //  (57, 4), (58, 4), (59, 4), (60, 4), (61, 4), (62, 4), (63, 4),
        //  (64, 4), (65, 4), (66, 4), (67, 4), (68, 4), (69, 4), (70, 4),
        //  (71, 4), (72, 4), (73, 4), (74, 4), (75, 4), (76, 4), (77, 4),
        //  (78, 4), (79, 4), (80, 4), (81, 4), (82, 4), (83, 4), (84, 4),
        //  (85, 4), (86, 4), (87, 4), (88, 4), (89, 4), (90, 4), (91, 4),
        //  (92, 4), (93, 4), (94, 4), (95, 4), (96, 4), (97, 4), (98, 4),
        //  (99, 4)]
        // ```
        //
        // Note: Another interesting thing to consider is how long we're likely
        // each (N, F, X) tuple is. Luckily we have a definite answer for this!
        // for a fixed N and F, every valid (N, F, X) pair is equally likely (so
        // averaging the cost of each -- as we do above -- seems sound). For
        // each (N, F) pair -- which is really completely defined by just N or
        // just F -- we know that we will encounter each *exactly once* in the
        // course of a game (the only reason we ask for an unoccupied cell is
        // when a food pellet needs to be spawned which only happens when the
        // snake eats the current food pellet; when the snake eats a pellet it
        // grows increasing N and decreasing F by exactly 1 apiece). So really,
        // returning sum without dividing by size in the `sum_across_all_values`
        // function above may be more different (but isn't materially different
        // in the end -- it will make comparisons easier though).
        //
        // Here are the results without sum being divided for 36:
        // ```
        // [(0, 704), (1, 389), (2, 284), (3, 234), (4, 204), (5, 185),
        //  (6, 172), (7, 163), (8, 156), (9, 151), (10, 147), (11, 143),
        //  (12, 141), (13, 138), (14, 137), (15, 135), (16, 134), (17, 133),
        //  (18, 132), (19, 132), (20, 131), (21, 131), (22, 131), (23, 130),
        //  (24, 130), (25, 130), (26, 130), (27, 130), (28, 130), (29, 130),
        //  (30, 130), (31, 130), (32, 130), (33, 130), (34, 130), (35, 131)]
        // ```
        //
        // So, for S = 36, 130 guesses is the number to beat (28 guesses).
        //
        // The next thing to try is a more sophisticated strategy. First let's
        // look at where those 130 guesses are coming from:
        //
        // (added `print(f"{N:2}, {F:2} -> {(inner_sum / max(1, F))}")` to
        // `sum_across_all_values` and called
        // `sum_across_all_values(36, lambda N, F, X: guess_K_times(28, N, F,
        // X))` )
        //
        // ```
        //  N   F    Cost
        // 36,  0 -> 0.0
        // 35,  1 -> 23.276890887756615
        // 34,  2 -> 14.165636722698888
        // 33,  3 -> 10.600282869546783
        // 32,  4 -> 8.464083197380324
        // 31,  5 -> 6.993368601882629
        // 30,  6 -> 5.921137877236606
        // 29,  7 -> 5.1133410714856256
        // 28,  8 -> 4.489233093421905
        // 27,  9 -> 3.99619024874304
        // 26, 10 -> 3.598697835866698
        // 25, 11 -> 3.2722990521764035
        // 24, 12 -> 2.999865059415557
        // 23, 13 -> 2.7691901971062123
        // 22, 14 -> 2.5714169864804632
        // 21, 15 -> 2.3999968747248874
        // 20, 16 -> 2.2499992081008506
        // 19, 17 -> 2.1176468716089225
        // 18, 18 -> 1.9999999590218067
        // 17, 19 -> 1.8947368338750803
        // 16, 20 -> 1.7999999984992232
        // 15, 21 -> 1.7142857140403571
        // 14, 22 -> 1.6363636363282155
        // 13, 23 -> 1.5652173912999146
        // 12, 24 -> 1.4999999999995302
        // 11, 25 -> 1.439999999999959
        // 10, 26 -> 1.3846153846153828
        //  9, 27 -> 1.3333333333333328
        //  8, 28 -> 1.285714285714285
        //  7, 29 -> 1.2413793103448263
        //  6, 30 -> 1.2
        //  5, 31 -> 1.1612903225806452
        //  4, 32 -> 1.1250000000000002
        //  3, 33 -> 1.0909090909090904
        //  2, 34 -> 1.058823529411765
        //  1, 35 -> 1.0285714285714282
        // ```
        //
        // The above tells us that we're 'paying' the most where N = 35, 32, 33
        // and so on.
        //
        // So, maybe we try to switch to approach two more eagerly in those
        // situations. Maybe we jump straight to approach two above a certain
        // threshold for N; let's say when N / S > 0.8 for now.
        //
        // ```python
        // def approach_two_above_threshold(threshold: float, N: int, F: int, X: int) -> float:
        //  S = N + F
        //  if (N / S) > threshold:
        //    return (X + N * (X / F))
        //  else:
        //    return guess_K_times(int(0.78 * S), N, F, X)
        //
        // sum_across_all_values(36, lambda N, F, X: approach_two_above_threshold(0.8, N, F, X))
        // ```
        //
        // In the above, we're calling guess_K_times again for convenience;
        // really we mean just guess again and again. The probabilities aren't
        // materially different and the above executes faster than calculating
        // infinite sums (and I'm too lazy to find a closed form :-/).
        //
        // The above yields (printed alongside the previous profile):
        // ```
        //  N   F    Cost (Guess 28 times)  N   F    Cost (2nd approach < 0.8)
        // 36,  0 -> 0.0                   36,  0 -> 0.0
        // 35,  1 -> 23.276890887756615    35,  1 -> 36.0
        // 34,  2 -> 14.165636722698888    34,  2 -> 27.0
        // 33,  3 -> 10.600282869546783    33,  3 -> 24.0
        // 32,  4 -> 8.464083197380324     32,  4 -> 22.5
        // 31,  5 -> 6.993368601882629     31,  5 -> 21.6
        // 30,  6 -> 5.921137877236606     30,  6 -> 21.0
        // 29,  7 -> 5.1133410714856256    29,  7 -> 20.571428571428573
        // 28,  8 -> 4.489233093421905     28,  8 -> 4.489233093421905
        // 27,  9 -> 3.99619024874304      27,  9 -> 3.99619024874304
        // 26, 10 -> 3.598697835866698     26, 10 -> 3.598697835866698
        // 25, 11 -> 3.2722990521764035    25, 11 -> 3.2722990521764035
        // 24, 12 -> 2.999865059415557     24, 12 -> 2.999865059415557
        // 23, 13 -> 2.7691901971062123    23, 13 -> 2.7691901971062123
        // 22, 14 -> 2.5714169864804632    22, 14 -> 2.5714169864804632
        // 21, 15 -> 2.3999968747248874    21, 15 -> 2.3999968747248874
        // 20, 16 -> 2.2499992081008506    20, 16 -> 2.2499992081008506
        // 19, 17 -> 2.1176468716089225    19, 17 -> 2.1176468716089225
        // 18, 18 -> 1.9999999590218067    18, 18 -> 1.9999999590218067
        // 17, 19 -> 1.8947368338750803    17, 19 -> 1.8947368338750803
        // 16, 20 -> 1.7999999984992232    16, 20 -> 1.7999999984992232
        // 15, 21 -> 1.7142857140403571    15, 21 -> 1.7142857140403571
        // 14, 22 -> 1.6363636363282155    14, 22 -> 1.6363636363282155
        // 13, 23 -> 1.5652173912999146    13, 23 -> 1.5652173912999146
        // 12, 24 -> 1.4999999999995302    12, 24 -> 1.4999999999995302
        // 11, 25 -> 1.439999999999959     11, 25 -> 1.439999999999959
        // 10, 26 -> 1.3846153846153828    10, 26 -> 1.3846153846153828
        //  9, 27 -> 1.3333333333333328     9, 27 -> 1.3333333333333328
        //  8, 28 -> 1.285714285714285      8, 28 -> 1.285714285714285
        //  7, 29 -> 1.2413793103448263     7, 29 -> 1.2413793103448263
        //  6, 30 -> 1.2                    6, 30 -> 1.2
        //  5, 31 -> 1.1612903225806452     5, 31 -> 1.1612903225806452
        //  4, 32 -> 1.1250000000000002     4, 32 -> 1.1250000000000002
        //  3, 33 -> 1.0909090909090904     3, 33 -> 1.0909090909090904
        //  2, 34 -> 1.058823529411765      2, 34 -> 1.058823529411765
        //  1, 35 -> 1.0285714285714282     1, 35 -> 1.0285714285714282
        // ```
        // ...for a total score of 229 guesses (compared to 130 for the previous
        // strategy).
        //
        // So, this strategy did uniformly worse than the previous strategy. If
        // you look at the numbers this makes sense; the lower bound for number
        // of unit work steps the 2nd approach has to do is X (which itself has
        // a *low* upper bound for high threshold values) but unfortunately the
        // cost of skipping over occupied cells hurts the 2nd approach too much.
        //
        // Here's a profile showing the cost of individual X values too:
        // ```
        // >>>> 36,  0 -> 0.0
        // 35,  1,  1 -> 36.0
        // >>>> 35,  1 -> 36.0
        // 34,  2,  1 -> 18.0
        // 34,  2,  2 -> 36.0
        // >>>> 34,  2 -> 27.0
        // 33,  3,  1 -> 12.0
        // 33,  3,  2 -> 24.0
        // 33,  3,  3 -> 36.0
        // >>>> 33,  3 -> 24.0
        // 32,  4,  1 -> 9.0
        // 32,  4,  2 -> 18.0
        // 32,  4,  3 -> 27.0
        // 32,  4,  4 -> 36.0
        // >>>> 32,  4 -> 22.5
        // 31,  5,  1 -> 7.2
        // 31,  5,  2 -> 14.4
        // 31,  5,  3 -> 21.599999999999998
        // 31,  5,  4 -> 28.8
        // 31,  5,  5 -> 36.0
        // >>>> 31,  5 -> 21.6
        // 30,  6,  1 -> 6.0
        // 30,  6,  2 -> 12.0
        // 30,  6,  3 -> 18.0
        // 30,  6,  4 -> 24.0
        // 30,  6,  5 -> 30.0
        // 30,  6,  6 -> 36.0
        // >>>> 30,  6 -> 21.0
        // 29,  7,  1 -> 5.142857142857142
        // 29,  7,  2 -> 10.285714285714285
        // 29,  7,  3 -> 15.428571428571427
        // 29,  7,  4 -> 20.57142857142857
        // 29,  7,  5 -> 25.714285714285715
        // 29,  7,  6 -> 30.857142857142854
        // 29,  7,  7 -> 36.0
        // >>>> 29,  7 -> 20.571428571428573
        // ```
        //
        // Also illuminating is seeing the scores when the threshold is 0 (i.e
        // always go with the 2nd approach):
        //
        // ```
        // 36,  0 -> 0.0
        // 35,  1 -> 36.0
        // 34,  2 -> 27.0
        // 33,  3 -> 24.0
        // 32,  4 -> 22.5
        // 31,  5 -> 21.6
        // 30,  6 -> 21.0
        // 29,  7 -> 20.571428571428573
        // 28,  8 -> 20.25
        // 27,  9 -> 20.0
        // 26, 10 -> 19.8
        // 25, 11 -> 19.636363636363637
        // 24, 12 -> 19.5
        // 23, 13 -> 19.384615384615383
        // 22, 14 -> 19.285714285714285
        // 21, 15 -> 19.2
        // 20, 16 -> 19.125
        // 19, 17 -> 19.058823529411764
        // 18, 18 -> 19.0
        // 17, 19 -> 18.94736842105263
        // 16, 20 -> 18.9
        // 15, 21 -> 18.857142857142854
        // 14, 22 -> 18.818181818181817
        // 13, 23 -> 18.782608695652176
        // 12, 24 -> 18.75
        // 11, 25 -> 18.720000000000002
        // 10, 26 -> 18.692307692307693
        //  9, 27 -> 18.666666666666668
        //  8, 28 -> 18.642857142857142
        //  7, 29 -> 18.620689655172413
        //  6, 30 -> 18.6
        //  5, 31 -> 18.580645161290324
        //  4, 32 -> 18.5625
        //  3, 33 -> 18.545454545454547
        //  2, 34 -> 18.529411764705884
        //  1, 35 -> 18.514285714285712
        // ```
        //
        // We see to converge on about half of the size of the grid.
        //
        // So, it seems the 2nd approach does *categorically* worse than the 2nd
        // approach regardless of N/F values. As in, even for the N/F (going to
        // just say N from here on since, as mentioned above, F is uniquely
        // defined by N and vice versa) value that's most favorable for the 2nd
        // approach, the score for the 1st approach performs better.
        //
        // Since the 1st approach's performance worst performance is inverse
        // exponentially correlated with the size while the 2nd approach's
        // best performance linearly related (~1/2 of the size), I think the
        // above holds regardless of input size.
        //
        // Since N is our only parameter that we're allowed to use to pick
        // an approach (F is determined by N and we're not allowed to use X as
        // mentioned above) and since the first approach is better then the 2nd
        // for effectively all N values, I think this concludes the strategy
        // search. Other than N and the number of guesses we've made thus far,
        // I can't think of any other reasonable heuristics to use (current
        // position of the snake and thereby distribution of the occupied cells
        // is one that's relevant to the performance of the 2nd approach but
        // using that as a heuristic is likely to compromise randomness and be
        // expensive to meaningfully quantify in the first place).
        //
        // For fun, we'll keep the 2nd approach in above 78% of size # guesses,
        // anyways even though its unlikely to manifest except for absurdly
        // large sizes.
        //
        // Note that the above still makes a couple of (potentially significant)
        // assumptions:
        //   - it's as costly to check one guess as it is to get that the next
        //     cell in left to right, top to bottom order is unoccupied
        //   - rng cost is insignificant
        //   - we're likely to encounter a number of occupied cells proportional
        //     to the number of unoccupied cells we encounter when going through
        //     the grid in left to right, top to bottom order
        let guesses = Self::CELLS as f32;
        let mut guesses = (0.78f32 * guesses).ceil() as usize;

        // For the case where we do not eagerly maintain a map, we'll just
        // create a map here.
        //
        // If we do not, the cost to check a single guess rises to the number
        // of occupied cells which is essentially the cost to create a map,
        // modulo any memory allocation costs (should just be on the stack).
        let map = self.update_map();

        while guesses > 0 {
            let pos = rand_pos(&mut self.rng);
            if !self.is_occupied(pos) { return pos; }

            guesses -= 1;
        }

        let idx = Self::CELLS - self.snake.len();
        let idx = self.rng.gen_range(0, idx);

        let mut cell_iter = self.map.as_ref().unwrap().iter()
            .enumerate()
            .flat_map(|(r, row)| row.cell_iter().enumerate().map(move |(c, cell)| ((r as u16, c as u16).into(), cell)))
            .filter(|(_, c)| (*c).is_empty());
            // .filter(Cell::is_empty);

        cell_iter.nth(idx).unwrap().0
    }
}

impl<const SIZE: u16, const MAP: bool> Game<SIZE, MAP> {
    fn update_map(&mut self) -> &Map<SIZE> {
        if MAP {
            self.map.as_ref().unwrap()
        } else {
            self.map = Some(self.make_map());
            self.map.as_ref().unwrap()
        }
    }

    fn make_map(&self) -> Map<SIZE> {
        let mut map = Map::new();

        let mut snek = self.snake.iter();
        let head = snek.next().unwrap();

        map[*head] = Cell::Head(self.current_direction);

        for pos in snek {
            // (**map)[pos] = Cell::Snake;
            map[*pos] = Cell::Snake;
        }

        if let Some(pos) = self.food {
            map[pos] = Cell::Food;
        }

        map
    }
}

// impl<const SIZE: u16> Game<SIZE, true> {
//     fn update_map(&mut self) -> &Map<SIZE> {
//         &self.map.unwrap()
//     }
// }

// impl<const SIZE: u16> Game<SIZE, false> {
//     fn update_map(&mut self) -> &Map<SIZE> {
//         self.map = Some(self.make_map());

//         &self.map.unwrap()
//     }

//     fn make_map(&self) -> Map<SIZE> {
//         let mut map = Map::new();

//         let mut snek = self.snake.iter();
//         let head = snek.next().unwrap();

//         map[*head] = Cell::Head(self.current_direction);

//         for pos in snek {
//             // (**map)[pos] = Cell::Snake;
//             map[*pos] = Cell::Snake;
//         }

//         if let Some(pos) = self.food {
//             map[pos] = Cell::Food;
//         }

//         map
//     }
// }

impl<const SIZE: u16, const MAP: bool> Game<SIZE, MAP> {
    fn print(fmt: &mut fmt::Formatter, map: &Map<SIZE>) -> fmt::Result {
        write!(fmt, "┏")?;
        for _ in 0..SIZE { write!(fmt, "━")? }
        writeln!(fmt, "┓")?;

        for row in map.iter() {
            row.fmt(fmt)?
        }

        write!(fmt, "┗")?;
        for _ in 0..SIZE { write!(fmt, "━")? }
        writeln!(fmt, "┛")
    }
}

impl<const SIZE: u16, const MAP: bool> Display for Game<SIZE, MAP> {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        // Self::print(fmt, self.update_map())
        if MAP {
            let map = self.make_map();
            Self::print(fmt, &map)
        } else {
            Self::print(fmt, &self.map.as_ref().unwrap())
        }
    }
}

// impl<const SIZE: u16> Display for Game<SIZE, false> {
//     fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
//         let map = self.make_map();

//         Self::print(fmt, &map)
//     }
// }

// impl<const SIZE: u16> Display for Game<SIZE, true> {
//     fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
//         Self::print(fmt, &self.map.as_ref().unwrap())
//     }
// }
