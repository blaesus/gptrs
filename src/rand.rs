use std::sync::Mutex;

// Constants for the LCG (These are common values used by numerical recipes)
const A: u64 = 1664525; // Multiplier
const C: u64 = 1013904223; // Increment
const M: f64 = (2u64.pow(32)) as f64; // Modulus (2^32)

// RNG State: wrapped in a Mutex for thread-safe mutable access
static mut RNG_STATE: Mutex<u64> = Mutex::new(123456789); // Arbitrary initial seed

/// Generates a random f32 number between 0.0 and 1.0
pub fn random_f32() -> f32 {
    // Safe access to the mutable static
    let mut rng = unsafe { RNG_STATE.lock().unwrap() };
    // LCG formula: (A * seed + C) % M
    *rng = (A * (*rng) + C) % (M as u64);
    // Return the random number as a f32 in the range [0, 1)
    (*rng as f32) / (M as f32)
}
