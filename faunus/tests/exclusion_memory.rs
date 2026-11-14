//! Building a system must take memory linear in the number of particles.
//! Own test binary, since the counting allocator is global.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};

use faunus::Simulation;

struct PeakAllocator;

static CURRENT: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

// SAFETY: forwards to the system allocator; only adds byte counting.
unsafe impl GlobalAlloc for PeakAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            let now = CURRENT.fetch_add(layout.size(), Relaxed) + layout.size();
            PEAK.fetch_max(now, Relaxed);
        }
        ptr
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) };
        CURRENT.fetch_sub(layout.size(), Relaxed);
    }
}

#[global_allocator]
static ALLOCATOR: PeakAllocator = PeakAllocator;

/// 3000 rigid 10-bead molecules: 30 000 particles, where any N×N byte table would need 900 MB.
#[test]
fn many_rigid_molecules_build_in_little_memory() {
    let beads: Vec<String> = (0..10).map(|i| format!("{{A: [{i}, 0, 0]}}")).collect();
    let yaml = format!(
        "\
atoms:
  - {{name: A, mass: 1.0, sigma: 1.0, eps: 0.1}}
molecules:
  - name: rod
    from_structure: [{}]
    degrees_of_freedom: Rigid
system:
  cell: !Cuboid [1000.0, 1000.0, 1000.0]
  medium: {{permittivity: !Vacuum, temperature: 298.15}}
  energy:
    - !Nonbonded
      default:
        - !LennardJones {{mixing: LB}}
      spline:
        cutoff: 5.0
        bounding_spheres: true
        cell_list: false # a cell list over this sparse box would dominate the peak
  blocks:
    - {{molecule: rod, N: 3000, insert: !RandomCOM {{rotate: true}}}}
analysis: []
propagate:
  seed: !Fixed 1
  criterion: Metropolis
  steps: 0
  collections: []
",
        beads.join(", ")
    );
    let baseline = CURRENT.load(Relaxed);
    PEAK.store(baseline, Relaxed);
    let simulation = Simulation::from_yaml(&yaml, None).expect("valid input");
    let peak_mb = (PEAK.load(Relaxed) - baseline) as f64 / 1e6;
    drop(simulation);
    assert!(
        peak_mb < 50.0,
        "building 30 000 particles peaked at {peak_mb:.0} MB"
    );
}
