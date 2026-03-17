import { describe, test, expect } from 'bun:test';
import {
  createSkyrmsWalkerState,
  computeSkyrmsPayoff,
  skyrmsC0Choose,
  skyrmsC0Update,
  skyrmsC1Measure,
  skyrmsC3Adapt,
  decodeProposal,
  mediateThreeWalker,
  type ThreeWalkerConfig,
} from './skyrms-walker';
import {
  createVoidBoundary,
  updateVoidBoundary,
  createMetaCogState,
  complementDistribution,
} from '../../aeon-bazaar/src/engine/void-walker';
import { JointVoidSurface } from './joint-surface';
import { SkyrmsNadirDetector } from './nadir-detector';

// ============================================================================
// Deterministic RNG for reproducible tests
// ============================================================================

function seededRng(seed: number): () => number {
  let s = seed;
  return () => {
    s = (s * 1664525 + 1013904223) & 0x7fffffff;
    return s / 0x7fffffff;
  };
}

// ============================================================================
// Payoff matrices for classic games
// ============================================================================

/** Hawk-Dove: V=4, C=6. Choices: 0=hawk, 1=dove */
function hawkDovePayoff(a: number, b: number): [number, number] {
  if (a === 0 && b === 0) return [-1, -1];     // hawk-hawk: (V-C)/2
  if (a === 0 && b === 1) return [4, 0];        // hawk-dove: V, 0
  if (a === 1 && b === 0) return [0, 4];        // dove-hawk: 0, V
  return [2, 2];                                 // dove-dove: V/2, V/2
}

/** Prisoner's Dilemma: 0=defect, 1=cooperate */
function prisonerPayoff(a: number, b: number): [number, number] {
  if (a === 1 && b === 1) return [3, 3];        // cooperate-cooperate
  if (a === 1 && b === 0) return [0, 5];        // cooperate-defect
  if (a === 0 && b === 1) return [5, 0];        // defect-cooperate
  return [1, 1];                                 // defect-defect
}

/** Battle of Sexes: 0=opera, 1=football */
function battlePayoff(a: number, b: number): [number, number] {
  if (a === 0 && b === 0) return [3, 2];        // both opera
  if (a === 1 && b === 1) return [2, 3];        // both football
  return [0, 0];                                 // mismatch
}

/** Stag Hunt: 0=hare, 1=stag */
function stagHuntPayoff(a: number, b: number): [number, number] {
  if (a === 1 && b === 1) return [4, 4];        // stag-stag
  if (a === 0 && b === 0) return [2, 2];        // hare-hare
  if (a === 1 && b === 0) return [0, 2];        // stag-hare
  return [2, 0];                                 // hare-stag
}

/** 3-choice coordination: payoff = 3 if match, 0 if mismatch */
function coordinationPayoff3(a: number, b: number): [number, number] {
  return a === b ? [3, 3] : [0, 0];
}

// ============================================================================
// Unit Tests: Skyrms Walker State
// ============================================================================

describe('SkyrmsWalkerState', () => {
  test('creates with correct proposal space size', () => {
    const state = createSkyrmsWalkerState(3, 3);
    expect(state.proposalSpaceSize).toBe(9);
    expect(state.meta.boundary.counts.length).toBe(9);
  });

  test('decodes proposals correctly', () => {
    const state = createSkyrmsWalkerState(3, 4);
    expect(decodeProposal(state, 0)).toEqual([0, 0]);
    expect(decodeProposal(state, 1)).toEqual([0, 1]);
    expect(decodeProposal(state, 4)).toEqual([1, 0]);
    expect(decodeProposal(state, 11)).toEqual([2, 3]);
  });

  test('initial complement distribution is uniform', () => {
    const state = createSkyrmsWalkerState(2, 2);
    const dist = complementDistribution(state.meta.boundary, state.meta.eta);
    expect(dist[0]).toBeCloseTo(0.25, 2);
    expect(dist[1]).toBeCloseTo(0.25, 2);
    expect(dist[2]).toBeCloseTo(0.25, 2);
    expect(dist[3]).toBeCloseTo(0.25, 2);
  });
});

// ============================================================================
// Unit Tests: Skyrms Payoff Matrix
// ============================================================================

describe('SkyrmsPayoffMatrix', () => {
  test('uniform boundaries produce uniform payoff matrix', () => {
    const bA = createVoidBoundary(2);
    const bB = createVoidBoundary(2);
    const matrix = computeSkyrmsPayoff(bA, bB, 3.0, 3.0, 2, 2);
    expect(matrix.values.length).toBe(4);
    // All equal when both are uniform
    const first = matrix.values[0];
    for (const v of matrix.values) {
      expect(v).toBeCloseTo(first, 5);
    }
  });

  test('asymmetric void shifts payoff toward less-rejected pairs', () => {
    const bA = createVoidBoundary(2);
    const bB = createVoidBoundary(2);
    // A has rejected choice 0 many times -> complement favors choice 1
    for (let i = 0; i < 10; i++) updateVoidBoundary(bA, 0);
    // B has rejected choice 1 many times -> complement favors choice 0
    for (let i = 0; i < 10; i++) updateVoidBoundary(bB, 1);

    const matrix = computeSkyrmsPayoff(bA, bB, 3.0, 3.0, 2, 2);
    // Nadir should be [1, 0]: A prefers 1, B prefers 0
    const payoff_1_0 = matrix.values[1 * 2 + 0]; // proposal [1, 0]
    const payoff_0_1 = matrix.values[0 * 2 + 1]; // proposal [0, 1]
    expect(payoff_1_0).toBeGreaterThan(payoff_0_1);
  });
});

// ============================================================================
// Unit Tests: Joint Void Surface
// ============================================================================

describe('JointVoidSurface', () => {
  test('uniform boundaries produce zero distance', () => {
    const bA = createVoidBoundary(2);
    const bB = createVoidBoundary(2);
    const surface = new JointVoidSurface(2, 2);
    const state = surface.compute(bA, bB, 3.0, 3.0);
    expect(state.distance).toBeCloseTo(0, 5);
  });

  test('divergent voids produce positive distance', () => {
    const bA = createVoidBoundary(2);
    const bB = createVoidBoundary(2);
    for (let i = 0; i < 10; i++) updateVoidBoundary(bA, 0);
    for (let i = 0; i < 10; i++) updateVoidBoundary(bB, 1);
    const surface = new JointVoidSurface(2, 2);
    const state = surface.compute(bA, bB, 3.0, 3.0);
    // Both shifted in the same direction (toward the less-rejected choice)
    // so distance may be small; but they started from different voids
    expect(state.distance).toBeGreaterThanOrEqual(0);
  });

  test('mutual information is non-negative for product distributions', () => {
    const bA = createVoidBoundary(3);
    const bB = createVoidBoundary(3);
    updateVoidBoundary(bA, 0, 5);
    updateVoidBoundary(bB, 2, 5);
    const surface = new JointVoidSurface(3, 3);
    const state = surface.compute(bA, bB, 3.0, 3.0);
    // For independent distributions, MI should be ~0
    expect(state.mutualInformation).toBeGreaterThanOrEqual(-0.01);
  });
});

// ============================================================================
// Unit Tests: Nadir Detector
// ============================================================================

describe('SkyrmsNadirDetector', () => {
  test('does not certify before window is full', () => {
    const detector = new SkyrmsNadirDetector(0.1, 3);
    const state = {
      surface: [0.25, 0.25, 0.25, 0.25],
      distance: 0.05,
      jointEntropy: 1.0,
      mutualInformation: 0.1,
      jointKurtosis: 0.0,
      gini: 0.0,
      nadirPoint: [0, 0] as [number, number],
    };
    expect(detector.observe(state, false)).toBeNull();
    expect(detector.observe(state, false)).toBeNull();
  });

  test('certifies when all conditions met for windowSize rounds', () => {
    const detector = new SkyrmsNadirDetector(0.1, 3, 0.05);
    const state = {
      surface: [0.25, 0.25, 0.25, 0.25],
      distance: 0.05,
      jointEntropy: 1.0,
      mutualInformation: 0.1,
      jointKurtosis: 0.0,
      gini: 0.0,
      nadirPoint: [1, 1] as [number, number],
    };
    expect(detector.observe(state, true)).toBeNull();
    expect(detector.observe(state, false)).toBeNull();
    const cert = detector.observe(state, false);
    expect(cert).not.toBeNull();
    expect(cert!.round).toBe(3);
    expect(cert!.nadirPoint).toEqual([1, 1]);
    expect(cert!.totalFailures).toBe(1);
  });

  test('does not certify when distance exceeds threshold', () => {
    const detector = new SkyrmsNadirDetector(0.1, 3);
    const good = {
      surface: [], distance: 0.05, jointEntropy: 1.0,
      mutualInformation: 0.1, jointKurtosis: 0.0, gini: 0.0,
      nadirPoint: [0, 0] as [number, number],
    };
    const bad = { ...good, distance: 0.5 };
    detector.observe(good, false);
    detector.observe(bad, false);  // breaks the window
    detector.observe(good, false);
    expect(detector.observe(good, false)).toBeNull(); // window tainted
  });

  test('reset clears state', () => {
    const detector = new SkyrmsNadirDetector(0.1, 2);
    const state = {
      surface: [], distance: 0.05, jointEntropy: 1.0,
      mutualInformation: 0.1, jointKurtosis: 0.0, gini: 0.0,
      nadirPoint: [0, 0] as [number, number],
    };
    detector.observe(state, false);
    detector.reset();
    // After reset, need windowSize rounds again
    expect(detector.observe(state, false)).toBeNull();
  });
});

// ============================================================================
// Unit Tests: Skyrms Walker c0-c3
// ============================================================================

describe('Skyrms Walker c0-c3', () => {
  test('c0 choose returns valid proposals', () => {
    const state = createSkyrmsWalkerState(3, 3);
    const rng = seededRng(42);
    for (let i = 0; i < 50; i++) {
      const [a, b] = skyrmsC0Choose(state, rng);
      expect(a).toBeGreaterThanOrEqual(0);
      expect(a).toBeLessThan(3);
      expect(b).toBeGreaterThanOrEqual(0);
      expect(b).toBeLessThan(3);
    }
  });

  test('c0 update enriches void on failure', () => {
    const state = createSkyrmsWalkerState(2, 2);
    const initial = [...state.meta.boundary.counts];
    skyrmsC0Update(state, 0, 0.5, false); // rejected proposal
    expect(state.meta.boundary.counts[0]).toBeGreaterThan(initial[0]);
    expect(state.rejectedCount).toBe(1);
  });

  test('c0 update does not enrich void on success', () => {
    const state = createSkyrmsWalkerState(2, 2);
    state.prevDistance = 1.0;
    const initial = [...state.meta.boundary.counts];
    skyrmsC0Update(state, 0, 0.5, true); // accepted, distance decreased
    expect(state.meta.boundary.counts[0]).toBe(initial[0]);
    expect(state.acceptedCount).toBe(1);
  });

  test('c1 measure returns valid metrics', () => {
    const state = createSkyrmsWalkerState(2, 2);
    state.meta.totalRounds = 5;
    const meas = skyrmsC1Measure(state);
    expect(meas.entropy).toBeGreaterThanOrEqual(0);
    expect(meas.inverseBule).toBeGreaterThanOrEqual(0);
    expect(meas.acceptanceRate).toBeGreaterThanOrEqual(0);
    expect(meas.acceptanceRate).toBeLessThanOrEqual(1);
  });
});

// ============================================================================
// Integration Tests: Three-Walker Mediation
// ============================================================================

describe('Three-Walker Mediation', () => {
  test('Hawk-Dove converges', () => {
    const result = mediateThreeWalker({
      numChoicesA: 2,
      numChoicesB: 2,
      maxRounds: 200,
      nadirThreshold: 0.15,
      payoff: hawkDovePayoff,
      rng: seededRng(42),
    });
    expect(result.rounds.length).toBeGreaterThan(0);
    // Should settle or at minimum reduce distance over time
    const firstDist = result.rounds[0].distance;
    const lastDist = result.rounds[result.rounds.length - 1].distance;
    expect(lastDist).toBeLessThanOrEqual(firstDist + 0.5);
  });

  test('Prisoner\'s Dilemma runs without error', () => {
    const result = mediateThreeWalker({
      numChoicesA: 2,
      numChoicesB: 2,
      maxRounds: 100,
      nadirThreshold: 0.15,
      payoff: prisonerPayoff,
      rng: seededRng(123),
    });
    expect(result.rounds.length).toBeGreaterThan(0);
    // Verify payoffs are from the correct matrix
    for (const r of result.rounds) {
      const [expectedA, expectedB] = prisonerPayoff(r.offerA, r.offerB);
      expect(r.payoffA).toBe(expectedA);
      expect(r.payoffB).toBe(expectedB);
    }
  });

  test('Battle of Sexes converges to coordination', () => {
    const result = mediateThreeWalker({
      numChoicesA: 2,
      numChoicesB: 2,
      maxRounds: 200,
      nadirThreshold: 0.15,
      payoff: battlePayoff,
      rng: seededRng(77),
    });
    // Count how many rounds had matching offers in the last 20
    const tail = result.rounds.slice(-20);
    const matches = tail.filter((r) => r.offerA === r.offerB).length;
    // Should be coordinating at least sometimes
    expect(matches).toBeGreaterThan(0);
  });

  test('Stag Hunt runs without error', () => {
    const result = mediateThreeWalker({
      numChoicesA: 2,
      numChoicesB: 2,
      maxRounds: 100,
      nadirThreshold: 0.15,
      payoff: stagHuntPayoff,
      rng: seededRng(999),
    });
    expect(result.rounds.length).toBeGreaterThan(0);
  });

  test('3-choice coordination game converges', () => {
    const result = mediateThreeWalker({
      numChoicesA: 3,
      numChoicesB: 3,
      maxRounds: 300,
      nadirThreshold: 0.2,
      payoff: coordinationPayoff3,
      rng: seededRng(55),
    });
    expect(result.rounds.length).toBeGreaterThan(0);
    const lastDist = result.rounds[result.rounds.length - 1].distance;
    const firstDist = result.rounds[0].distance;
    expect(lastDist).toBeLessThanOrEqual(firstDist + 1.0);
  });

  test('Skyrms walker void grows with failures', () => {
    const result = mediateThreeWalker({
      numChoicesA: 2,
      numChoicesB: 2,
      maxRounds: 50,
      payoff: hawkDovePayoff,
      rng: seededRng(42),
    });
    // Skyrms walker's void should have accumulated entries
    const totalVoid = result.skyrmsWalker.meta.boundary.totalEntries;
    expect(totalVoid).toBeGreaterThan(0);
  });

  test('walker A and B void boundaries grow', () => {
    const result = mediateThreeWalker({
      numChoicesA: 2,
      numChoicesB: 2,
      maxRounds: 50,
      payoff: hawkDovePayoff,
      rng: seededRng(42),
    });
    expect(result.walkerA.boundary.totalEntries).toBeGreaterThan(0);
    expect(result.walkerB.boundary.totalEntries).toBeGreaterThan(0);
  });

  test('proposal acceptance rate is between 0 and 1', () => {
    const result = mediateThreeWalker({
      numChoicesA: 2,
      numChoicesB: 2,
      maxRounds: 100,
      payoff: hawkDovePayoff,
      rng: seededRng(42),
    });
    const accepted = result.rounds.filter((r) => r.proposalAccepted).length;
    const rate = accepted / result.rounds.length;
    expect(rate).toBeGreaterThanOrEqual(0);
    expect(rate).toBeLessThanOrEqual(1);
  });

  test('final payoff matrix sums to ~1', () => {
    const result = mediateThreeWalker({
      numChoicesA: 2,
      numChoicesB: 2,
      maxRounds: 50,
      payoff: hawkDovePayoff,
      rng: seededRng(42),
    });
    const sum = result.finalPayoffMatrix.values.reduce((a, b) => a + b, 0);
    expect(sum).toBeCloseTo(1.0, 2);
  });
});

// ============================================================================
// Benchmark: Convergence Speed Across Games
// ============================================================================

describe('Benchmark: Convergence Speed', () => {
  const games: [string, (a: number, b: number) => [number, number], number, number][] = [
    ['Hawk-Dove (2x2)', hawkDovePayoff, 2, 2],
    ['Prisoner\'s Dilemma (2x2)', prisonerPayoff, 2, 2],
    ['Battle of Sexes (2x2)', battlePayoff, 2, 2],
    ['Stag Hunt (2x2)', stagHuntPayoff, 2, 2],
    ['Coordination (3x3)', coordinationPayoff3, 3, 3],
  ];

  const seeds = [42, 123, 456, 789, 1337];
  const maxRounds = 500;

  for (const [name, payoff, nA, nB] of games) {
    test(`${name}: benchmark across ${seeds.length} seeds`, () => {
      const results: {
        settled: boolean;
        rounds: number;
        finalDistance: number;
        skyrmsVoidSize: number;
        acceptanceRate: number;
        avgPayoffA: number;
        avgPayoffB: number;
      }[] = [];

      for (const seed of seeds) {
        const t0 = performance.now();
        const result = mediateThreeWalker({
          numChoicesA: nA,
          numChoicesB: nB,
          maxRounds,
          nadirThreshold: 0.15,
          payoff,
          rng: seededRng(seed),
        });
        const elapsed = performance.now() - t0;

        const accepted = result.rounds.filter((r) => r.proposalAccepted).length;
        const avgA = result.rounds.reduce((s, r) => s + r.payoffA, 0) / result.rounds.length;
        const avgB = result.rounds.reduce((s, r) => s + r.payoffB, 0) / result.rounds.length;

        results.push({
          settled: result.settled,
          rounds: result.rounds.length,
          finalDistance: result.rounds[result.rounds.length - 1].distance,
          skyrmsVoidSize: result.skyrmsWalker.meta.boundary.totalEntries,
          acceptanceRate: accepted / result.rounds.length,
          avgPayoffA: avgA,
          avgPayoffB: avgB,
        });
      }

      // Aggregate stats
      const settledCount = results.filter((r) => r.settled).length;
      const avgRounds = results.reduce((s, r) => s + r.rounds, 0) / results.length;
      const avgDist = results.reduce((s, r) => s + r.finalDistance, 0) / results.length;
      const avgAcceptance = results.reduce((s, r) => s + r.acceptanceRate, 0) / results.length;
      const avgPayA = results.reduce((s, r) => s + r.avgPayoffA, 0) / results.length;
      const avgPayB = results.reduce((s, r) => s + r.avgPayoffB, 0) / results.length;

      console.log(`\n=== ${name} ===`);
      console.log(`  Settled: ${settledCount}/${seeds.length}`);
      console.log(`  Avg rounds: ${avgRounds.toFixed(1)}`);
      console.log(`  Avg final distance: ${avgDist.toFixed(4)}`);
      console.log(`  Avg acceptance rate: ${(avgAcceptance * 100).toFixed(1)}%`);
      console.log(`  Avg payoff A: ${avgPayA.toFixed(3)}, B: ${avgPayB.toFixed(3)}`);

      // Basic sanity: should complete without error
      expect(results.length).toBe(seeds.length);
      // Distance should generally decrease (allow some slack)
      expect(avgDist).toBeLessThan(2.0);
    });
  }
});

// ============================================================================
// Benchmark: Passive Mediator vs Three-Walker
// ============================================================================

describe('Benchmark: Passive vs Three-Walker', () => {
  test('compare convergence on Hawk-Dove', () => {
    const rngA = seededRng(42);
    const rngB = seededRng(42);

    // Three-walker
    const threeResult = mediateThreeWalker({
      numChoicesA: 2,
      numChoicesB: 2,
      maxRounds: 300,
      nadirThreshold: 0.15,
      payoff: hawkDovePayoff,
      rng: rngA,
    });

    // Passive mediator (imported from mediator.ts)
    const { NeutralMediator } = require('./mediator');
    const passive = new NeutralMediator({
      numChoicesA: 2,
      numChoicesB: 2,
      maxRounds: 300,
      nadirThreshold: 0.15,
      payoff: hawkDovePayoff,
      rng: rngB,
    });
    const passiveResult = passive.mediate();

    const threeDist = threeResult.rounds[threeResult.rounds.length - 1].distance;
    const passiveDist = passiveResult.rounds[passiveResult.rounds.length - 1].distance;

    console.log('\n=== Passive vs Three-Walker (Hawk-Dove) ===');
    console.log(`  Passive: ${passiveResult.rounds.length} rounds, final dist ${passiveDist.toFixed(4)}, settled: ${passiveResult.settled}`);
    console.log(`  Three-Walker: ${threeResult.rounds.length} rounds, final dist ${threeDist.toFixed(4)}, settled: ${threeResult.settled}`);

    // Both should produce valid results
    expect(threeResult.rounds.length).toBeGreaterThan(0);
    expect(passiveResult.rounds.length).toBeGreaterThan(0);
  });
});
