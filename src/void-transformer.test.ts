import { describe, test, expect } from 'bun:test';
import {
  VoidAttentionHead,
  VoidCrossAttentionHead,
  VoidTransformerBlock,
} from './void-transformer';
import { createVoidBoundary, updateVoidBoundary } from '../../gnosis/src/runtime/void-walker.js';

function seededRng(seed: number): () => number {
  let s = seed;
  return () => {
    s = (s * 1664525 + 1013904223) & 0x7fffffff;
    return s / 0x7fffffff;
  };
}

// ============================================================================
// Payoff matrices
// ============================================================================

function hawkDovePayoff(a: number, b: number): [number, number] {
  if (a === 0 && b === 0) return [-1, -1];
  if (a === 0 && b === 1) return [4, 0];
  if (a === 1 && b === 0) return [0, 4];
  return [2, 2];
}

function prisonerPayoff(a: number, b: number): [number, number] {
  if (a === 1 && b === 1) return [3, 3];
  if (a === 1 && b === 0) return [0, 5];
  if (a === 0 && b === 1) return [5, 0];
  return [1, 1];
}

function coordinationPayoff3(a: number, b: number): [number, number] {
  return a === b ? [3, 3] : [0, 0];
}

/** Chester v Maxell (simplified: 5 choices $140K-$180K) */
function chesterMaxellSmall(a: number, b: number): [number, number] {
  const amountA = 140_000 + a * 10_000;
  const amountB = 140_000 + b * 10_000;
  if (amountA <= amountB) {
    const mid = (amountA + amountB) / 2;
    return [(mid - 95_000) / 1000, (200_000 - mid) / 1000];
  }
  const gap = (amountA - amountB) / 1000;
  return [-gap * 0.4, -gap * 0.5];
}

// ============================================================================
// Self-Attention Head Tests
// ============================================================================

describe('VoidAttentionHead', () => {
  test('uniform void produces uniform attention', () => {
    const head = new VoidAttentionHead({
      numChoices: 4,
      eta: 2.0,
      neighborhoodRadius: 0,
      decayRate: 0,
    });
    const out = head.attend();
    expect(out.weights.length).toBe(4);
    expect(out.weights[0]).toBeCloseTo(0.25, 2);
    expect(out.weights[3]).toBeCloseTo(0.25, 2);
  });

  test('rejection shifts attention away from rejected choice', () => {
    const head = new VoidAttentionHead({
      numChoices: 3,
      eta: 3.0,
      neighborhoodRadius: 0,
      decayRate: 0,
    });
    for (let i = 0; i < 10; i++) head.reject(0);
    const out = head.attend();
    // Choice 0 has been rejected 10 times -- should have lowest weight
    expect(out.weights[0]).toBeLessThan(out.weights[1]);
    expect(out.weights[0]).toBeLessThan(out.weights[2]);
  });

  test('neighborhood spread poisons adjacent choices', () => {
    const head = new VoidAttentionHead({
      numChoices: 5,
      eta: 2.0,
      neighborhoodRadius: 1,
      decayRate: 0,
    });
    head.reject(2, 5);  // Reject choice 2 with magnitude 5
    // Choice 2 should have highest void, neighbors should also have some
    expect(head.boundary.counts[2]).toBeGreaterThanOrEqual(head.boundary.counts[1]);
    expect(head.boundary.counts[1]).toBeGreaterThan(head.boundary.counts[0]);
    expect(head.boundary.counts[3]).toBeGreaterThan(head.boundary.counts[4]);
  });

  test('layer norm (decay) reduces void density', () => {
    const head = new VoidAttentionHead({
      numChoices: 3,
      eta: 2.0,
      neighborhoodRadius: 0,
      decayRate: 0.5,
    });
    for (let i = 0; i < 10; i++) head.reject(0);
    const before = head.boundary.counts[0];
    head.layerNorm();
    expect(head.boundary.counts[0]).toBeLessThan(before);
  });

  test('entropy decreases as void concentrates', () => {
    const head = new VoidAttentionHead({
      numChoices: 4,
      eta: 3.0,
      neighborhoodRadius: 0,
      decayRate: 0,
    });
    const before = head.attend().entropy;
    // Reject choices 0, 1, 2 heavily -- attention should concentrate on 3
    for (let i = 0; i < 20; i++) {
      head.reject(0, 3);
      head.reject(1, 3);
      head.reject(2, 3);
    }
    const after = head.attend().entropy;
    expect(after).toBeLessThan(before);
  });
});

// ============================================================================
// Cross-Attention Head Tests
// ============================================================================

describe('VoidCrossAttentionHead', () => {
  test('uniform voids produce uniform cross-attention', () => {
    const head = new VoidCrossAttentionHead({
      numChoicesA: 2,
      numChoicesB: 2,
      eta: 2.0,
      neighborhoodRadius: 0,
      decayRate: 0,
    });
    const bA = createVoidBoundary(2);
    const bB = createVoidBoundary(2);
    const out = head.crossAttend(bA, bB, 2.0, 2.0);
    expect(out.weights.length).toBe(4);
    expect(out.weights[0]).toBeCloseTo(0.25, 1);
  });

  test('asymmetric voids shift cross-attention', () => {
    const head = new VoidCrossAttentionHead({
      numChoicesA: 2,
      numChoicesB: 2,
      eta: 3.0,
      neighborhoodRadius: 0,
      decayRate: 0,
    });
    const bA = createVoidBoundary(2);
    const bB = createVoidBoundary(2);
    // A rejects 0, B rejects 1 → cross should favor [1, 0]
    for (let i = 0; i < 10; i++) { updateVoidBoundary(bA, 0); updateVoidBoundary(bB, 1); }
    const out = head.crossAttend(bA, bB, 3.0, 3.0);
    const weight_1_0 = out.weights[1 * 2 + 0]; // [1, 0]
    const weight_0_1 = out.weights[0 * 2 + 1]; // [0, 1]
    expect(weight_1_0).toBeGreaterThan(weight_0_1);
  });

  test('proposal rejection updates Skyrms void', () => {
    const head = new VoidCrossAttentionHead({
      numChoicesA: 3,
      numChoicesB: 3,
      eta: 2.0,
      neighborhoodRadius: 1,
      decayRate: 0,
    });
    head.rejectProposal(1, 1, 5);
    // Center [1,1] should have high void, neighbors should also be elevated
    const center = head.proposalVoid.counts[1 * 3 + 1];
    const neighbor = head.proposalVoid.counts[0 * 3 + 1];
    expect(center).toBeGreaterThanOrEqual(neighbor);
    expect(neighbor).toBeGreaterThan(0);
  });

  test('gated attention: Skyrms void gates the joint surface', () => {
    const head = new VoidCrossAttentionHead({
      numChoicesA: 2,
      numChoicesB: 2,
      eta: 3.0,
      neighborhoodRadius: 0,
      decayRate: 0,
    });
    const bA = createVoidBoundary(2);
    const bB = createVoidBoundary(2);
    // Heavily reject proposal [0,0] in the Skyrms void
    for (let i = 0; i < 20; i++) head.rejectProposal(0, 0, 3);
    const out = head.crossAttend(bA, bB, 2.0, 2.0);
    // [0,0] should have much lower weight than others
    expect(out.weights[0]).toBeLessThan(out.weights[1]);
    expect(out.weights[0]).toBeLessThan(out.weights[2]);
    expect(out.weights[0]).toBeLessThan(out.weights[3]);
  });
});

// ============================================================================
// Void Transformer Block Tests
// ============================================================================

describe('VoidTransformerBlock', () => {
  test('runs one round without error', () => {
    const block = new VoidTransformerBlock({
      numChoicesA: 2,
      numChoicesB: 2,
    });
    const rng = seededRng(42);
    const out = block.forward(hawkDovePayoff, rng);
    expect(out.offerA).toBeGreaterThanOrEqual(0);
    expect(out.offerA).toBeLessThan(2);
    expect(out.selfA.length).toBe(1);
    expect(out.cross.weights.length).toBe(4);
  });

  test('runs 100 rounds on Hawk-Dove', () => {
    const block = new VoidTransformerBlock({
      numChoicesA: 2,
      numChoicesB: 2,
      neighborhoodRadius: 0,
    });
    const rng = seededRng(42);
    const results = [];
    for (let i = 0; i < 100; i++) {
      results.push(block.forward(hawkDovePayoff, rng));
    }
    expect(results.length).toBe(100);
    // Gait should have progressed from stand
    const lastGait = results[results.length - 1].gaitS;
    expect(['stand', 'trot', 'canter', 'gallop']).toContain(lastGait);
  });

  test('PD converges: cooperation emerges', () => {
    const block = new VoidTransformerBlock({
      numChoicesA: 2,
      numChoicesB: 2,
      neighborhoodRadius: 1,
    });
    const rng = seededRng(123);
    let cooperationCount = 0;
    for (let i = 0; i < 200; i++) {
      const out = block.forward(prisonerPayoff, rng);
      if (out.offerA === 1 && out.offerB === 1) cooperationCount++;
    }
    // Some cooperation should emerge (void walking finds it)
    expect(cooperationCount).toBeGreaterThan(0);
  });

  test('multi-head: 3 heads per walker', () => {
    const block = new VoidTransformerBlock({
      numChoicesA: 2,
      numChoicesB: 2,
      numHeads: 3,
    });
    const rng = seededRng(42);
    const out = block.forward(hawkDovePayoff, rng);
    expect(out.selfA.length).toBe(3);
    expect(out.selfB.length).toBe(3);
    // All heads should have valid outputs
    for (const h of out.selfA) {
      expect(h.weights.length).toBe(2);
      expect(h.entropy).toBeGreaterThanOrEqual(0);
    }
  });

  test('3x3 coordination game', () => {
    const block = new VoidTransformerBlock({
      numChoicesA: 3,
      numChoicesB: 3,
      neighborhoodRadius: 1,
    });
    const rng = seededRng(55);
    let coordCount = 0;
    for (let i = 0; i < 300; i++) {
      const out = block.forward(coordinationPayoff3, rng);
      if (out.offerA === out.offerB) coordCount++;
    }
    expect(coordCount).toBeGreaterThan(0);
  });

  test('with void decay (layer norm)', () => {
    const block = new VoidTransformerBlock({
      numChoicesA: 2,
      numChoicesB: 2,
      decayRate: 0.01,
    });
    const rng = seededRng(42);
    for (let i = 0; i < 50; i++) {
      block.forward(hawkDovePayoff, rng);
    }
    // Void should still have entries (decay is slow)
    const state = block.getState();
    expect(state.headsA[0].boundary.totalEntries).toBeGreaterThan(0);
  });

  test('getState returns valid inspection data', () => {
    const block = new VoidTransformerBlock({
      numChoicesA: 2,
      numChoicesB: 2,
      numHeads: 2,
    });
    const rng = seededRng(42);
    block.forward(hawkDovePayoff, rng);
    const state = block.getState();
    expect(state.headsA.length).toBe(2);
    expect(state.headsB.length).toBe(2);
    expect(state.roundCount).toBe(1);
    expect(['stand', 'trot', 'canter', 'gallop']).toContain(state.gaits.a);
  });
});

// ============================================================================
// Benchmark: Void Transformer vs Three-Walker
// ============================================================================

describe('Benchmark: Void Transformer', () => {
  const games: [string, (a: number, b: number) => [number, number], number][] = [
    ['Hawk-Dove (2x2)', hawkDovePayoff, 2],
    ['PD (2x2)', prisonerPayoff, 2],
    ['Coordination (3x3)', coordinationPayoff3, 3],
    ['Chester-Maxell (5x5)', chesterMaxellSmall, 5],
  ];

  for (const [name, payoff, n] of games) {
    test(`${name}: 500 rounds, 5 seeds`, () => {
      const seeds = [42, 123, 456, 789, 1337];
      const results: { coordRate: number; avgEntropy: number; finalGait: string }[] = [];

      for (const seed of seeds) {
        const block = new VoidTransformerBlock({
          numChoicesA: n,
          numChoicesB: n,
          neighborhoodRadius: Math.min(2, n - 1),
          decayRate: 0.005,
          ffnCadence: 5,
        });
        const rng = seededRng(seed);
        let coord = 0;
        let totalEntropy = 0;
        let lastGait = 'stand';

        for (let i = 0; i < 500; i++) {
          const out = block.forward(payoff, rng);
          if (out.offerA === out.offerB) coord++;
          totalEntropy += out.cross.entropy;
          lastGait = out.gaitS;
        }

        results.push({
          coordRate: coord / 500,
          avgEntropy: totalEntropy / 500,
          finalGait: lastGait,
        });
      }

      const avgCoord = results.reduce((s, r) => s + r.coordRate, 0) / results.length;
      const avgEntropy = results.reduce((s, r) => s + r.avgEntropy, 0) / results.length;
      const gaits = results.map((r) => r.finalGait);

      console.log(`\n=== Void Transformer: ${name} ===`);
      console.log(`  Coordination rate: ${(avgCoord * 100).toFixed(1)}%`);
      console.log(`  Avg cross-attention entropy: ${avgEntropy.toFixed(3)}`);
      console.log(`  Final gaits: ${gaits.join(', ')}`);

      expect(results.length).toBe(seeds.length);
    });
  }
});
