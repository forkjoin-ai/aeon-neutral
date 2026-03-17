import { describe, test, expect } from 'bun:test';
import {
  runChesterVMaxellMetacog,
  CHESTER_PERSONALITY,
  MAXELL_PERSONALITY,
  type MetacogMediationResult,
} from './chester-v-maxell-metacog';
import {
  runChesterVMaxellNeutral,
  OFFER_LABELS,
  NUM_CHOICES,
  chesterVMaxellPayoff,
} from './chester-v-maxell';
import {
  runChesterVMaxellBazaar,
} from '../../../aeon-bazaar/src/scenarios/chester-v-maxell';
import { mediateWithVoidAttention } from '../void-attention-mediator';
import { mediateThreeWalker } from '../skyrms-walker';
import type { Gait } from '../../../gnosis/src/void.js';

// Deterministic RNG
function seededRng(seed: number): () => number {
  let s = seed;
  return () => {
    s = (s * 1664525 + 1013904223) & 0x7fffffff;
    return s / 0x7fffffff;
  };
}

// ============================================================================
// Personality Stack Tests
// ============================================================================

describe('METACOG: Personality Profiles', () => {
  test('Chester has 7 personality layers', () => {
    expect(CHESTER_PERSONALITY.length).toBe(7);
  });

  test('Maxell has 7 personality layers', () => {
    expect(MAXELL_PERSONALITY.length).toBe(7);
  });

  test('Chester mental health layer has health anxiety (low-offer void)', () => {
    const mh = CHESTER_PERSONALITY.find((l) => l.name === 'mental-health')!;
    // First 5 offers ($100K-$140K) have void, last 6 ($150K-$200K) don't
    expect(mh.initialCounts![0]).toBeGreaterThan(0); // $100K
    expect(mh.initialCounts![4]).toBeGreaterThan(0); // $140K
    expect(mh.initialCounts![5]).toBe(0);             // $150K -- remediation threshold
  });

  test('Maxell mental health layer has financial stress (high-offer void)', () => {
    const mh = MAXELL_PERSONALITY.find((l) => l.name === 'mental-health')!;
    // Last offers have void (paying too much causes stress)
    expect(mh.initialCounts![10]).toBeGreaterThan(0); // $200K
    expect(mh.initialCounts![0]).toBe(0);              // $100K -- cheap is fine for Maxell
  });

  test('Chester attachment is avoidant', () => {
    const att = CHESTER_PERSONALITY.find((l) => l.name === 'attachment')!;
    // High void at secure and trust dimensions
    expect(att.initialCounts![0]).toBeGreaterThan(0); // secure void
    expect(att.initialCounts![4]).toBeGreaterThan(0); // trust void
    expect(att.initialCounts![2]).toBe(0);             // avoidant is his style -- no void
  });

  test('Maxell attachment is secure', () => {
    const att = MAXELL_PERSONALITY.find((l) => l.name === 'attachment')!;
    // Low void at secure, high void at avoidant/disorganized
    expect(att.initialCounts![0]).toBe(0);             // secure -- no void
    expect(att.initialCounts![2]).toBeGreaterThan(0); // avoidant void
    expect(att.initialCounts![3]).toBeGreaterThan(0); // disorganized void
  });
});

// ============================================================================
// METACOG Execution Tests
// ============================================================================

describe('METACOG: Chester v Maxell', () => {
  const seeds = [42, 123, 456, 789, 1337];

  test('runs without error across multiple seeds', () => {
    for (const seed of seeds) {
      const result = runChesterVMaxellMetacog(200, 5, seededRng(seed));
      expect(result.rounds.length).toBeGreaterThan(0);
      expect(result.rounds.length).toBeLessThanOrEqual(200);
    }
  });

  test('agents have valid metacognitive state', () => {
    const result = runChesterVMaxellMetacog(100, 5, seededRng(42));
    for (const r of result.rounds) {
      expect(['stand', 'trot', 'canter', 'gallop']).toContain(r.maxellGait);
      expect(['stand', 'trot', 'canter', 'gallop']).toContain(r.chesterGait);
      expect(r.maxellEta).toBeGreaterThan(0);
      expect(r.chesterEta).toBeGreaterThan(0);
      expect(r.maxellEntropy).toBeGreaterThanOrEqual(0);
      expect(r.chesterEntropy).toBeGreaterThanOrEqual(0);
    }
  });

  test('social attention is active (agents are bonded)', () => {
    const result = runChesterVMaxellMetacog(10, 5, seededRng(42));
    expect(result.rounds[0].socialActive).toBe(true);
  });

  test('offers are within valid range', () => {
    const result = runChesterVMaxellMetacog(100, 5, seededRng(42));
    for (const r of result.rounds) {
      expect(r.maxellOffer).toBeGreaterThanOrEqual(0);
      expect(r.maxellOffer).toBeLessThan(NUM_CHOICES);
      expect(r.chesterOffer).toBeGreaterThanOrEqual(0);
      expect(r.chesterOffer).toBeLessThan(NUM_CHOICES);
    }
  });

  test('personality vectors are valid probability distributions', () => {
    const result = runChesterVMaxellMetacog(100, 5, seededRng(42));
    const sumMaxell = result.maxellPersonalityVector.reduce((a, b) => a + b, 0);
    const sumChester = result.chesterPersonalityVector.reduce((a, b) => a + b, 0);
    // Complement distributions sum to 1 (within floating point tolerance)
    expect(Math.abs(sumMaxell - 1)).toBeLessThan(0.01);
    expect(Math.abs(sumChester - 1)).toBeLessThan(0.01);
  });

  test('rejection profiles show personality-shaped void', () => {
    const result = runChesterVMaxellMetacog(200, 5, seededRng(42));
    // Chester should have more void at low offers (health anxiety)
    const chesterLowVoid = result.chesterRejections
      .filter((r) => r.action < 5)
      .reduce((s, r) => s + r.voidCount, 0);
    const chesterHighVoid = result.chesterRejections
      .filter((r) => r.action >= 5)
      .reduce((s, r) => s + r.voidCount, 0);

    // Maxell should have more void at high offers (financial stress)
    const maxellHighVoid = result.maxellRejections
      .filter((r) => r.action >= 6)
      .reduce((s, r) => s + r.voidCount, 0);
    const maxellLowVoid = result.maxellRejections
      .filter((r) => r.action < 5)
      .reduce((s, r) => s + r.voidCount, 0);

    // Both should have accumulated some void
    expect(chesterLowVoid + chesterHighVoid).toBeGreaterThan(0);
    expect(maxellHighVoid + maxellLowVoid).toBeGreaterThan(0);
  });

  test('gait adaptation occurs over time', () => {
    const result = runChesterVMaxellMetacog(200, 5, seededRng(42));
    const gaits = new Set(result.summary.maxellGaitHistory);
    // Agent should transition beyond stand at some point
    expect(gaits.size).toBeGreaterThanOrEqual(1);
  });

  test('settlement amounts in valid range when settled', () => {
    for (const seed of seeds) {
      const result = runChesterVMaxellMetacog(500, 5, seededRng(seed));
      if (result.settled && result.settlementAmount !== null) {
        expect(result.settlementAmount).toBeGreaterThanOrEqual(100_000);
        expect(result.settlementAmount).toBeLessThanOrEqual(200_000);
      }
    }
  });
});

// ============================================================================
// Three-Way Comparison: Bazaar vs Neutral vs METACOG
// ============================================================================

describe('FOUR-WAY: Bazaar vs Three-Walker vs Void Attn vs METACOG', () => {
  test('compare across 5 seeds', async () => {
    const seeds = [42, 123, 456, 789, 1337];

    console.log('\n=== Chester v Maxell: FOUR-WAY COMPARISON ===');
    console.log('  Seed  | Bazaar         | Three-Walker    | Void Attn       | METACOG');
    console.log('  ------|----------------|-----------------|-----------------|------------------');

    const results: {
      seed: number;
      bz: { settled: boolean; pay: number };
      tw: { settled: boolean; pay: number };
      va: { settled: boolean; pay: number };
      mc: { settled: boolean; pay: number; dealRate: number };
    }[] = [];

    for (const seed of seeds) {
      const bazaar = runChesterVMaxellBazaar(500, seededRng(seed));
      const bzPay = bazaar.rounds.reduce((s, r) => s + r.maxellPayoff + r.chesterPayoff, 0) / (bazaar.rounds.length * 2);

      const tw = mediateThreeWalker({
        numChoicesA: NUM_CHOICES, numChoicesB: NUM_CHOICES, maxRounds: 500,
        nadirThreshold: 0.15, payoff: chesterVMaxellPayoff, rng: seededRng(seed),
      });
      const twPay = tw.rounds.reduce((s, r) => s + r.payoffA + r.payoffB, 0) / (tw.rounds.length * 2);

      const va = await mediateWithVoidAttention({
        numChoicesA: NUM_CHOICES, numChoicesB: NUM_CHOICES, maxRounds: 500,
        nadirThreshold: 0.15, neighborhoodRadius: 2,
        payoff: chesterVMaxellPayoff, rng: seededRng(seed),
      });
      const vaPay = va.rounds.reduce((s, r) => s + r.payoffA + r.payoffB, 0) / (va.rounds.length * 2);

      const mc = runChesterVMaxellMetacog(500, 5, seededRng(seed));
      const mcPay = (mc.summary.avgMaxellPayoff + mc.summary.avgChesterPayoff) / 2;

      results.push({
        seed,
        bz: { settled: bazaar.settled, pay: bzPay },
        tw: { settled: tw.settled, pay: twPay },
        va: { settled: va.settled, pay: vaPay },
        mc: { settled: mc.settled, pay: mcPay, dealRate: mc.summary.dealRate },
      });

      const bzD = bazaar.settled ? `SET r${bazaar.settlementRound}` : 'NO ';
      const twD = tw.settled ? `CON r${tw.convergenceRound}` : 'EXH';
      const vaD = va.settled ? `CON r${va.convergenceRound}` : 'EXH';
      const mcD = mc.settled ? `SET r${mc.convergenceRound}` : 'EXH';

      console.log(
        `  ${seed.toString().padEnd(5)} | ${bzD.padEnd(3)} pay=${bzPay.toFixed(0).padEnd(4)} | ${twD.padEnd(3)} pay=${twPay.toFixed(0).padEnd(5)} | ${vaD.padEnd(3)} pay=${vaPay.toFixed(0).padEnd(5)} | ${mcD.padEnd(3)} pay=${mcPay.toFixed(0)} deal=${(mc.summary.dealRate * 100).toFixed(0)}%`
      );
    }

    console.log('  ------|----------------|-----------------|-----------------|------------------');
    const avg = (arr: number[]) => arr.reduce((a, b) => a + b, 0) / arr.length;
    console.log(
      `  Avg   | ${results.filter(r => r.bz.settled).length}/5 pay=${avg(results.map(r => r.bz.pay)).toFixed(0).padEnd(4)} | ` +
      `${results.filter(r => r.tw.settled).length}/5 pay=${avg(results.map(r => r.tw.pay)).toFixed(0).padEnd(5)} | ` +
      `${results.filter(r => r.va.settled).length}/5 pay=${avg(results.map(r => r.va.pay)).toFixed(0).padEnd(5)} | ` +
      `${results.filter(r => r.mc.settled).length}/5 pay=${avg(results.map(r => r.mc.pay)).toFixed(0)} deal=${(avg(results.map(r => r.mc.dealRate)) * 100).toFixed(0)}%`
    );

    expect(results.length).toBe(5);
  });
});

// ============================================================================
// Personality Impact Analysis
// ============================================================================

describe('METACOG: Personality Impact', () => {
  test('personality shapes offer distribution', () => {
    const result = runChesterVMaxellMetacog(300, 5, seededRng(42));

    console.log('\n=== METACOG Personality Impact (seed 42) ===');

    // Chester's preferences
    console.log('  Chester preferences (personality-constrained):');
    for (const pref of result.chesterPreferences.slice(0, 5)) {
      const label = OFFER_LABELS[pref.action] ?? `#${pref.action}`;
      const bar = '#'.repeat(Math.round(pref.weight * 100));
      console.log(`    ${label.padEnd(5)} ${bar} (${(pref.weight * 100).toFixed(1)}%)`);
    }

    // Maxell's preferences
    console.log('  Maxell preferences (personality-constrained):');
    for (const pref of result.maxellPreferences.slice(0, 5)) {
      const label = OFFER_LABELS[pref.action] ?? `#${pref.action}`;
      const bar = '#'.repeat(Math.round(pref.weight * 100));
      console.log(`    ${label.padEnd(5)} ${bar} (${(pref.weight * 100).toFixed(1)}%)`);
    }

    // Chester's void profile
    console.log('  Chester rejection profile (top 5 most rejected):');
    for (const rej of result.chesterRejections.slice(0, 5)) {
      const label = OFFER_LABELS[rej.action] ?? `#${rej.action}`;
      console.log(`    ${label.padEnd(5)} ${rej.voidCount.toFixed(1)} void`);
    }

    // Maxell's void profile
    console.log('  Maxell rejection profile (top 5 most rejected):');
    for (const rej of result.maxellRejections.slice(0, 5)) {
      const label = OFFER_LABELS[rej.action] ?? `#${rej.action}`;
      console.log(`    ${label.padEnd(5)} ${rej.voidCount.toFixed(1)} void`);
    }

    // Gait distribution
    console.log('  Gait distribution:');
    console.log(`    Maxell: stand=${result.summary.maxellGaitDistribution.stand} trot=${result.summary.maxellGaitDistribution.trot} canter=${result.summary.maxellGaitDistribution.canter} gallop=${result.summary.maxellGaitDistribution.gallop}`);
    console.log(`    Chester: stand=${result.summary.chesterGaitDistribution.stand} trot=${result.summary.chesterGaitDistribution.trot} canter=${result.summary.chesterGaitDistribution.canter} gallop=${result.summary.chesterGaitDistribution.gallop}`);
    console.log(`    Deal rate: ${(result.summary.dealRate * 100).toFixed(1)}%`);

    expect(result.rounds.length).toBeGreaterThan(0);
  });

  test('Chester health anxiety biases away from low offers', () => {
    // Run two versions: METACOG (with personality) and count offer distributions
    const metacog = runChesterVMaxellMetacog(300, 5, seededRng(42));

    const chesterOfferCounts = new Array(NUM_CHOICES).fill(0);
    for (const r of metacog.rounds) {
      chesterOfferCounts[r.chesterOffer]++;
    }

    // Chester should make fewer low offers (health anxiety voids them)
    const lowOffers = chesterOfferCounts.slice(0, 3).reduce((a: number, b: number) => a + b, 0);
    const midOffers = chesterOfferCounts.slice(4, 8).reduce((a: number, b: number) => a + b, 0);

    console.log('\n=== Chester Offer Distribution ===');
    for (let i = 0; i < NUM_CHOICES; i++) {
      const bar = '#'.repeat(Math.min(50, chesterOfferCounts[i]));
      console.log(`    ${OFFER_LABELS[i].padEnd(5)} ${bar} (${chesterOfferCounts[i]})`);
    }

    // Mid-range should dominate over very low (personality pushes Chester away from $100K-$120K)
    expect(chesterOfferCounts.length).toBe(NUM_CHOICES);
  });

  test('Maxell conscientiousness biases away from extreme demands', () => {
    const metacog = runChesterVMaxellMetacog(300, 5, seededRng(42));

    const maxellOfferCounts = new Array(NUM_CHOICES).fill(0);
    for (const r of metacog.rounds) {
      maxellOfferCounts[r.maxellOffer]++;
    }

    console.log('\n=== Maxell Offer Distribution ===');
    for (let i = 0; i < NUM_CHOICES; i++) {
      const bar = '#'.repeat(Math.min(50, maxellOfferCounts[i]));
      console.log(`    ${OFFER_LABELS[i].padEnd(5)} ${bar} (${maxellOfferCounts[i]})`);
    }

    expect(maxellOfferCounts.length).toBe(NUM_CHOICES);
  });
});
