/**
 * convergence-theory.ts -- Ch17 Convergence Theory for Mediation
 *
 * Deepens aeon-neutral with mechanized convergence theory from ch17:
 *
 * 1. Convergence sandwich formula (Lyapunov drift bound)
 * 2. Skyrms-Bule-zero biconditional (nadir iff Bule = 0)
 * 3. Mediation-as-attenuation (Skyrms reduces deficit)
 * 4. Non-convergence detection (infeasible ZOPA proof)
 * 5. Personality-as-void encoding
 *
 * From SkyrmsNadirBule.lean (12 theorems):
 *   - Skyrms-as-community mapping
 *   - Bule-zero-iff-nadir biconditional
 *   - Algebraic nadir identification
 *   - Mediation-is-attenuation
 *   - Master theorem composing CommunityDominance + NegotiationEquilibrium + VoidWalking
 *
 * data-proof="lean4:bule_zero_iff_nadir" data-proof-status="verified"
 */

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/** Convergence certificate issued by the Skyrms nadir detector */
export interface ConvergenceCertificate {
  /** Round at which convergence was detected */
  round: number;
  /** Inter-walker distance at convergence */
  distance: number;
  /** Joint kurtosis at convergence */
  jointKurtosis: number;
  /** Mutual information at convergence */
  mutualInformation: number;
  /** Bule number at convergence (should be near 0) */
  buleNumber: number;
  /** Whether the nadir biconditional holds */
  nadirBiconditional: boolean;
  /** Predicted vs actual rounds (sandwich formula accuracy) */
  sandwichAccuracy: number;
  /** Certificate is valid */
  valid: boolean;
}

/** Personality encoded as initial void state */
export interface PersonalityVoid {
  /** Agent name */
  name: string;
  /** Initial rejection masses per choice (prior belief) */
  initialRejections: number[];
  /** Risk aversion coefficient (0 = risk-neutral, 1 = maximally averse) */
  riskAversion: number;
  /** External option value (BATNA) */
  batnaValue: number;
  /** Worst-case tolerance (WATNA threshold) */
  watnaThreshold: number;
  /** Description */
  description: string;
}

/** Non-convergence proof */
export interface NonConvergenceProof {
  /** Whether non-convergence is provably correct */
  provablyCorrect: boolean;
  /** The structural reason */
  reason: string;
  /** ZOPA feasibility */
  zopaFeasible: boolean;
  /** Minimum settlement price A would accept */
  minAcceptableA: number;
  /** Maximum settlement price B would accept */
  maxAcceptableB: number;
  /** The gap (negative = infeasible) */
  gap: number;
}

// ---------------------------------------------------------------------------
// Convergence Sandwich Formula
// ---------------------------------------------------------------------------

/**
 * Predict convergence bounds from initial conditions.
 *
 * From ch17: rounds ≤ ceil(deficit / delta)
 *   deficit = initial inverse Bule
 *   delta = per-round deficit reduction (from mutual information gain)
 *
 * THM-CONVERGENCE-SANDWICH: the bound is tight.
 */
export function predictConvergence(
  initialBule: number,
  perRoundDelta: number,
  confidenceLevel: number = 0.95
): {
  expectedRounds: number;
  lowerBound: number;
  upperBound: number;
  guaranteed: boolean;
} {
  if (perRoundDelta <= 0) {
    return {
      expectedRounds: Infinity,
      lowerBound: Infinity,
      upperBound: Infinity,
      guaranteed: false,
    };
  }

  const expected = Math.ceil(initialBule / perRoundDelta);
  // Confidence bounds: variance scales as sqrt(T)
  const variance = Math.sqrt(expected);
  const zScore = confidenceLevel === 0.95 ? 1.96 : 1.645;

  return {
    expectedRounds: expected,
    lowerBound: Math.max(1, Math.floor(expected - zScore * variance)),
    upperBound: Math.ceil(expected + zScore * variance),
    guaranteed: true,
  };
}

// ---------------------------------------------------------------------------
// Skyrms-Bule-Zero Biconditional
// ---------------------------------------------------------------------------

/**
 * Check the Bule-zero-iff-nadir biconditional.
 *
 * From SkyrmsNadirBule.lean:
 *   Bule = 0 ↔ walkers are at the nadir (fixed point)
 *
 * When Bule reaches 0, the complement distributions have converged
 * to identical shapes. The nadir IS the equilibrium.
 *
 * data-proof="lean4:bule_zero_iff_nadir" data-proof-status="verified"
 */
export function checkBuleNadirBiconditional(
  buleA: number,
  buleB: number,
  interWalkerDistance: number,
  epsilon: number = 0.05
): {
  buleZero: boolean;
  atNadir: boolean;
  biconditionalHolds: boolean;
} {
  const buleZero = buleA < epsilon && buleB < epsilon;
  const atNadir = interWalkerDistance < epsilon;

  // The biconditional: Bule = 0 ↔ at nadir
  // Both must be true or both false
  const biconditionalHolds = buleZero === atNadir;

  return { buleZero, atNadir, biconditionalHolds };
}

/**
 * Compute the Bule number from a complement distribution.
 *
 * Bule = 1 - entropy / max_entropy
 * When entropy is maximum (uniform): Bule = 0 (no structure, convergence)
 * When entropy is minimum (peaked): Bule → 1 (high structure, exploring)
 *
 * Note: this is INVERSE to the aeon-3d convention where inverseBule
 * measures structure. Here Bule = 0 means converged (peace).
 */
export function buleFromDistribution(distribution: number[]): number {
  if (distribution.length <= 1) return 0;

  let entropy = 0;
  for (const p of distribution) {
    if (p > 1e-10) {
      entropy -= p * Math.log2(p);
    }
  }

  const maxEntropy = Math.log2(distribution.length);
  return maxEntropy > 0 ? 1 - entropy / maxEntropy : 0;
}

// ---------------------------------------------------------------------------
// Mediation-as-Attenuation
// ---------------------------------------------------------------------------

/**
 * Measure the attenuation effect of mediation.
 *
 * From SkyrmsNadirBule.lean: mediation-is-attenuation.
 * The Skyrms walker reduces the inter-walker distance faster
 * than unmediated negotiation. The attenuation ratio measures
 * how much faster.
 *
 * attenuation = unmediatedRate / mediatedRate
 * > 1 means mediation helps (faster convergence)
 * = 1 means mediation has no effect
 * < 1 means mediation hurts (unlikely in well-designed games)
 */
export function mediationAttenuation(
  unmediatedConvergenceRate: number,
  mediatedConvergenceRate: number
): {
  ratio: number;
  effective: boolean;
  description: string;
} {
  if (mediatedConvergenceRate <= 0) {
    return {
      ratio: Infinity,
      effective: false,
      description: 'Mediation failed to produce convergence',
    };
  }

  const ratio = unmediatedConvergenceRate / mediatedConvergenceRate;
  const effective = ratio > 1;

  return {
    ratio,
    effective,
    description: effective
      ? `Mediation attenuates deficit ${ratio.toFixed(1)}x faster`
      : `Mediation provides no improvement (ratio ${ratio.toFixed(2)})`,
  };
}

// ---------------------------------------------------------------------------
// Non-Convergence Detection
// ---------------------------------------------------------------------------

/**
 * Prove that non-convergence is correct for a given scenario.
 *
 * From ch17: Chester v Maxell is intentionally no-win.
 * Non-convergence IS the correct answer when the ZOPA is infeasible.
 * The void walkers are rationally refusing to settle.
 *
 * This is a PROOF, not a failure mode. The algorithm works correctly
 * by detecting that no Pareto improvement exists.
 */
export function proveNonConvergence(
  partyABatna: number,
  partyAWatna: number,
  partyBBatna: number,
  partyBWatna: number,
  partyADescription: string = 'Party A',
  partyBDescription: string = 'Party B'
): NonConvergenceProof {
  // ZOPA = range where both parties can agree
  // A accepts: payoff >= A's BATNA
  // B accepts: payoff <= B's WATNA
  // Feasible when A's BATNA < B's WATNA
  const gap = partyBWatna - partyABatna;
  const feasible = gap > 0;

  if (feasible) {
    return {
      provablyCorrect: false,
      reason: `ZOPA exists (range ${gap.toFixed(2)}). Convergence should be possible.`,
      zopaFeasible: true,
      minAcceptableA: partyABatna,
      maxAcceptableB: partyBWatna,
      gap,
    };
  }

  return {
    provablyCorrect: true,
    reason:
      `No settlement zone: ${partyADescription}'s minimum acceptable (${partyABatna}) ` +
      `exceeds ${partyBDescription}'s maximum acceptable (${partyBWatna}). ` +
      `Gap = ${gap.toFixed(2)}. Non-convergence is rational -- ` +
      `the void walkers correctly refuse to settle.`,
    zopaFeasible: false,
    minAcceptableA: partyABatna,
    maxAcceptableB: partyBWatna,
    gap,
  };
}

// ---------------------------------------------------------------------------
// Personality-as-Void
// ---------------------------------------------------------------------------

/**
 * Create a personality-encoded void state for Chester v Maxell.
 *
 * From ch17: the void IS the personality. Two different void boundaries
 * encode two different risk profiles. Settlement emerges when both
 * personalities' rejection patterns converge to the same nadir.
 */
export function createChesterPersonality(numChoices: number): PersonalityVoid {
  // Chester: risk-averse, health anxiety, favors certainty
  const rejections = new Array(numChoices).fill(0);
  // Pre-load rejections on high-payment options (conservative)
  for (let i = Math.floor(numChoices * 0.6); i < numChoices; i++) {
    rejections[i] = 3; // prior belief: high payments are bad
  }

  return {
    name: 'Chester',
    initialRejections: rejections,
    riskAversion: 0.8,
    batnaValue: 0, // Can refuse to pay, countersue
    watnaThreshold: 95_000, // Trial costs
    description:
      'Risk-averse. Health anxiety about mold. Favors certainty over savings. ' +
      'Initial void rejects high-payment options.',
  };
}

export function createMaxellPersonality(numChoices: number): PersonalityVoid {
  // Maxell: profit-driven, aggressive, threatens trial
  const rejections = new Array(numChoices).fill(0);
  // Pre-load rejections on low-payment options (maximalist)
  for (let i = 0; i < Math.floor(numChoices * 0.4); i++) {
    rejections[i] = 3; // prior belief: low payments are bad
  }

  return {
    name: 'Maxell',
    initialRejections: rejections,
    riskAversion: 0.3,
    batnaValue: -95_000, // Trial costs Maxell $95K too
    watnaThreshold: 20_000, // Drywall difference
    description:
      'Profit-driven. Aggressive trial threat. Initial void rejects ' +
      'low-payment options. BATNA includes trial cost risk.',
  };
}

/**
 * Issue a convergence certificate from round data.
 *
 * Composes all the above: Bule-nadir biconditional, sandwich accuracy,
 * and certificate validity.
 */
export function issueConvergenceCertificate(
  round: number,
  distance: number,
  jointKurtosis: number,
  mutualInformation: number,
  distributionA: number[],
  distributionB: number[],
  predictedRounds: number
): ConvergenceCertificate {
  const buleA = buleFromDistribution(distributionA);
  const buleB = buleFromDistribution(distributionB);
  const avgBule = (buleA + buleB) / 2;

  const nadir = checkBuleNadirBiconditional(buleA, buleB, distance);
  const sandwichAccuracy =
    predictedRounds > 0 ? 1 - Math.abs(round - predictedRounds) / predictedRounds : 0;

  return {
    round,
    distance,
    jointKurtosis,
    mutualInformation,
    buleNumber: avgBule,
    nadirBiconditional: nadir.biconditionalHolds,
    sandwichAccuracy,
    valid: nadir.biconditionalHolds && distance < 0.05 && mutualInformation > 0,
  };
}
