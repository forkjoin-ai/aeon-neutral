/**
 * Void Walking Transformer
 *
 * The structural identification between void walking and attention:
 *
 *   Query  = current proposal (what the walker is considering)
 *   Key    = void boundary entries (rejection history per choice)
 *   Value  = complement weights (exp(-eta * voidDensity))
 *   Score  = Q·K^T → complement(query, keys) = softmax(-eta * voidCounts)
 *   Output = weighted sum of values = the proposal distribution
 *
 * This is not an analogy. The complement distribution IS softmax attention
 * over the void boundary. The eta parameter IS the temperature. Neighborhood
 * poisoning IS the attention pattern spreading to adjacent keys.
 *
 * Multi-head: each walker is a head. The Skyrms walker is cross-attention.
 *
 *   Head A: self-attention over Walker A's void boundary
 *   Head B: self-attention over Walker B's void boundary
 *   Head S: cross-attention over the joint void surface
 *
 * The transformer processes one "token" per round: the interaction outcome.
 * The residual stream is the void boundary itself -- it accumulates.
 * Layer norm is void decay. Feed-forward is the c3 adaptation.
 *
 * The forward pass of one round:
 *   1. Multi-head void attention (Q=proposal, K=void, V=complement)
 *   2. Residual connection (void boundary persists across rounds)
 *   3. Layer norm (optional void decay)
 *   4. Feed-forward (c3 adapt: gait selection, eta/exploration update)
 *   5. Output: next proposal distribution
 */

import {
  type VoidBoundary,
  type Gait,
  createVoidBoundary,
  complementDistribution,
  updateVoidBoundary,
  decayVoidBoundary,
  shannonEntropy,
  excessKurtosis,
} from '../../gnosis/src/runtime/void-walker.js';

// ============================================================================
// Void Attention Head
// ============================================================================

export interface VoidAttentionConfig {
  /** Number of choices (sequence length of keys/values) */
  numChoices: number;
  /** Initial temperature (eta = 1/temperature in standard attention) */
  eta: number;
  /** Neighborhood radius for attention spread (0 = sharp, >0 = soft) */
  neighborhoodRadius: number;
  /** Void decay rate per round (0 = no decay, 0.01 = slow forgetting) */
  decayRate: number;
}

export interface AttentionOutput {
  /** Attention weights (the complement distribution) */
  weights: number[];
  /** Selected index (argmax or sampled) */
  selected: number;
  /** Entropy of the attention distribution */
  entropy: number;
  /** Kurtosis of the attention distribution */
  kurtosis: number;
}

/**
 * A single void attention head.
 *
 * Keys and values are stored in the void boundary.
 * The query is implicit: "what should I do next?"
 * The attention score is the complement distribution.
 */
export class VoidAttentionHead {
  readonly config: VoidAttentionConfig;
  /** The void boundary IS the KV cache */
  boundary: VoidBoundary;
  /** Current temperature (eta) */
  eta: number;
  /** Total queries processed */
  queryCount: number;

  constructor(config: VoidAttentionConfig) {
    this.config = config;
    this.boundary = createVoidBoundary(config.numChoices);
    this.eta = config.eta;
    this.queryCount = 0;
  }

  /**
   * Forward pass: compute attention over the void boundary.
   *
   * attention(Q, K, V) = softmax(Q·K^T / sqrt(d_k)) · V
   *
   * In void walking:
   *   Q·K^T = -eta * voidCounts (negative because more rejection = less attention)
   *   softmax(-eta * voidCounts) = complementDistribution(boundary, eta)
   *   V = the choices themselves (identity projection)
   *
   * The output is the complement distribution: a probability over choices
   * weighted inversely by rejection history.
   */
  attend(rng?: () => number): AttentionOutput {
    this.queryCount++;
    const weights = complementDistribution(this.boundary, this.eta);
    const entropy = shannonEntropy(weights);
    const kurtosis = excessKurtosis(weights);

    // Sample or argmax
    let selected: number;
    if (rng) {
      const r = rng();
      let cum = 0;
      selected = weights.length - 1;
      for (let i = 0; i < weights.length; i++) {
        cum += weights[i];
        if (r < cum) { selected = i; break; }
      }
    } else {
      // Argmax (greedy decoding)
      selected = 0;
      for (let i = 1; i < weights.length; i++) {
        if (weights[i] > weights[selected]) selected = i;
      }
    }

    return { weights, selected, entropy, kurtosis };
  }

  /**
   * Update the KV cache: a choice was rejected.
   *
   * This is the "write" to the void boundary -- the attention head
   * remembers what failed. With neighborhood radius > 0, adjacent
   * keys also get lighter updates (the attention pattern spreads).
   */
  reject(choiceIdx: number, magnitude: number = 1): void {
    updateVoidBoundary(this.boundary, choiceIdx, magnitude);

    // Neighborhood spread (attention pattern)
    const r = this.config.neighborhoodRadius;
    if (r > 0) {
      for (let d = 1; d <= r; d++) {
        const neighborMag = Math.max(1, Math.round(magnitude / d));
        if (choiceIdx - d >= 0) {
          updateVoidBoundary(this.boundary, choiceIdx - d, neighborMag);
        }
        if (choiceIdx + d < this.config.numChoices) {
          updateVoidBoundary(this.boundary, choiceIdx + d, neighborMag);
        }
      }
    }
  }

  /**
   * Layer norm: apply void decay (forgetting factor).
   *
   * In transformers, LayerNorm stabilizes the residual stream.
   * In void walking, decay prevents the boundary from saturating.
   */
  layerNorm(): void {
    if (this.config.decayRate > 0) {
      decayVoidBoundary(this.boundary, this.config.decayRate);
    }
  }
}

// ============================================================================
// Cross-Attention Head (Skyrms Walker = Cross-Attention)
// ============================================================================

export interface CrossAttentionConfig {
  numChoicesA: number;
  numChoicesB: number;
  eta: number;
  neighborhoodRadius: number;
  decayRate: number;
}

export interface CrossAttentionOutput {
  /** Joint attention weights (flattened A x B) */
  weights: number[];
  /** Selected proposal [offerA, offerB] */
  selected: [number, number];
  /** Joint entropy */
  entropy: number;
  /** Joint kurtosis */
  kurtosis: number;
}

/**
 * Cross-attention head: attends to two void boundaries simultaneously.
 *
 * The Skyrms walker IS cross-attention:
 *   Q = "what proposal should I make?"
 *   K_A = Walker A's void boundary
 *   K_B = Walker B's void boundary
 *   score(i,j) = complementA[i] * complementB[j]  (outer product)
 *   output = argmax over the joint surface
 *
 * This is exactly cross-attention where the query attends to two
 * key-value stores and combines them multiplicatively.
 */
export class VoidCrossAttentionHead {
  readonly config: CrossAttentionConfig;
  /** The Skyrms walker's own void boundary (over proposal space) */
  proposalVoid: VoidBoundary;
  eta: number;
  queryCount: number;

  constructor(config: CrossAttentionConfig) {
    this.config = config;
    this.proposalVoid = createVoidBoundary(config.numChoicesA * config.numChoicesB);
    this.eta = config.eta;
    this.queryCount = 0;
  }

  /**
   * Cross-attend: compute joint attention from two void boundaries.
   *
   * Three attention sources are combined:
   * 1. Walker A's complement (what A would choose)
   * 2. Walker B's complement (what B would choose)
   * 3. Skyrms walker's own complement (which proposals have worked)
   *
   * The joint score is the element-wise product of all three.
   * This is gated cross-attention: the Skyrms walker's void
   * acts as a gate on the joint surface.
   */
  crossAttend(
    boundaryA: VoidBoundary,
    boundaryB: VoidBoundary,
    etaA: number,
    etaB: number,
    rng?: () => number,
  ): CrossAttentionOutput {
    this.queryCount++;

    const distA = complementDistribution(boundaryA, etaA);
    const distB = complementDistribution(boundaryB, etaB);
    const distS = complementDistribution(this.proposalVoid, this.eta);

    // Gated cross-attention: joint = distA[i] * distB[j] * distS[i*B+j]
    const nA = this.config.numChoicesA;
    const nB = this.config.numChoicesB;
    const weights = new Array(nA * nB);
    let sum = 0;

    for (let i = 0; i < nA; i++) {
      for (let j = 0; j < nB; j++) {
        const flat = i * nB + j;
        const w = distA[i] * distB[j] * distS[flat];
        weights[flat] = w;
        sum += w;
      }
    }

    // Normalize
    if (sum > 0) {
      for (let k = 0; k < weights.length; k++) weights[k] /= sum;
    }

    const entropy = shannonEntropy(weights);
    const kurtosis = excessKurtosis(weights);

    // Select proposal
    let selectedFlat: number;
    if (rng) {
      const r = rng();
      let cum = 0;
      selectedFlat = weights.length - 1;
      for (let k = 0; k < weights.length; k++) {
        cum += weights[k];
        if (r < cum) { selectedFlat = k; break; }
      }
    } else {
      selectedFlat = 0;
      for (let k = 1; k < weights.length; k++) {
        if (weights[k] > weights[selectedFlat]) selectedFlat = k;
      }
    }

    const selected: [number, number] = [
      Math.floor(selectedFlat / nB),
      selectedFlat % nB,
    ];

    return { weights, selected, entropy, kurtosis };
  }

  /**
   * Reject a proposal: update the Skyrms void with 2D neighborhood.
   */
  rejectProposal(
    offerA: number,
    offerB: number,
    magnitude: number = 1,
  ): void {
    const nB = this.config.numChoicesB;
    const flat = offerA * nB + offerB;
    updateVoidBoundary(this.proposalVoid, flat, magnitude);

    const r = this.config.neighborhoodRadius;
    if (r > 0) {
      for (let da = -r; da <= r; da++) {
        for (let db = -r; db <= r; db++) {
          if (da === 0 && db === 0) continue;
          const nA = offerA + da;
          const nBi = offerB + db;
          if (nA >= 0 && nA < this.config.numChoicesA && nBi >= 0 && nBi < this.config.numChoicesB) {
            const neighborFlat = nA * nB + nBi;
            const dist = Math.abs(da) + Math.abs(db);
            updateVoidBoundary(this.proposalVoid, neighborFlat, Math.max(1, Math.round(magnitude / dist)));
          }
        }
      }
    }
  }

  layerNorm(): void {
    if (this.config.decayRate > 0) {
      decayVoidBoundary(this.proposalVoid, this.config.decayRate);
    }
  }
}

// ============================================================================
// Void Transformer Block
// ============================================================================

export interface VoidTransformerConfig {
  numChoicesA: number;
  numChoicesB: number;
  /** Number of self-attention heads per walker (default 1) */
  numHeads?: number;
  /** Temperature for self-attention heads */
  selfEta?: number;
  /** Temperature for cross-attention */
  crossEta?: number;
  /** Neighborhood radius (0=sharp attention, 1+=soft) */
  neighborhoodRadius?: number;
  /** Void decay rate per round */
  decayRate?: number;
  /** Feed-forward adaptation cadence (rounds between c3 adapt) */
  ffnCadence?: number;
}

export interface TransformerRoundOutput {
  /** Self-attention outputs for walker A (one per head) */
  selfA: AttentionOutput[];
  /** Self-attention outputs for walker B */
  selfB: AttentionOutput[];
  /** Cross-attention output (the Skyrms proposal) */
  cross: CrossAttentionOutput;
  /** What the walkers actually played */
  offerA: number;
  offerB: number;
  /** Whether the cross-attention proposal was accepted */
  proposalAccepted: boolean;
  /** Current gait of each component */
  gaitA: Gait;
  gaitB: Gait;
  gaitS: Gait;
}

/**
 * Void Walking Transformer: one block = one round of mediation.
 *
 * Architecture per round:
 *
 *   ┌─────────────────────────────────────────┐
 *   │         Multi-Head Self-Attention        │
 *   │  Head A: attend(voidA) → complementA     │
 *   │  Head B: attend(voidB) → complementB     │
 *   ├─────────────────────────────────────────┤
 *   │         Cross-Attention (Skyrms)         │
 *   │  crossAttend(voidA, voidB, voidS)       │
 *   │  → joint proposal distribution           │
 *   ├─────────────────────────────────────────┤
 *   │         Residual + LayerNorm             │
 *   │  void boundaries persist (residual)      │
 *   │  optional decay (layer norm)             │
 *   ├─────────────────────────────────────────┤
 *   │         Feed-Forward (c3 Adapt)          │
 *   │  gait selection, eta/exploration update  │
 *   └─────────────────────────────────────────┘
 *
 * The "token" is the interaction outcome.
 * The "residual stream" is the void boundary.
 * The "KV cache" is the accumulated rejection history.
 * "Training" is the mediation itself -- every round is a gradient step.
 */
export class VoidTransformerBlock {
  readonly config: VoidTransformerConfig;
  readonly headsA: VoidAttentionHead[];
  readonly headsB: VoidAttentionHead[];
  readonly crossHead: VoidCrossAttentionHead;

  private roundCount = 0;
  private gaitA: Gait = 'stand';
  private gaitB: Gait = 'stand';
  private gaitS: Gait = 'stand';
  private momentumA = 0;
  private momentumB = 0;
  private momentumS = 0;

  constructor(config: VoidTransformerConfig) {
    this.config = config;
    const numHeads = config.numHeads ?? 1;
    const selfEta = config.selfEta ?? 2.0;
    const crossEta = config.crossEta ?? 2.0;
    const radius = config.neighborhoodRadius ?? 1;
    const decay = config.decayRate ?? 0;

    // Multi-head self-attention for each walker
    this.headsA = Array.from({ length: numHeads }, () =>
      new VoidAttentionHead({
        numChoices: config.numChoicesA,
        eta: selfEta,
        neighborhoodRadius: radius,
        decayRate: decay,
      }),
    );
    this.headsB = Array.from({ length: numHeads }, () =>
      new VoidAttentionHead({
        numChoices: config.numChoicesB,
        eta: selfEta,
        neighborhoodRadius: radius,
        decayRate: decay,
      }),
    );

    // Cross-attention (Skyrms)
    this.crossHead = new VoidCrossAttentionHead({
      numChoicesA: config.numChoicesA,
      numChoicesB: config.numChoicesB,
      eta: crossEta,
      neighborhoodRadius: radius,
      decayRate: decay,
    });
  }

  /**
   * Forward pass: one round of the void walking transformer.
   *
   * 1. Self-attention: each walker attends to its own void
   * 2. Cross-attention: Skyrms walker attends to both voids + its own
   * 3. Interaction: walkers decide, payoffs evaluated
   * 4. Residual update: void boundaries absorb the outcome
   * 5. Layer norm: optional decay
   * 6. Feed-forward: gait adaptation
   */
  forward(
    payoff: (a: number, b: number) => [number, number],
    rng: () => number,
  ): TransformerRoundOutput {
    this.roundCount++;
    const cadence = this.config.ffnCadence ?? 5;

    // 1. Multi-head self-attention
    const selfA = this.headsA.map((h) => h.attend(rng));
    const selfB = this.headsB.map((h) => h.attend(rng));

    // Merge multi-head outputs (average the selected indices, take mode)
    const ownChoiceA = selfA[0].selected; // For now: first head wins
    const ownChoiceB = selfB[0].selected;

    // 2. Cross-attention
    // Use the primary head's boundary as the walker's void
    const cross = this.crossHead.crossAttend(
      this.headsA[0].boundary,
      this.headsB[0].boundary,
      this.headsA[0].eta,
      this.headsB[0].eta,
      rng,
    );

    // 3. Walker decisions: accept cross-attention proposal or play own
    const distA = complementDistribution(this.headsA[0].boundary, this.headsA[0].eta);
    const distB = complementDistribution(this.headsB[0].boundary, this.headsB[0].eta);
    const [proposalA, proposalB] = cross.selected;

    const offerA = distA[proposalA] >= distA[ownChoiceA] ? proposalA : ownChoiceA;
    const offerB = distB[proposalB] >= distB[ownChoiceB] ? proposalB : ownChoiceB;
    const proposalAccepted = offerA === proposalA && offerB === proposalB;

    // 4. Evaluate and update residual (void boundaries)
    const [payA, payB] = payoff(offerA, offerB);

    // Self-attention heads: update based on own payoff
    if (payA < payB || payA < 0) {
      for (const head of this.headsA) {
        head.reject(offerA, payA < 0 ? Math.abs(payA) : 1);
      }
      this.momentumA = 0;
    } else {
      this.momentumA++;
    }

    if (payB < payA || payB < 0) {
      for (const head of this.headsB) {
        head.reject(offerB, payB < 0 ? Math.abs(payB) : 1);
      }
      this.momentumB = 0;
    } else {
      this.momentumB++;
    }

    // Cross-pollinate: each walker learns from the other's choice
    if (offerA !== offerB) {
      for (const head of this.headsA) {
        head.reject(Math.min(offerB, this.config.numChoicesA - 1), 1);
      }
      for (const head of this.headsB) {
        head.reject(Math.min(offerA, this.config.numChoicesB - 1), 1);
      }
    }

    // Cross-attention head: update based on whether proposal improved things
    if (!proposalAccepted || offerA !== offerB) {
      this.crossHead.rejectProposal(proposalA, proposalB, proposalAccepted ? 1 : 2);
      this.momentumS = 0;
    } else {
      this.momentumS++;
    }

    // 5. Layer norm (void decay)
    for (const head of this.headsA) head.layerNorm();
    for (const head of this.headsB) head.layerNorm();
    this.crossHead.layerNorm();

    // 6. Feed-forward (c3 gait adaptation)
    if (this.roundCount % cadence === 0) {
      this.gaitA = this.adaptGait(this.gaitA, selfA[0].kurtosis, this.momentumA, this.roundCount);
      this.gaitB = this.adaptGait(this.gaitB, selfB[0].kurtosis, this.momentumB, this.roundCount);
      this.gaitS = this.adaptGait(this.gaitS, cross.kurtosis, this.momentumS, this.roundCount);

      // Update etas based on gait
      for (const head of this.headsA) head.eta = this.etaForGait(this.gaitA);
      for (const head of this.headsB) head.eta = this.etaForGait(this.gaitB);
      this.crossHead.eta = this.etaForGait(this.gaitS);
    }

    return {
      selfA,
      selfB,
      cross,
      offerA,
      offerB,
      proposalAccepted,
      gaitA: this.gaitA,
      gaitB: this.gaitB,
      gaitS: this.gaitS,
    };
  }

  /** Gait adaptation (the feed-forward network) */
  private adaptGait(current: Gait, kurtosis: number, momentum: number, round: number): Gait {
    if (round < 5) return 'stand';
    if (momentum >= 5) {
      // Accelerate
      if (current === 'stand') return 'trot';
      if (current === 'trot') return 'canter';
      return 'gallop';
    }
    if (kurtosis > 0.5 && round > 20) return current === 'stand' ? 'trot' : current === 'trot' ? 'canter' : 'gallop';
    if (kurtosis < -0.5 && current !== 'stand') {
      // Decelerate
      if (current === 'gallop') return 'canter';
      if (current === 'canter') return 'trot';
      return 'stand';
    }
    return current;
  }

  /** Map gait to eta (temperature) */
  private etaForGait(gait: Gait): number {
    switch (gait) {
      case 'stand': return 1.5;
      case 'trot': return 2.5;
      case 'canter': return 4.0;
      case 'gallop': return 7.0;
    }
  }

  /** Get the current state for external inspection */
  getState(): {
    headsA: { boundary: VoidBoundary; eta: number }[];
    headsB: { boundary: VoidBoundary; eta: number }[];
    crossHead: { boundary: VoidBoundary; eta: number };
    gaits: { a: Gait; b: Gait; s: Gait };
    roundCount: number;
  } {
    return {
      headsA: this.headsA.map((h) => ({ boundary: h.boundary, eta: h.eta })),
      headsB: this.headsB.map((h) => ({ boundary: h.boundary, eta: h.eta })),
      crossHead: { boundary: this.crossHead.proposalVoid, eta: this.crossHead.eta },
      gaits: { a: this.gaitA, b: this.gaitB, s: this.gaitS },
      roundCount: this.roundCount,
    };
  }
}
