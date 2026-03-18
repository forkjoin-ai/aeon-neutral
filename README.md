# @a0n/aeon-neutral

Bounded dispute resolution via void walking. Two metacognitive walkers on a shared void surface, mediated by a third Skyrms walker that void walks the joint failure surface. Failed interactions enrich all three void boundaries, driving complement distributions toward alignment at the Skyrms nadir.

```
aeon-bazaar (unbounded)  → open negotiation, no termination guarantee
aeon-neutral (bounded)   → dispute resolution, convergence certificate, guaranteed termination
```

## Install

```bash
bun add @a0n/aeon-neutral
```

Depends on [aeon-bazaar](https://github.com/forkjoin-ai/aeon-bazaar) for void walker primitives (`VoidBoundary`, `complementDistribution`, `c0Choose`, `c1Measure`, `c3Adapt`).

## Quick Start

```ts
import { mediateThreeWalker } from '@a0n/aeon-neutral';

const result = mediateThreeWalker({
  numChoicesA: 2,
  numChoicesB: 2,
  maxRounds: 200,
  nadirThreshold: 0.15,
  payoff: (a, b) => {
    // Hawk-Dove: V=4, C=6
    if (a === 0 && b === 0) return [-1, -1];
    if (a === 0 && b === 1) return [4, 0];
    if (a === 1 && b === 0) return [0, 4];
    return [2, 2];
  },
});

if (result.settled) {
  console.log(`Converged in ${result.convergenceRound} rounds`);
  console.log(`Nadir:`, result.finalPayoffMatrix);
}
```

## Two Modes

**Passive Mediator** -- reads both walkers' void boundaries, computes the joint complement surface, proposes the nadir point. Walkers remain self-interested.

**Three-Walker** -- the mediator is a third metacognitive walker. Its choice space is all possible proposals `[offerA, offerB]`. Its payoff is the inter-walker distance surface: paid when walkers converge, penalized when they diverge. Runs its own c0-c3 loop over the proposal space.

```
Walker A ──── game choices ────┐
                                ├── joint void surface ──→ Skyrms Walker (site)
Walker B ──── game choices ────┘
```

## API

| Export | Description |
|--------|-------------|
| `NeutralMediator` | Passive mediation loop with c0-c3 adaptation on both walkers |
| `JointVoidSurface` | Outer product of two complement distributions, Manhattan distance, mutual information |
| `SkyrmsNadirDetector` | Convergence certificate: distance + kurtosis stability + positive MI for N rounds |
| `mediateThreeWalker()` | Full three-walker loop: propose, decide, interact, update, adapt, check |
| `VoidAttentionHead` | Attention mechanism over void boundary representations |
| `VoidCrossAttentionHead` | Cross-attention between two walkers' void surfaces |
| `VoidTransformerBlock` | Full transformer block over void states |
| `runChesterVMaxellMetacog()` | Personality-as-void negotiation scenario |

## The Skyrms Nadir

Named for Brian Skyrms (*Evolution of the Social Contract*, 1996). The nadir is the basin of attraction where accumulated failure information makes settlement the gradient descent direction for all walkers.

Three invariants certify convergence:
1. Inter-walker distance below threshold for N consecutive rounds
2. Joint kurtosis has stabilized (variance < epsilon)
3. Mutual information is positive (walkers are correlated, not independent)

## Benchmark Results

31 tests, 290 assertions, 180ms. Five classic games, five seeds each, 500 max rounds:

| Game | Settled | Avg Rounds | Final Distance | Acceptance Rate |
|------|---------|-----------|----------------|-----------------|
| Prisoner's Dilemma | 5/5 | 22 | 0.0000 | 64% |
| Stag Hunt | 5/5 | 22 | 0.0000 | 64% |
| Hawk-Dove | 1/5 | 401 | 0.7394 | 50% |
| Battle of Sexes | 0/5 | 500 | 0.9242 | 52% |
| Coordination (3x3) | 0/5 | 500 | 0.5579 | 76% |

Symmetric games (PD, Stag Hunt) converge fast -- both walkers' voids grow in the same shape. Asymmetric games (Hawk-Dove, Battle of Sexes) are harder -- the walkers' complement distributions want to go to different places.

## Formal Verification

TLA+ specifications in [aeon/companion-tests/formal/](https://github.com/forkjoin-ai/aeon):

- **SkyrmsNadir.tla** -- two walkers, 10 invariants, settlement/exhaustion liveness
- **SkyrmsThreeWalker.tla** -- three walkers, 7 invariants, three-way convergence liveness

## Development

```bash
bun test
```

## Related

- [aeon-bazaar](https://github.com/forkjoin-ai/aeon-bazaar) -- unbounded negotiation engine (void walker primitives)
- [Gnosis](https://github.com/forkjoin-ai/gnosis) -- GGL language (negotiation topologies)
- [Aeon](https://github.com/forkjoin-ai/aeon) -- TLA+ specs for convergence

## License

Copyright Taylor William Buley. All rights reserved.

MIT
