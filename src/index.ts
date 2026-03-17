/**
 * aeon-neutral -- Neutral Mediation Engine
 *
 * Takes two negotiation topologies (two metacognitive walkers' void boundaries)
 * and mediates them toward the Skyrms nadir -- the basin of attraction where
 * accumulated failure information makes settlement the gradient descent direction.
 *
 * Two modes:
 * 1. Passive mediator: reads both void boundaries, computes joint complement
 *    surface, proposes offers. The walkers remain self-interested.
 * 2. Three-walker: the mediator IS a third metacognitive walker whose payoff
 *    matrix is the Skyrms convergence site itself. It void walks the joint
 *    failure surface -- its tombstones are proposals that didn't converge.
 *
 * Uses aeon-bazaar's void walker engine for complement distribution, kurtosis,
 * inverse Bule, and void boundary primitives.
 */

export { NeutralMediator, type MediationConfig, type MediationResult } from './mediator';
export { JointVoidSurface, type JointState } from './joint-surface';
export { SkyrmsNadirDetector, type NadirCertificate } from './nadir-detector';
export {
  mediateThreeWalker,
  type ThreeWalkerConfig,
  type ThreeWalkerResult,
  type ThreeWalkerRoundResult,
  type SkyrmsWalkerState,
  type SkyrmsPayoffMatrix,
  createSkyrmsWalkerState,
  computeSkyrmsPayoff,
  skyrmsC0Choose,
  skyrmsC0Update,
  skyrmsC1Measure,
  skyrmsC3Adapt,
} from './skyrms-walker';
