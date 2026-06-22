# Industrial Extension: Controlled Near-Critical Policy Benchmarks

## Purpose of the controlled benchmark program

The controlled benchmark campaign was designed as a methodological bridge between the single-buffer near-critical model and a future multi-node supply-chain digital twin. The central question was not whether a large simulated supply network can produce interesting behavior, but whether the near-critical control logic retains operational value in the simplest possible industrial setting. For this reason, the first experimental layer was deliberately restricted to a single inventory node with stochastic replenishment interruptions and Poisson demand. This design isolates the mechanism before introducing network effects, routing, multiple suppliers, warehouse interactions, or demand correlations.

The benchmark therefore functions as a controlled validation stage. If the threshold logic does not show signal in the one-node setting, then embedding it in a digital twin would risk confusing genuine model value with artifacts of a larger simulation. Conversely, if the one-node benchmark reveals a stable relationship between physical criticality, preventive thresholds, and economic value, then the next step is justified: testing whether the same relationship survives in a small supply network and eventually in a large supply-chain digital twin.

## Model setting

The controlled system follows the same basic inventory-disruption structure used in the near-critical paper. Inventory evolves under fixed replenishment capacity when the system is operational and zero replenishment during disruptive periods. Demand is stochastic and modeled as Poisson. The principal stability margin is δ = (1 − p)C − λ, where C is replenishment capacity, p is disruption probability, and λ is the demand rate.

Small positive values of δ correspond to lean, near-critical operation: the system remains stable in expectation but has limited slack against random interruptions and demand fluctuations. The experiments compare run-to-failure behavior with preventive threshold policies. A threshold policy intervenes when inventory falls below a buffer level s*, paying preventive cost K_prevent and resetting the buffer. Collapse or stockout incurs cost K_fail.

## Benchmark 1: policy comparison

The first benchmark compared several simple policies across representative regimes: relaxed, moderate, near-critical, and fragile. The main observation was that a threshold policy becomes relevant as the system moves toward small δ. However, simple near-critical formulas of the form proportional to σ²/δ were too aggressive in the most fragile regimes. The oracle grid search found moderate threshold levels rather than exploding thresholds.

The key methodological result from this stage was the discovery that the threshold problem is real but nontrivial: the optimal action is not simply to intervene as early as possible as δ approaches zero. Instead, the system exhibits a cost-sensitive optimum in which preventive resets must balance avoided stockouts against excessive intervention frequency.

## Benchmark 2: oracle surface discovery

The second benchmark mapped the cost-minimizing threshold over a grid of disruption probabilities and demand rates. The resulting oracle surface showed that the optimal threshold increases as physical slack decreases. Near-critical scenarios had a substantially larger mean threshold than relaxed scenarios.

The interpretable rule s* ≈ a + b log(1 + 1/δ) + c p + d ρ captured meaningful signal but achieved only moderate explanatory power. This indicates structure, but not a globally smooth linear law. The threshold surface appears closer to a regime transition or nonlinear saturated response than to a simple linear function of δ, p, and utilization.

## Benchmark 3: critical boundary discovery

The third benchmark shifted the question from “what is the optimal threshold?” to “when is preventive control economically valuable?” For each scenario, the benchmark compared run-to-failure with the oracle threshold policy and computed relative benefit. At the baseline cost ratio K_fail/K_prevent = 10, threshold intervention improved service and reduced risk but often yielded only modest economic improvement.

This was a decisive clarification. The near-critical system can benefit operationally from threshold control, but if collapse is not sufficiently costly relative to prevention, the economic advantage is muted. This finding motivated the fourth benchmark.

## Benchmark 4: economic critical boundary

The fourth benchmark varied the collapse-to-prevention cost ratio. This produced the strongest result of the controlled campaign. In the broad sweep, mean relative benefit was essentially zero at low cost ratios, became visible around ratios 10–25, and became clearly material at ratios 50 and 100. In the fine sweep over ratios 10–50, the global empirical boundary appeared around K_fail/K_prevent ≈ 40.

At this boundary, more than half of the tested scenarios exceeded a 5% relative benefit threshold. Below this region, intervention was often not economically material. Above this region, preventive threshold control became broadly valuable.

The delta-stratified results sharpen the interpretation. For low-delta regimes, especially δ ≤ 5, the benefit of threshold control became large as the cost ratio increased. For relaxed regimes with large δ, even high collapse costs produced much smaller gains. Thus the benchmark does not merely say that higher failure cost makes intervention more useful. It shows that high failure cost and physical near-criticality interact.

## Main conclusion

The controlled experiments support a two-dimensional criticality framework. The value of preventive threshold control is governed by the interaction of physical criticality and economic criticality.

Physical criticality is measured by the stability margin δ = (1 − p)C − λ. Economic criticality is measured by the cost ratio K_fail/K_prevent.

Neither dimension is sufficient by itself. A system may be physically fragile, but if failure is cheap relative to preventive action, intervention has limited economic value. Conversely, a system may have a very high collapse cost, but if it operates far from the stability boundary, preventive intervention yields only modest incremental value. Material benefit emerges when the system is both near-critical and economically exposed.

This result reframes the managerial interpretation of near-critical systems. Traditional KPIs such as average inventory, utilization, or historical service level measure current performance. The near-critical framework instead measures distance to operational fragility. The controlled benchmarks indicate that this distance becomes decision-relevant when combined with the economics of collapse. The practical question is therefore not only “how close is the system to instability?” but “how expensive is instability once reached?”

## Relevance to the original paper

The original paper establishes the stochastic near-critical model, the role of the stability margin, the limitations of classical diffusion approximations, and the usefulness of threshold control in the single-buffer setting. The controlled benchmark program provides an industrial extension: it translates the mathematical threshold policy into a decision experiment framed in terms of service, stockouts, total cost, and economic benefit.

The strongest new contribution is the empirical economic boundary near K_fail/K_prevent ≈ 40 in the tested grid. This should not be presented as a universal constant. It is an empirical boundary for the chosen parameter grid, cost structure, time horizon, and reset mechanism. Its value may shift under different demand distributions, holding costs, replenishment assumptions, or network topology. Nevertheless, the qualitative conclusion is robust and important: the operational value of near-critical threshold policies depends on a joint physical-economic boundary.

## Implication for the digital twin roadmap

The controlled benchmark justifies moving to the next stage, but it also prevents overclaiming. The correct next experiment is a small network, not an immediate large digital twin. A minimal network such as supplier → warehouse → customer with five to ten nodes can test whether the same boundary persists when disruptions propagate across links and inventory positions.

Only after this small-network validation should the framework be scaled to a synthetic supply-chain digital twin with many suppliers, factories, distribution centers, warehouses, and customers. The digital twin should therefore be built around the controlled finding: preventive value = F(physical criticality, economic criticality, network propagation). The one-node benchmarks have characterized the first two terms. The small-network and full digital-twin stages should test the third.

## Recommended insertion in the paper

This material can be used as an applied or managerial extension section rather than as a replacement for the main theoretical contribution. A suitable placement would be after the numerical results or in an appendix titled “Industrial cost-ratio benchmark” or “Managerial interpretation: physical and economic criticality.” The paper should emphasize that the benchmark is controlled, synthetic, and designed to test transferability of the threshold policy to supply-chain decision settings.

