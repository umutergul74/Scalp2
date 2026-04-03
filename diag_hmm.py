"""Diagnostic: Reproduce the HMM SS underflow root cause."""
import json
import numpy as np

# Load the saved stats
with open("regime_online_stats.json") as f:
    data = json.load(f)

N = np.array(data["N"])
S = np.array(data["S"])
SS = np.array(data["SS"])
SX = np.array(data["SX"]) if "SX" in data else None
trans = np.array(data["trans_counts"])
total = data["total_bars_seen"]
bars_since = data["bars_since_update"]

print("=" * 70)
print("HMM ONLINE STATS DIAGNOSTIC")
print("=" * 70)

print(f"\ntotal_bars_seen = {total}")
print(f"bars_since_update = {bars_since}")

# --- N (state weights) ---
print(f"\n--- N (state weights) ---")
total_n = N.sum()
for i in range(3):
    pct = N[i] / total_n * 100
    print(f"  State {i}: N = {N[i]:.4f}  ({pct:.1f}%)")
print(f"  Total: {total_n:.4f}")
print(f"  Max/Min ratio: {N.max()/N.min():.2f}")

# --- SS diagnostic ---
print(f"\n--- SS (sum-of-squares, diag) ---")
for i in range(3):
    print(f"  State {i}: {SS[i]}")
    print(f"    min abs = {np.abs(SS[i]).min():.2e}, max abs = {np.abs(SS[i]).max():.2e}")

# --- SX diagnostic ---
if SX is not None:
    print(f"\n--- SX (outer product sums, full cov) ---")
    for i in range(3):
        diag = np.diag(SX[i])
        print(f"  State {i} diagonal: {diag}")
        print(f"    min abs = {np.abs(diag).min():.2e}")

# --- The Root Cause Analysis ---
print("\n" + "=" * 70)
print("ROOT CAUSE ANALYSIS")
print("=" * 70)

# Prove: exponential decay on SS when state doesn't receive weight
decay = 0.995
print(f"\nDecay factor per bar: {decay}")
print(f"Bars seen: {total}")

# How much does SS decay after N bars if no new weight added?
# SS *= decay^1 each bar. After N bars: SS_original * decay^N
# But also receives += g[:, None] * (x^2) each bar
# If g[state] ≈ 0 (forward pass assigns near-zero responsibility),
# then SS for that state just decays exponentially.

print(f"\nDecay^{total} = {decay**total:.2e}")
print(f"Decay^144 = {decay**144:.2e}")  # ~0.487

# But wait - SS is decayed EACH bar, not once for all bars
# After 144 bars at decay=0.995:
cumulative = 1.0
for b in range(144):
    cumulative *= decay
print(f"Cumulative decay after 144 bars: {cumulative:.6f}")

# So if a state consistently gets gamma ≈ 0, its SS decays to:
# original_SS * 0.995^144 ≈ original_SS * 0.487
# That CAN'T explain 1e-220. Something else is happening.

print(f"\n--- KEY INSIGHT ---")
print(f"0.995^144 = {0.995**144:.6f}")
print(f"This is ~0.49, NOT 1e-220.")
print(f"So pure decay cannot explain SS = 1e-220 after 144 bars.")
print(f"")
print(f"The SS is initialized to ZERO for full covariance mode!")
print(f"Look at _init_online_stats line 354:")
print(f'  SS = np.zeros((n_states, n_features))  # "unused for full"')
print(f"")
print(f"So SS starts at 0.0, and the only update is:")
print(f"  SS += g[:, None] * (x^2)")
print(f"  SS *= batch_decay  (each bar)")
print(f"")

# Simulate: SS starts at 0, gets tiny gamma contributions, then decays
# What happens if gamma for a state is ~0.001 per bar?
print("--- SIMULATION: SS accumulation for a starving state ---")
ss_sim = np.zeros(5)
for bar in range(144):
    ss_sim *= decay  # decay first
    # Assume gamma ≈ 0.001 for the starving state, x^2 ≈ 1.0
    g_state = 0.001 if bar < 100 else 0.2  # gets more weight later
    ss_sim += g_state * np.ones(5)

print(f"  After 144 bars (starving gamma=0.001 then 0.2): SS min = {ss_sim.min():.6f}")
print(f"  This is reasonable, NOT 1e-220")

# The real issue: SS NEVER gets meaningful accumulation because
# the init is 0 AND gamma for state 2 is near-zero AND decay keeps shrinking
# Let's check: what gamma would produce 1e-220?
print(f"\n--- What could produce SS = 6.4e-221? ---")
# If gamma is exactly 0 for 144 bars, SS stays at 0 + floating point noise
# Actually, even np.zeros() * 0.995 is still 0.
# So 6.4e-221 comes from gamma being INCREDIBLY small (like 1e-220 itself)
print(f"  SS = 6.4e-221 means gamma for state 2 was essentially 0")
print(f"  across ALL 144 bars (forward pass never assigned any")
print(f"  responsibility to state 2)")
print(f"")

# Check: is the _enforce_state_floor protecting N but NOT SS?
print("--- THE BUG: _enforce_state_floor doesn't protect SS! ---")
print(f"  _enforce_state_floor transfers N and S and SX/SS proportionally")
print(f"  BUT: with covariance_type='full', it transfers SX, not SS")
print(f"  Line 488-490: if stats.SX is not None -> transfers SX")
print(f"  Line 492-493: else -> transfers SS")
print(f"  So SS is NEVER protected in full-covariance mode!")
print(f"")
print(f"  Meanwhile, SS is still decayed every bar (line 557):")
print(f"    stats.SS *= batch_decay")
print(f"  And only receives tiny additions (line 567):")
print(f"    stats.SS += g * x^2")
print(f"")
print(f"  If gamma for state 2 is ~1e-300, then:")
print(f"    SS += 1e-300 * x^2 (negligible)")
print(f"    SS *= 0.995 (shrinks each bar)")
print(f"    -> SS approaches 0 via exponential decay from tiny seed")

# Actual forward pass check: what does the transition matrix tell us?
print(f"\n--- Transition matrix tells the story ---")
row_sums = trans.sum(axis=1)
for i in range(3):
    probs = trans[i] / row_sums[i]
    print(f"  State {i}: {probs}")
print(f"  State 1->2 transition: {trans[1,2]/row_sums[1]*100:.6f}%")
print(f"  State 2->1 transition: {trans[2,1]/row_sums[2]*100:.6f}%")
print(f"  State 1->2 count: {trans[1,2]:.6f}")
print(f"  State 2->1 count: {trans[2,1]:.2e}")

print(f"\n--- CONCLUSION ---")
print(f"  1. SS is init to 0 in full-cov mode (by design, since SX is used)")
print(f"  2. SS is decayed every bar (line 557) regardless of cov type")
print(f"  3. SS gets += g * x^2, but if gamma->0, this contributes nothing")
print(f"  4. The forward pass is STICKY (trans probs ~97% self-transition)")
print(f"     so once the market enters State 1, it stays there")
print(f"  5. State 2 (choppy) gets gamma->0, so SS->0 via decay")
print(f"  6. _enforce_state_floor saves N and SX but NOT SS")
print(f"  7. Health check only checks SX diag (which IS protected), NOT SS")
print(f"")
print(f"  IMPACT: SS underflow is HARMLESS because re-estimation uses SX.")
print(f"  But the REAL problem is the sticky transition matrix +")
print(f"  forward-only pass = regime detection becomes almost frozen.")
print(f"  Once State 1 dominates, the forward pass keeps assigning")
print(f"  gamma->State 1, which reinforces the dominance (positive feedback).")
