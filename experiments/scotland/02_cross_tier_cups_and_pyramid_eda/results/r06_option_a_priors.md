| parameter | family | mean | sd | note |
|---|---|---|---|---|
| d_j (net θ step, each of T1→T2, T2→T3, T3→T4) | TruncatedNormal(μ, σ; 0, ∞) | 0.473 | 0.162 | recommended; linearity not rejected post-2020 once Celtic/Rangers are club effects |
| d_j (net θ step) — weak alternative | HalfNormal(s) | 0.473 | 0.357 | s = 0.593 matches the mean; puts 26% mass below 0.2 |
| attack part of each step, Δα | TruncatedNormal | 0.226 | 0.078 | share 0.48 of τ |
| concession part of each step, Δβ (sign: lower tier concedes more) | TruncatedNormal | 0.247 | 0.085 | share 0.52 of τ |
| Old Firm over rest of T1 (club effect, not a tier) | Normal | 1.15 | 0.085 | do NOT fold into τ_1; give Celtic/Rangers ordinary team effects |
| T4 → non-league step (only if T5 nodes enter) | TruncatedNormal | 0.438 | 0.189 | Highland/Lowland/EoS/WoS pooled; heterogeneous |
| club σ_θ within T1 | Normal(0, σ) around tier mean | 0.0 | 0.269 | σ_α 0.176, σ_β 0.144 |
| club σ_θ within T2 | Normal(0, σ) around tier mean | 0.0 | 0.326 | σ_α 0.174, σ_β 0.187 |
| club σ_θ within T3 | Normal(0, σ) around tier mean | 0.0 | 0.41 | σ_α 0.203, σ_β 0.247 |
| club σ_θ within T4 | Normal(0, σ) around tier mean | 0.0 | 0.23 | σ_α 0.171, σ_β 0.133 |
| δ_league T1 (A1, zero-sum) | point / Normal(δ, 0.05) | -0.045 | 0.05 | tier goal level is flat across the SPFL (|δ| ≤ 0.1) |
| δ_league T2 (A1, zero-sum) | point / Normal(δ, 0.05) | -0.044 | 0.05 | tier goal level is flat across the SPFL (|δ| ≤ 0.1) |
| δ_league T3 (A1, zero-sum) | point / Normal(δ, 0.05) | 0.069 | 0.05 | tier goal level is flat across the SPFL (|δ| ≤ 0.1) |
| δ_league T4 (A1, zero-sum) | point / Normal(δ, 0.05) | 0.02 | 0.05 | tier goal level is flat across the SPFL (|δ| ≤ 0.1) |
