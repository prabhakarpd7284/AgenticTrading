### 1a. Open vs PP: persistence against a mechanical null

`close` = official bhavcopy close (as in the Pivot Boss study). *Null* = the rate expected if each day's open→close return were drawn at random from the same stock's IS returns (i.e. pure "the open is already on that side").

| condition | n | close same side of PP (actual) | mechanical null | excess | close beyond open, same direction | unconditional close > open |
|---|---|---|---|---|---|---|
| open > PP | 2244 | 66.6% | 67.4% | -0.8 pp | 45.1% | 47.0% (close>open, all days) |
| open < PP | 2242 | 68.0% | 69.6% | -1.6 pp | 50.4% | 53.0% (close<open, all days) |

### 1b. Open side of PP as a trade (side = +1 above PP / −1 below; signed returns)

*cont %* = share of days the signed return is > 0. *mean/median* = signed return in %. Net = signed return − 0.04 % slippage − NSE costs at that notional (no stop, 09:16 entry → 15:14 close). *t* clusters the 49 stocks by date (one mean per day).

| condition | n | days | cont % | mean % | median % | t (day-clustered) | net % @₹50k | net % @₹5L | net win % @₹50k |
|---|---|---|---|---|---|---|---|---|---|
| all days, LONG every stock (drift baseline), 09:16→15:14 | 4494 | 92 | 47.1% | -0.037 | -0.069 | -0.69 | -0.209 | -0.122 | 40.1% |
| open > PP → long, bhav O→C (descriptive) | 2244 | 92 | 45.1% | -0.032 | -0.137 | -0.89 | -0.205 | -0.117 | 39.6% |
| open > PP → long, 09:15 open→15:14 | 2244 | 92 | 43.6% | -0.080 | -0.172 | -1.46 | -0.253 | -0.165 | 37.4% |
| open > PP → long, **09:16 open→15:14 (tradeable)** | 2244 | 92 | 46.7% | -0.025 | -0.080 | -0.76 | -0.197 | -0.109 | 39.7% |
| open < PP → short, bhav O→C (descriptive) | 2242 | 92 | 50.4% | -0.005 | +0.019 | +0.16 | -0.178 | -0.090 | 44.7% |
| open < PP → short, 09:15 open→15:14 | 2242 | 92 | 51.8% | +0.028 | +0.037 | +0.80 | -0.145 | -0.057 | 45.2% |
| open < PP → short, **09:16 open→15:14 (tradeable)** | 2242 | 92 | 52.1% | +0.048 | +0.055 | +1.38 | -0.124 | -0.037 | 45.3% |
| both sides, bhav O→C (descriptive) | 4486 | 92 | 47.7% | -0.019 | -0.064 | -0.55 | -0.191 | -0.103 | 42.1% |
| both sides, 09:15 open→15:14 | 4486 | 92 | 47.7% | -0.026 | -0.056 | -0.78 | -0.199 | -0.111 | 41.3% |
| both sides, **09:16 open→15:14 (tradeable)** | 4486 | 92 | 49.4% | +0.012 | -0.015 | +0.36 | -0.161 | -0.073 | 42.5% |
| both sides, 09:16→15:14, **market-neutral** (minus same-day mean of all 49) | 4486 | 92 | 49.4% | +0.017 | -0.012 | +0.89 | -0.156 | -0.068 | 42.0% |

### 1c. Open side of PP by open distance from PP, % of PP (09:16→15:14, signed by PP side)

| condition | n | days | cont % | mean % | median % | t (day-clustered) | net % @₹50k | net % @₹5L | net win % @₹50k |
|---|---|---|---|---|---|---|---|---|---|
| long (open > PP), 0–0.25 | 609 | 91 | 49.9% | +0.025 | +0.000 | +0.33 | -0.147 | -0.060 | 42.0% |
| long (open > PP), 0.25–0.5 | 507 | 88 | 47.9% | +0.041 | -0.059 | -0.98 | -0.132 | -0.044 | 38.7% |
| long (open > PP), 0.5–1 | 631 | 88 | 44.4% | -0.083 | -0.126 | -2.01 | -0.256 | -0.168 | 38.2% |
| long (open > PP), 1–2 | 406 | 75 | 43.8% | -0.097 | -0.140 | +0.33 | -0.269 | -0.182 | 38.7% |
| long (open > PP), ≥ 2 | 91 | 38 | 47.3% | +0.009 | -0.073 | -0.09 | -0.163 | -0.075 | 44.0% |
| short (open < PP), 0–0.25 | 673 | 91 | 55.0% | +0.075 | +0.100 | +1.04 | -0.097 | -0.010 | 44.7% |
| short (open < PP), 0.25–0.5 | 521 | 87 | 54.3% | +0.093 | +0.140 | +2.04 | -0.079 | +0.008 | 48.9% |
| short (open < PP), 0.5–1 | 636 | 86 | 52.4% | +0.120 | +0.078 | +1.62 | -0.052 | +0.035 | 46.2% |
| short (open < PP), 1–2 | 340 | 61 | 43.8% | -0.146 | -0.139 | +0.30 | -0.318 | -0.231 | 39.4% |
| short (open < PP), ≥ 2 | 72 | 32 | 45.8% | -0.257 | -0.220 | -0.10 | -0.428 | -0.342 | 43.1% |

### 1c. Open side of PP by open distance from PP, daily ATR14 units (09:16→15:14, signed by PP side)

| condition | n | days | cont % | mean % | median % | t (day-clustered) | net % @₹50k | net % @₹5L | net win % @₹50k |
|---|---|---|---|---|---|---|---|---|---|
| long (open > PP), 0–0.25 | 1031 | 78 | 48.1% | -0.008 | -0.034 | -0.46 | -0.181 | -0.093 | 39.8% |
| long (open > PP), 0.25–0.5 | 556 | 74 | 43.9% | -0.083 | -0.123 | -1.29 | -0.256 | -0.168 | 37.2% |
| long (open > PP), 0.5–1 | 285 | 61 | 42.1% | -0.144 | -0.147 | -1.25 | -0.316 | -0.229 | 36.8% |
| long (open > PP), ≥ 1 | 47 | 23 | 48.9% | +0.108 | -0.122 | +0.39 | -0.064 | +0.023 | 46.8% |
| short (open < PP), 0–0.25 | 1096 | 78 | 55.6% | +0.121 | +0.116 | +1.74 | -0.051 | +0.037 | 47.5% |
| short (open < PP), 0.25–0.5 | 535 | 73 | 49.9% | +0.016 | -0.021 | +0.13 | -0.156 | -0.068 | 43.9% |
| short (open < PP), 0.5–1 | 213 | 47 | 42.7% | -0.130 | -0.187 | +0.49 | -0.302 | -0.215 | 36.6% |
| short (open < PP), ≥ 1 | 39 | 21 | 35.9% | -0.746 | -0.822 | -0.49 | -0.918 | -0.831 | 33.3% |

### 1c. Open side of PP by gap (09:15 open vs prior close), % (09:16→15:14, signed by PP side)

| condition | n | days | cont % | mean % | median % | t (day-clustered) | net % @₹50k | net % @₹5L | net win % @₹50k |
|---|---|---|---|---|---|---|---|---|---|
| long (open > PP), |gap| 0–0.25, gap with PP side | 399 | 81 | 44.4% | -0.014 | -0.102 | +0.39 | -0.187 | -0.099 | 36.8% |
| long (open > PP), |gap| 0–0.25, gap against PP side | 245 | 71 | 48.2% | -0.067 | -0.038 | -1.13 | -0.239 | -0.151 | 39.6% |
| long (open > PP), |gap| 0.25–0.5, gap with PP side | 445 | 81 | 49.4% | +0.021 | -0.030 | -0.91 | -0.151 | -0.063 | 39.6% |
| long (open > PP), |gap| 0.25–0.5, gap against PP side | 84 | 43 | 44.0% | -0.150 | -0.148 | -0.50 | -0.322 | -0.235 | 32.1% |
| long (open > PP), |gap| 0.5–1, gap with PP side | 540 | 79 | 42.6% | -0.113 | -0.157 | -1.86 | -0.285 | -0.198 | 38.0% |
| long (open > PP), |gap| 0.5–1, gap against PP side | 30 | 19 | 50.0% | +0.364 | +0.039 | +2.33 | +0.192 | +0.279 | 46.7% |
| long (open > PP), |gap| 1–2, gap with PP side | 272 | 61 | 51.5% | +0.093 | +0.046 | +0.42 | -0.080 | +0.008 | 47.4% |
| long (open > PP), |gap| 1–2, gap against PP side | 6 * | 4 | 16.7% | -0.737 | -0.814 | -2.17 | -0.911 | -0.821 | 16.7% |
| long (open > PP), |gap| ≥ 2, gap with PP side | 50 | 31 | 50.0% | +0.072 | -0.037 | +0.48 | -0.100 | -0.013 | 48.0% |
| long (open > PP), |gap| ≥ 2, gap against PP side | 0 | | | | | | | | |
| short (open < PP), |gap| 0–0.25, gap with PP side | 415 | 84 | 58.8% | +0.146 | +0.179 | +1.99 | -0.027 | +0.061 | 50.6% |
| short (open < PP), |gap| 0–0.25, gap against PP side | 338 | 84 | 52.7% | +0.052 | +0.104 | +1.23 | -0.120 | -0.032 | 45.6% |
| short (open < PP), |gap| 0.25–0.5, gap with PP side | 356 | 78 | 51.7% | +0.048 | +0.044 | +1.31 | -0.125 | -0.037 | 44.9% |
| short (open < PP), |gap| 0.25–0.5, gap against PP side | 149 | 55 | 53.0% | +0.113 | +0.116 | +0.20 | -0.059 | +0.029 | 47.7% |
| short (open < PP), |gap| 0.5–1, gap with PP side | 451 | 76 | 47.7% | -0.007 | -0.080 | +1.44 | -0.179 | -0.092 | 41.5% |
| short (open < PP), |gap| 0.5–1, gap against PP side | 39 | 30 | 46.2% | -0.085 | -0.120 | -0.58 | -0.257 | -0.170 | 38.5% |
| short (open < PP), |gap| 1–2, gap with PP side | 229 | 57 | 48.0% | -0.015 | -0.032 | +0.34 | -0.187 | -0.100 | 43.2% |
| short (open < PP), |gap| 1–2, gap against PP side | 2 * | 2 | 50.0% | -1.096 | -1.096 | — | -1.266 | -1.180 | 50.0% |
| short (open < PP), |gap| ≥ 2, gap with PP side | 43 | 24 | 39.5% | -0.247 | -0.491 | -0.44 | -0.418 | -0.331 | 37.2% |
| short (open < PP), |gap| ≥ 2, gap against PP side | 0 | | | | | | | | |

### 1d. Intraday versions (entry at the next 1m open, exit 15:14, signed)

| condition | n | days | cont % | mean % | median % | t (day-clustered) | net % @₹50k | net % @₹5L | net win % @₹50k |
|---|---|---|---|---|---|---|---|---|---|
| first-15m close (09:29) vs PP → enter 09:30 | 4491 | 92 | 48.5% | -0.015 | -0.028 | -0.48 | -0.187 | -0.099 | 41.0% |
| first-30m close (09:44) vs PP → enter 09:45 | 4494 | 92 | 46.9% | -0.049 | -0.067 | -1.54 | -0.221 | -0.134 | 39.6% |
| PP untouched 09:15–10:14, above PP → enter 10:15 | 1001 | 89 | 45.0% | -0.048 | -0.099 | -1.01 | -0.221 | -0.133 | 38.0% |
| PP untouched 09:15–10:14, below PP → enter 10:15 | 990 | 89 | 50.8% | -0.010 | +0.018 | +1.53 | -0.183 | -0.095 | 42.6% |
| PP untouched by 10:15, both sides → enter 10:15 | 1991 | 92 | 47.9% | -0.029 | -0.038 | -0.41 | -0.202 | -0.114 | 40.3% |
| PP touched by 10:15 (reference; side = open vs PP) → 10:15 | 2495 | 92 | 52.1% | +0.029 | +0.048 | +0.72 | -0.143 | -0.055 | 43.1% |
| PP untouched by 10:15, **market-neutral** | 1991 | 92 | 49.4% | +0.001 | -0.013 | -0.11 | -0.171 | -0.083 | 40.0% |
| CPR: open above TC → long, 09:16 | 1969 | 89 | 45.9% | -0.044 | -0.104 | -1.40 | -0.217 | -0.129 | 39.0% |
| CPR: open below BC → short, 09:16 | 1903 | 92 | 51.8% | +0.035 | +0.049 | +1.32 | -0.137 | -0.050 | 45.1% |
| CPR: open inside CPR (long, reference), 09:16 | 622 | 92 | 48.9% | -0.019 | -0.032 | +0.40 | -0.191 | -0.104 | 42.0% |

### 1e. Stability inside IS (Apr 20 – Jun 30 vs Jul 1 – Aug 31), 09:16/10:15 → 15:14

| condition | n | days | cont % | mean % | median % | t (day-clustered) | net % @₹50k | net % @₹5L | net win % @₹50k |
|---|---|---|---|---|---|---|---|---|---|
| Apr–Jun: open side of PP | 2347 | 48 | 49.2% | +0.011 | -0.023 | +0.20 | -0.162 | -0.074 | 43.5% |
| Apr–Jun: PP untouched by 10:15 | 1129 | 48 | 47.9% | -0.044 | -0.038 | -0.37 | -0.216 | -0.128 | 40.8% |
| Apr–Jun: long every stock (drift) | 2352 | 48 | 47.3% | -0.067 | -0.078 | -0.78 | -0.239 | -0.151 | 41.1% |
| Jul–Aug: open side of PP | 2139 | 44 | 49.6% | +0.013 | -0.000 | +0.32 | -0.160 | -0.072 | 41.3% |
| Jul–Aug: PP untouched by 10:15 | 862 | 44 | 47.8% | -0.011 | -0.040 | -0.19 | -0.183 | -0.096 | 39.6% |
| Jul–Aug: long every stock (drift) | 2142 | 44 | 47.0% | -0.004 | -0.060 | -0.04 | -0.177 | -0.089 | 39.1% |

### 1f. Gap fill (prior close touched), IS

| |gap| % | n | filled 09:15–10:14 | filled after 09:16 by 10:14 | filled by 15:14 | mean |gap| % |
|---|---|---|---|---|---|
| 0–0.25 | 1400 | 81.4% | 61.5% | 86.8% | 0.11 |
| 0.25–0.5 | 1035 | 62.0% | 51.1% | 73.5% | 0.38 |
| 0.5–1 | 1062 | 47.3% | 41.8% | 64.0% | 0.70 |
| 1–2 | 509 | 21.6% | 20.2% | 37.9% | 1.33 |
| ≥ 2 | 93 | 14.0% | 12.9% | 23.7% | 3.04 |
