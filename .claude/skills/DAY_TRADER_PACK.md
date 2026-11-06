# Day Trader skill pack (vendored)

The eleven `trader-*` folders in `.claude/skills/` are copied unchanged from
[Jakeschincariol/day-trader-skill](https://github.com/Jakeschincariol/day-trader-skill)
at commit `3d0f4ef` (2026-10-20), MIT licence (full text below). Upstream's
offline suite (315 tests) passed on that commit before vendoring; the tests and
fixtures were not copied.

## What it is, and how it relates to AlphaDesk

- **US markets only.** Data comes from Yahoo Finance (unofficial chart API),
  Nasdaq Trader symbol lists, SEC EDGAR and the House Clerk. NYSE hours,
  New York time. Nothing here reads NSE data or Angel One.
- **Separate from AlphaDesk's trading path.** The tools write to `trader/` in
  the current directory (gitignored), not to Postgres. They do not go through
  @RiskGuard, `event_writer.emit()` or `TRADING_MODE`. Treat them as a research
  and ideas pack, for example for the US side of the Screens Lab plan.
- **No orders from code.** `trader-broker/ticket.py` never contacts a broker.
  An order needs Alpaca's MCP server in Claude Desktop (paper by default), a
  typed yes and an "Allow once" click. Keep it on paper. Alpaca is a US broker.
- SEC tools need `SEC_USER_AGENT="Your Name you@example.com"` (or
  `trader/config.json` `sec_user_agent`). Python 3.8+, standard library only.

Skills: `trader-scanner`, `trader-news`, `trader-entry`, `trader-exit`,
`trader-timemachine`, `trader-risk`, `trader-budget`, `trader-broker`,
`trader-alerts`, `trader-analyst`, `trader-pelosi`. Not financial advice.

## Licence

MIT License

Copyright (c) 2026 Jake Schincariol

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
