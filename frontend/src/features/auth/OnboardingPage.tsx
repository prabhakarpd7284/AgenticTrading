import * as React from "react";
import { useNavigate } from "react-router-dom";
import {
  ArrowRight, Bot, Briefcase, CheckCircle2, ShieldCheck, Sparkles, Wallet,
} from "lucide-react";

import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { Badge } from "@/components/ui/Badge";
import { cn } from "@/lib/utils";

type Step = 0 | 1 | 2 | 3;

const steps: { icon: any; title: string; body: string }[] = [
  {
    icon: Wallet,
    title: "Start with paper capital",
    body: "We fund your desk with ₹5,00,000 virtual capital. Every fill is simulated against live prices — so you can calibrate with zero risk.",
  },
  {
    icon: Bot,
    title: "Meet your AI desk",
    body: "@DataAnalyst fetches live data, @DirectionalTrader and @OptionsStrategist propose trades, @RiskGuard validates, @PortfolioTracker keeps score.",
  },
  {
    icon: ShieldCheck,
    title: "RiskGuard is non-negotiable",
    body: "Every plan passes a deterministic 9-criteria check before execution — daily-loss caps, position-size caps, kill-switch. No LLM can bypass it.",
  },
  {
    icon: Sparkles,
    title: "Graduate to live when ready",
    body: "Once you're comfortable, link your Angel One account. The same strategies route to real orders — with the same guardrails.",
  },
];

export function OnboardingPage() {
  const nav = useNavigate();
  const [step, setStep] = React.useState<Step>(0);
  const S = steps[step];

  return (
    <div className="min-h-screen grid place-items-center p-6 bg-bg">
      <div className="w-full max-w-[720px] space-y-5">
        {/* Top brand + stepper */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span
              aria-hidden
              className="inline-flex items-center justify-center h-8 w-8 rounded-md bg-gradient-to-br from-accent to-info text-accent-fg font-display font-bold"
            >
              α
            </span>
            <div>
              <div className="font-display text-body font-semibold text-fg">AlphaDesk</div>
              <div className="text-caption text-fg-subtle -mt-0.5">Your AI trading desk</div>
            </div>
          </div>
          <Badge tone="brand" dot>Paper mode</Badge>
        </div>

        {/* Step card */}
        <Card className="overflow-hidden">
          <CardHeader>
            <div className="flex items-start gap-3">
              <div className="inline-flex h-10 w-10 items-center justify-center rounded-sm bg-accent/15 text-accent shrink-0">
                <S.icon className="h-5 w-5" aria-hidden />
              </div>
              <div>
                <CardTitle>{S.title}</CardTitle>
                <CardDescription>Step {step + 1} of {steps.length}</CardDescription>
              </div>
            </div>
          </CardHeader>
          <CardContent>
            <p className="text-body text-fg-muted max-w-prose">{S.body}</p>

            {/* Role cards on step 1 */}
            {step === 1 && (
              <ul className="mt-5 grid grid-cols-1 sm:grid-cols-2 gap-2">
                {[
                  { role: "@DataAnalyst",       desc: "Live NSE/NFO data, no LLM" },
                  { role: "@DirectionalTrader", desc: "Plans BUY/SELL equity" },
                  { role: "@OptionsStrategist", desc: "Manages straddle lifecycle" },
                  { role: "@RiskGuard",         desc: "Deterministic last gate" },
                ].map((r) => (
                  <li key={r.role} className="rounded-sm border border-border bg-surface-2 p-3">
                    <div className="text-body-sm font-medium text-fg">{r.role}</div>
                    <div className="text-caption text-fg-subtle">{r.desc}</div>
                  </li>
                ))}
              </ul>
            )}

            {/* Risk checklist on step 2 */}
            {step === 2 && (
              <ul className="mt-5 space-y-1.5">
                {[
                  "Max risk / trade: 1% of capital",
                  "Max daily loss: 3% of capital",
                  "Max position size: 10% of capital",
                  "Straddle hard stop if combined premium > sold value",
                  "Expiry-day auto-close before 3:15 PM IST",
                ].map((c) => (
                  <li key={c} className="flex items-center gap-2 text-body-sm text-fg-muted">
                    <CheckCircle2 className="h-4 w-4 text-pnl-up shrink-0" aria-hidden /> {c}
                  </li>
                ))}
              </ul>
            )}
          </CardContent>
          <CardFooter className="justify-between">
            <StepDots count={steps.length} current={step} />
            <div className="flex items-center gap-2">
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setStep(Math.max(0, step - 1) as Step)}
                disabled={step === 0}
              >
                Back
              </Button>
              {step < steps.length - 1 ? (
                <Button
                  size="sm"
                  onClick={() => setStep(Math.min(steps.length - 1, step + 1) as Step)}
                  trailing={<ArrowRight className="h-4 w-4" />}
                >
                  Next
                </Button>
              ) : (
                <div className="flex items-center gap-2">
                  <Button variant="secondary" size="sm" onClick={() => nav("/brokers")} leading={<Briefcase className="h-4 w-4" />}>
                    Link broker
                  </Button>
                  <Button size="sm" onClick={() => nav("/dashboard")} trailing={<ArrowRight className="h-4 w-4" />}>
                    Enter desk
                  </Button>
                </div>
              )}
            </div>
          </CardFooter>
        </Card>

        <p className="text-center text-caption text-fg-subtle">
          You can change trading mode and risk caps at any time from Settings.
        </p>
      </div>
    </div>
  );
}

function StepDots({ count, current }: { count: number; current: number }) {
  return (
    <ol className="flex items-center gap-1.5" aria-label="Progress">
      {Array.from({ length: count }).map((_, i) => (
        <li
          key={i}
          className={cn(
            "h-1.5 rounded-full transition-all duration-240",
            i === current ? "w-6 bg-accent" : i < current ? "w-3 bg-accent/50" : "w-3 bg-border",
          )}
          aria-current={i === current ? "step" : undefined}
        />
      ))}
    </ol>
  );
}
