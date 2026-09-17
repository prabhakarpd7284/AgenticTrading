import * as React from "react";
import { Link, useNavigate } from "react-router-dom";
import { toast } from "sonner";
import { AtSign, Lock, ShieldCheck } from "lucide-react";

import { api } from "@/lib/api";
import { useAuthStore } from "@/stores/auth";

import { Button } from "@/components/ui/Button";
import { Input } from "@/components/ui/Input";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/Card";

export function LoginPage() {
  const nav = useNavigate();
  const [email, setEmail] = React.useState("");
  const [password, setPassword] = React.useState("");
  const [err, setErr] = React.useState<string | null>(null);
  const [loading, setLoading] = React.useState(false);

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    setErr(null);
    setLoading(true);
    try {
      const { data } = await api.post("/auth/token/", { email, password });
      useAuthStore.getState().setTokens(data.access, data.refresh);
      useAuthStore.getState().setEmail(email);
      // Backend echoes the resolved tenant in the login response so the SPA
      // can render tenant-scoped content immediately without a second call.
      if (data.tenant?.id) useAuthStore.getState().setTenant(data.tenant.id);
      nav("/dashboard");
    } catch {
      setErr("Email or password doesn't match our records.");
      toast.error("Sign-in failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <AuthFrame>
      <Card>
        <CardHeader>
          <CardTitle>Sign in to AlphaDesk</CardTitle>
          <CardDescription>Welcome back. Your desk is ready.</CardDescription>
        </CardHeader>
        <CardContent>
          <form onSubmit={onSubmit} className="space-y-3" noValidate>
            <Input
              type="email"
              label="Email"
              required
              autoComplete="email"
              leading={<AtSign className="h-4 w-4" />}
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              error={err ?? undefined}
            />
            <Input
              type="password"
              label="Password"
              required
              autoComplete="current-password"
              leading={<Lock className="h-4 w-4" />}
              value={password}
              onChange={(e) => setPassword(e.target.value)}
            />
            <Button type="submit" className="w-full" loading={loading}>
              Sign in
            </Button>
          </form>
        </CardContent>
        <CardFooter className="justify-center">
          <p className="text-caption text-fg-subtle">
            No account?{" "}
            <Link to="/signup" className="text-accent hover:underline underline-offset-4">
              Create one
            </Link>
          </p>
        </CardFooter>
      </Card>
    </AuthFrame>
  );
}

/* Shared frame — exported so SignupPage can reuse it. */
export function AuthFrame({ children }: { children: React.ReactNode }) {
  return (
    <div className="min-h-screen grid grid-cols-1 lg:grid-cols-[1fr_520px] bg-bg">
      <aside className="hidden lg:flex flex-col justify-between p-10 bg-surface border-r border-border">
        <div className="flex items-center gap-2">
          <span
            aria-hidden
            className="inline-flex items-center justify-center h-9 w-9 rounded-md bg-gradient-to-br from-accent to-info text-accent-fg font-display font-bold"
          >
            α
          </span>
          <div>
            <div className="font-display text-body font-semibold text-fg">AlphaDesk</div>
            <div className="text-caption text-fg-subtle -mt-0.5">AI trading desk for Indian markets</div>
          </div>
        </div>

        <div className="max-w-lg space-y-4">
          <h2 className="text-display-lg font-display text-fg tracking-tight">
            An AI desk that <span className="text-accent">thinks with you</span>, not for you.
          </h2>
          <p className="text-body text-fg-muted">
            @DataAnalyst streams NSE/NFO ticks. @DirectionalTrader and @OptionsStrategist plan.
            @RiskGuard vetoes anything that breaches your caps — always, deterministically.
          </p>
          <ul className="text-body-sm text-fg-muted space-y-1.5">
            <li className="flex gap-2"><ShieldCheck className="h-4 w-4 text-accent mt-0.5" aria-hidden /> Row-level tenant isolation</li>
            <li className="flex gap-2"><ShieldCheck className="h-4 w-4 text-accent mt-0.5" aria-hidden /> Secrets rotated daily in AWS Secrets Manager</li>
            <li className="flex gap-2"><ShieldCheck className="h-4 w-4 text-accent mt-0.5" aria-hidden /> Every decision journalled for audit</li>
          </ul>
        </div>

        <p className="text-caption text-fg-subtle">
          © {new Date().getFullYear()} AlphaDesk · Paper mode by default
        </p>
      </aside>

      <main className="flex items-center justify-center p-6">
        <div className="w-full max-w-[420px]">{children}</div>
      </main>
    </div>
  );
}
