import * as React from "react";
import { Link, useNavigate } from "react-router-dom";
import { toast } from "sonner";
import { AtSign, Lock, User } from "lucide-react";

import { api } from "@/lib/api";

import { Button } from "@/components/ui/Button";
import { Input } from "@/components/ui/Input";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/Card";
import { AuthFrame } from "./LoginPage";

export function SignupPage() {
  const nav = useNavigate();
  const [form, setForm] = React.useState({ email: "", full_name: "", password: "" });
  const [errors, setErrors] = React.useState<Partial<Record<keyof typeof form, string>>>({});
  const [loading, setLoading] = React.useState(false);

  function validate() {
    const e: typeof errors = {};
    if (!form.full_name.trim()) e.full_name = "Add your name so the desk knows who's at the helm.";
    if (!/^\S+@\S+\.\S+$/.test(form.email)) e.email = "That doesn't look like a valid email.";
    if (form.password.length < 12) e.password = "Use at least 12 characters — we recommend a passphrase.";
    setErrors(e);
    return Object.keys(e).length === 0;
  }

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!validate()) return;
    setLoading(true);
    try {
      await api.post("/auth/auth/signup/", form);
      toast.success("Account created — sign in to continue");
      nav("/login");
    } catch {
      toast.error("Could not create account");
    } finally {
      setLoading(false);
    }
  }

  return (
    <AuthFrame>
      <Card>
        <CardHeader>
          <CardTitle>Create your account</CardTitle>
          <CardDescription>14-day free trial · Paper mode by default · No card required</CardDescription>
        </CardHeader>
        <CardContent>
          <form onSubmit={onSubmit} className="space-y-3" noValidate>
            <Input
              label="Full name"
              required
              autoComplete="name"
              leading={<User className="h-4 w-4" />}
              value={form.full_name}
              onChange={(e) => setForm({ ...form, full_name: e.target.value })}
              error={errors.full_name}
            />
            <Input
              type="email"
              label="Email"
              required
              autoComplete="email"
              leading={<AtSign className="h-4 w-4" />}
              value={form.email}
              onChange={(e) => setForm({ ...form, email: e.target.value })}
              error={errors.email}
            />
            <Input
              type="password"
              label="Password"
              required
              autoComplete="new-password"
              leading={<Lock className="h-4 w-4" />}
              value={form.password}
              onChange={(e) => setForm({ ...form, password: e.target.value })}
              hint={errors.password ? undefined : "12+ characters. Avoid reused passwords."}
              error={errors.password}
            />
            <Button type="submit" className="w-full" loading={loading}>
              Create account
            </Button>
          </form>
        </CardContent>
        <CardFooter className="justify-center">
          <p className="text-caption text-fg-subtle">
            Already have one?{" "}
            <Link to="/login" className="text-accent hover:underline underline-offset-4">
              Sign in
            </Link>
          </p>
        </CardFooter>
      </Card>
    </AuthFrame>
  );
}
