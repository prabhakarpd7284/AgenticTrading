import { useNavigate, useRouteError } from "react-router-dom";

import { ErrorState } from "@/components/ui/ErrorState";

/**
 * Router errorElement — catches any error thrown while rendering a route
 * subtree (e.g. an `undefined.map()` from a shape mismatch) so it degrades to a
 * recoverable panel instead of escaping to the root and blanking the entire SPA.
 */
export function RouteError() {
  const error = useRouteError();
  const navigate = useNavigate();
  // Surface for triage. The frontend has no Sentry wired yet; this is the
  // single report point to attach one to later.
  // eslint-disable-next-line no-console
  console.error("Unhandled route render error:", error);

  return (
    <div className="flex min-h-screen items-center justify-center p-6">
      <ErrorState
        title="This page hit an error."
        description="Something went wrong rendering this view. Reload to try again, or navigate away — the rest of the app is unaffected."
        onRetry={() => navigate(0)}
      />
    </div>
  );
}
