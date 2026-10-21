import { FlaskConical } from "lucide-react";
import { EmptyState } from "@/components/ui/EmptyState";

/**
 * /lab — placeholder (UX redesign plan §7). The Analysis Lab — a saved
 * stock group × several strategies × a period, as a matrix — ships in R5,
 * once the Strategy Studio's real backtest service exists (R4).
 */
export function LabPage() {
  return (
    <div className="p-6 max-w-3xl">
      <EmptyState
        icon={<FlaskConical />}
        title="Lab — coming in R5"
        description={
          "Pick a group of stocks, a few strategies and a period, and see what works where: a " +
          "stock × strategy matrix with in-sample vs out-of-sample results and data provenance. " +
          "It needs the real backtest service from the Strategy Studio first."
        }
      />
    </div>
  );
}
