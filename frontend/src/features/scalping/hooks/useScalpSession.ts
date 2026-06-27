/**
 * Streaming engine for the scalp simulator.
 *
 * Three tiers keep high-frequency ticks from thrashing React:
 *   A. imperative ref → chart (updateBar/setMarkers/setSL) — every frame, no render
 *   B. throttled useState (≈6/s flush) → position / kpis / pressure / log / decisions
 *   C. plain useState → status / connState / speed (low frequency)
 */
import * as React from "react";

import { connect, type WSMessage } from "@/lib/ws";
import { epochToTime, type LiveCandleHandle } from "@/components/charts/LiveCandleChart";
import type {
  Annotation, ConnState, DecisionMsg, Kpis, PositionState, PressureState, SessionProfile, Status,
} from "../scalping.types";

const LOG_CAP = 600;
const UP = "#10b981", DOWN = "#f43f5e", WARN = "#f59e0b";

interface Marker { time: number; position: "aboveBar" | "belowBar"; color: string; shape: "arrowUp" | "arrowDown" | "circle" | "square"; text: string; }

export function useScalpSession() {
  const chartRef = React.useRef<LiveCandleHandle>(null);

  // Tier C
  const [status, setStatus] = React.useState<Status>("idle");
  const [connState, setConnState] = React.useState<ConnState>("connecting");
  const [speed, setSpeedState] = React.useState(4);
  const [source, setSource] = React.useState<string>("");
  const [info, setInfo] = React.useState<{ date?: string; expiry?: string }>({});

  // Tier B (flushed)
  const [position, setPosition] = React.useState<PositionState | null>(null);
  const [pressure, setPressure] = React.useState<PressureState | null>(null);
  const [profile, setProfile] = React.useState<SessionProfile | null>(null);
  const [kpis, setKpis] = React.useState<Kpis | null>(null);
  const [log, setLog] = React.useState<string[]>([]);
  const [decisions, setDecisions] = React.useState<DecisionMsg[]>([]);
  const [annotations, setAnnotations] = React.useState<Annotation[]>([]);

  // Tier A refs
  const wsRef = React.useRef<ReturnType<typeof connect> | null>(null);
  const connIdRef = React.useRef(0);        // bumped per connect() — stale socket callbacks bail
  const startedRef = React.useRef(false);   // sent {op:start} once — a reconnect must NOT restart the sim
  const endedRef = React.useRef(false);     // done/stopped — ignore any late/buffered frames
  const markersRef = React.useRef<Marker[]>([]);
  const decisionSecsRef = React.useRef(600);
  // trail-SL line is drawn at CANDLE-bucket times (never per-tick) so it shares
  // the candle series' time slots and can't distort the chart.
  const slRef = React.useRef(0);
  const sideRef = React.useRef("FLAT");
  const lotSizeRef = React.useRef(0);   // captured from position frames → live per-tick INR P&L
  const lastTrailBucketRef = React.useRef<number | null>(null);
  const lastCandleEpochRef = React.useRef(0);   // session-anchored bucket of the current candle
  const pendingRef = React.useRef<{
    position?: PositionState; pressure?: PressureState; profile?: SessionProfile;
    logs: string[]; decisions: DecisionMsg[]; dirty: boolean;
  }>({ logs: [], decisions: [], dirty: false });

  // Tier B flush loop
  React.useEffect(() => {
    const id = window.setInterval(() => {
      const p = pendingRef.current;
      if (!p.dirty) return;
      if (p.position) setPosition(p.position);
      if (p.pressure) setPressure(p.pressure);
      if (p.profile) setProfile(p.profile);
      if (p.logs.length) {
        const batch = p.logs; p.logs = [];
        setLog((prev) => [...prev, ...batch].slice(-LOG_CAP));
      }
      if (p.decisions.length) {
        const batch = p.decisions; p.decisions = [];
        setDecisions((prev) => [...prev, ...batch].slice(-LOG_CAP));
      }
      p.dirty = false;
    }, 160);
    return () => window.clearInterval(id);
  }, []);

  const onMessage = React.useCallback((msg: WSMessage) => {
    if (endedRef.current) return;   // sim is over — drop any late/buffered/duplicate frames
    const p = pendingRef.current;
    switch (msg.type) {
      case "meta":
        decisionSecsRef.current = (msg.decision_secs as number) || 600;
        setSource((msg.symbol as string) ?? "");
        break;
      case "started":
        setStatus("running");
        setSource((msg.source as string) ?? "");
        setInfo({ date: msg.date as string, expiry: msg.expiry as string });
        break;
      case "tick":
        if (p.position) {
          // Recompute Unrealized from the live tick instead of waiting for the
          // next (throttled) `position` frame — otherwise P&L visibly lags the
          // price on fast moves. pts = dir·(ltp−avg)·lots; INR via captured lot size.
          const pos = p.position;
          const ltp = msg.ltp as number;
          const dir = pos.side === "LONG" ? 1 : pos.side === "SHORT" ? -1 : 0;
          const pts = dir * (ltp - pos.avg) * pos.lots;
          p.position = {
            ...pos,
            ltp,
            unrealized_pts: pts,
            unrealized_inr: lotSizeRef.current ? pts * lotSizeRef.current : pos.unrealized_inr,
          };
          p.dirty = true;
        }
        break;
      case "candle": {
        const epoch = msg.epoch as number;
        lastCandleEpochRef.current = epoch;
        chartRef.current?.updateBar({
          time: epochToTime(epoch),
          open: msg.o as number, high: msg.h as number, low: msg.l as number, close: msg.c as number,
        });
        // draw the trail SL at this candle's time (aligned to the candle series)
        if (sideRef.current !== "FLAT" && slRef.current > 0) {
          chartRef.current?.updateTrailSL(epochToTime(epoch), slRef.current);
          lastTrailBucketRef.current = epoch;
        } else if (lastTrailBucketRef.current != null && epoch > lastTrailBucketRef.current) {
          chartRef.current?.breakTrailSL(epochToTime(epoch));  // gap once when flat resumes
          lastTrailBucketRef.current = null;
        }
        break;
      }
      case "pressure":
        p.pressure = {
          pressure: msg.pressure as number, poc: msg.poc as number, vah: msg.vah as number,
          val: msg.val as number, bias: (msg.bias as string) ?? "neutral",
          bins: (msg.bins as PressureState["bins"]) ?? [],
        };
        p.dirty = true;
        break;
      case "position":
        p.position = msg as unknown as PositionState;
        // Capture lot size (inr/pts) so per-tick recompute can render live INR.
        if ((msg.unrealized_pts as number)) {
          lotSizeRef.current = (msg.unrealized_inr as number) / (msg.unrealized_pts as number);
        }
        p.dirty = true;
        slRef.current = (msg.sl as number) || 0;
        sideRef.current = (msg.side as string) || "FLAT";
        chartRef.current?.setSL((msg.sl as number) || null);  // current-level line + axis label
        break;
      case "profile":
        p.profile = msg as unknown as SessionProfile;
        p.dirty = true;
        break;
      case "decision": {
        const d = msg as unknown as DecisionMsg;
        p.decisions.push(d); p.dirty = true;
        pushMarker(markersRef, lastCandleEpochRef.current, d);   // snap to the current candle bar
        chartRef.current?.setMarkers(markersRef.current as never);
        break;
      }
      case "exit":
        // flat now — the next candle frame draws the gap (keeps the exit candle's SL)
        sideRef.current = "FLAT";
        slRef.current = 0;
        chartRef.current?.setSL(null);
        break;
      case "log":
        p.logs.push(msg.line as string); p.dirty = true;
        break;
      case "order": {
        const verb = msg.mode === "placed" ? "ORDER" : "RISK";
        const ok = msg.approved ? "✓" : "✗";
        p.logs.push(`[${verb} ${ok}] ${msg.side} ${msg.qty} @ ${msg.price} SL ${msg.sl} TP ${msg.tp} — ${msg.reason}`);
        p.dirty = true;
        break;
      }
      case "warn":
        p.logs.push(`⚠ ${msg.detail as string}`); p.dirty = true;
        break;
      case "done":
        endedRef.current = true;
        setStatus("done");
        setConnState("live");          // not a disconnect — keep the pill calm
        setKpis(msg.kpis as Kpis);
        wsRef.current?.close();
        break;
      case "error":
        endedRef.current = true;
        setStatus("error");
        p.logs.push(`✗ ${msg.detail as string}`); p.dirty = true;
        wsRef.current?.close();
        break;
    }
  }, []);

  const send = (m: WSMessage) => wsRef.current?.send(m);

  const start = React.useCallback((runId: string) => {
    // reset visual + connection state for a fresh run
    markersRef.current = [];
    slRef.current = 0;
    sideRef.current = "FLAT";
    lastTrailBucketRef.current = null;
    lastCandleEpochRef.current = 0;
    startedRef.current = false;
    endedRef.current = false;
    pendingRef.current = { logs: [], decisions: [], dirty: false };
    chartRef.current?.reset();
    setLog([]); setDecisions([]); setPosition(null); setPressure(null); setProfile(null); setKpis(null);
    setStatus("connecting");
    wsRef.current?.close();
    // Tag this connection. The previous socket's onClose closure shares these
    // refs; a stale close landing after the new onOpen would otherwise take the
    // "dropped mid-sim" branch and flip this fresh session to error.
    const myConnId = ++connIdRef.current;
    wsRef.current = connect(`/ws/scalp/${runId}/`, onMessage, {
      onOpen: () => {
        if (connIdRef.current !== myConnId) return;   // superseded by a newer connect
        setConnState("live");
        // Start the replay exactly once. A reconnect must NOT re-send start —
        // that would replay the whole sim from scratch ("keeps running").
        if (!startedRef.current) { startedRef.current = true; send({ op: "start" }); }
      },
      onClose: (ev) => {
        if (connIdRef.current !== myConnId) return;   // stale close from an old socket
        if (endedRef.current) return;   // clean finish — leave the pill as-is
        if (ev.code === 4401 || ev.code === 4403) { setConnState("closed_auth"); return; }
        if (startedRef.current) {
          // Dropped mid-sim. The server-side replay can't resume on a fresh
          // consumer, so terminate cleanly (no restart, no reconnect loop).
          endedRef.current = true;
          setConnState("reconnecting");
          setStatus("error");
          wsRef.current?.close();
        }
      },
    });
  }, [onMessage]);

  const stop = React.useCallback(() => {
    endedRef.current = true;
    send({ op: "stop" });
    wsRef.current?.close();
    setStatus("done");
  }, []);
  React.useEffect(() => () => wsRef.current?.close(), []);

  const controls = React.useMemo(() => ({
    start, stop,
    pause: () => { send({ op: "pause" }); setStatus("paused"); },
    resume: () => { send({ op: "resume" }); setStatus("running"); },
    step: () => send({ op: "step" }),
    setSpeed: (v: number) => { send({ op: "speed", value: v }); setSpeedState(v); },
    manualOrder: (action: string, opts: { lots?: number; sl?: number; price?: number } = {}) =>
      send({ op: "manual_order", action, ...opts }),
    adjustSL: (value: number) => send({ op: "adjust_sl", value }),
    annotate: (candle_ts: string, note: string) => {
      send({ op: "annotate", candle_ts, note });
      setAnnotations((a) => [...a, { candle_ts, note }]);
    },
  }), [start, stop]);

  return { chartRef, status, connState, speed, source, info, position, pressure, profile, kpis, log, decisions, annotations, controls };
}

const MAX_MARKERS = 600;

function pushMarker(ref: React.MutableRefObject<Marker[]>, bucketEpoch: number, d: DecisionMsg) {
  const time = Math.floor(bucketEpoch) + 5.5 * 3600;   // candle-bar time + IST offset
  let m: Marker;
  if (d.action === "exit") {
    m = { time, position: "aboveBar", color: WARN, shape: "circle", text: `Exit ${d.pnl_pts ?? ""}` };
  } else if (d.action === "enter_long" || (d.action === "add" && d.side === "LONG")) {
    m = { time, position: "belowBar", color: UP, shape: "arrowUp", text: d.action === "add" ? "+" : "Long" };
  } else {
    m = { time, position: "aboveBar", color: DOWN, shape: "arrowDown", text: d.action === "add" ? "+" : "Short" };
  }
  ref.current.push(m);
  // Decisions arrive in candle-time order, so the array stays sorted on push —
  // no O(n log n) re-sort per decision. Cap it so a long sim can't grow the
  // array (and the per-decision setMarkers payload) without bound.
  if (ref.current.length > MAX_MARKERS) {
    ref.current.splice(0, ref.current.length - MAX_MARKERS);
  }
}
