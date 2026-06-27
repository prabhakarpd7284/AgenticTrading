/**
 * Imperative candlestick chart for live/streaming data.
 *
 * High-frequency tick updates must NOT flow through React props/state — they'd
 * thrash renders. Instead the parent holds a ref to this component and calls
 * `updateBar` per frame (lightweight-charts `series.update` mutates the last bar
 * if the time matches, else appends — so candles form progressively).
 *
 * Built on lightweight-charts v4 (already used in monthly/TradeChartModal).
 */
import * as React from "react";
import {
  ColorType,
  createChart,
  LineStyle,
  LineType,
  type IChartApi,
  type ISeriesApi,
  type IPriceLine,
  type SeriesMarker,
  type Time,
  type UTCTimestamp,
  type WhitespaceData,
} from "lightweight-charts";

// IST is a fixed +5:30; lightweight-charts renders UTCTimestamps as UTC, so we
// shift epoch-seconds to read in IST wall-clock.
const IST_OFFSET_SEC = 5.5 * 3600;

export function epochToTime(epochSecs: number): UTCTimestamp {
  return (Math.floor(epochSecs) + IST_OFFSET_SEC) as UTCTimestamp;
}

export interface Bar {
  time: UTCTimestamp;
  open: number;
  high: number;
  low: number;
  close: number;
}

export interface LiveCandleHandle {
  updateBar(bar: Bar): void;
  setMarkers(markers: SeriesMarker<Time>[]): void;
  setSL(price: number | null): void;
  updateTrailSL(time: UTCTimestamp, price: number): void;
  breakTrailSL(time: UTCTimestamp): void;
  reset(): void;
}

function tokenRgb(name: string, fallback = "#888888"): string {
  if (typeof window === "undefined") return fallback;
  const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return v ? `rgb(${v.split(/\s+/).join(", ")})` : fallback;
}

export const LiveCandleChart = React.forwardRef<LiveCandleHandle, { className?: string }>(
  function LiveCandleChart({ className }, ref) {
    const elRef = React.useRef<HTMLDivElement>(null);
    const chartRef = React.useRef<IChartApi | null>(null);
    const seriesRef = React.useRef<ISeriesApi<"Candlestick"> | null>(null);
    const slLineRef = React.useRef<IPriceLine | null>(null);
    const trailRef = React.useRef<ISeriesApi<"Line"> | null>(null);

    React.useImperativeHandle(ref, (): LiveCandleHandle => ({
      updateBar(bar) {
        seriesRef.current?.update(bar);
      },
      setMarkers(markers) {
        seriesRef.current?.setMarkers(markers);
      },
      setSL(price) {
        const s = seriesRef.current;
        if (!s) return;
        if (price == null) {
          if (slLineRef.current) { s.removePriceLine(slLineRef.current); slLineRef.current = null; }
          return;
        }
        if (slLineRef.current) {
          slLineRef.current.applyOptions({ price });
        } else {
          slLineRef.current = s.createPriceLine({
            price, color: tokenRgb("--warn", "#f59e0b"), lineWidth: 2,
            lineStyle: LineStyle.Dashed, axisLabelVisible: true, title: "SL",
          });
        }
      },
      updateTrailSL(time, price) {
        // stepped line records where the trailing stop has ratcheted over time
        if (price > 0) trailRef.current?.update({ time, value: price });
      },
      breakTrailSL(time) {
        // A whitespace point (time only, no value) ends the line segment so it
        // doesn't span flat gaps. lightweight-charts v4 accepts WhitespaceData
        // via series.update — typed (not `as never`) so a shape drift is caught.
        trailRef.current?.update({ time } as WhitespaceData<Time>);
      },
      reset() {
        seriesRef.current?.setData([]);
        seriesRef.current?.setMarkers([]);
        trailRef.current?.setData([]);
        if (slLineRef.current && seriesRef.current) {
          seriesRef.current.removePriceLine(slLineRef.current);
          slLineRef.current = null;
        }
      },
    }), []);

    React.useEffect(() => {
      const el = elRef.current;
      if (!el) return;
      const chart = createChart(el, {
        width: el.clientWidth,
        height: el.clientHeight,
        layout: {
          background: { type: ColorType.Solid, color: "rgba(0,0,0,0)" },
          textColor: "rgb(160, 166, 174)",
          fontSize: 11,
        },
        grid: {
          vertLines: { color: "rgba(148,163,184,0.07)" },
          horzLines: { color: "rgba(148,163,184,0.07)" },
        },
        rightPriceScale: { borderColor: "rgba(148,163,184,0.2)" },
        timeScale: { borderColor: "rgba(148,163,184,0.2)", timeVisible: true, secondsVisible: false },
        crosshair: { mode: 0 },
      });
      const series = chart.addCandlestickSeries({
        upColor: tokenRgb("--pnl-up", "#10b981"),
        downColor: tokenRgb("--pnl-down", "#f43f5e"),
        wickUpColor: tokenRgb("--pnl-up", "#10b981"),
        wickDownColor: tokenRgb("--pnl-down", "#f43f5e"),
        borderVisible: false, lastValueVisible: true, priceLineVisible: false,
      });
      const trail = chart.addLineSeries({
        color: tokenRgb("--warn", "#f59e0b"), lineWidth: 1, lineType: LineType.WithSteps,
        priceLineVisible: false, lastValueVisible: false, crosshairMarkerVisible: false,
      });
      chartRef.current = chart;
      seriesRef.current = series;
      trailRef.current = trail;

      const ro = new ResizeObserver(() => {
        chart.applyOptions({ width: el.clientWidth, height: el.clientHeight });
      });
      ro.observe(el);
      return () => {
        ro.disconnect();
        chart.remove();
        chartRef.current = null;
        seriesRef.current = null;
        trailRef.current = null;
        slLineRef.current = null;
      };
    }, []);

    return <div ref={elRef} className={className} />;
  },
);
