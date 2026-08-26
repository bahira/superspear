import * as S from "../src/lib/spear/tasks/shared";

async function main() {
  const rows = 500;
  const preds: number[] = [], ys: number[] = [];
  for (let i = 0; i < rows; i++) {
    const s = 92 + 16 * ((i * 0.6180339887) % 1);
    const k = 92 + 16 * ((i * 0.7548776662) % 1);
    const t = 0.15 + 0.75 * ((i * 0.4192388219) % 1);
    const vol = 0.1 + 0.9 * ((i * 0.5412417173) % 1);
    const c = S.bsCall(s, k, t, vol);
    const y = S.impliedVol(c, s, k, t);
    const brenner = 2.5066 * (c / s) / Math.sqrt(t);
    preds.push(brenner); ys.push(y);
    if (i < 5 || i === 250) console.log(`s=${s.toFixed(1)} k=${k.toFixed(1)} t=${t.toFixed(2)} vol=${vol.toFixed(2)} -> c=${c.toFixed(3)} iv=${y.toFixed(4)} brenner=${brenner.toFixed(4)}`);
  }
  let m = 0; for (let i = 0; i < rows; i++) { const e = preds[i] - ys[i]; m += e * e; }
  console.log("Brenner MSE manuel:", (m / rows).toExponential(3));
  const mean = ys.reduce((a, b) => a + b, 0) / rows;
  console.log("y: mean=", mean.toFixed(3), "min=", Math.min(...ys).toFixed(3), "max=", Math.max(...ys).toFixed(3));
}
main().catch((e) => { console.error(e); process.exit(1); });