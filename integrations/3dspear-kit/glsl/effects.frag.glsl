// SPEAR discovered kernels — GLSL translations for Three.js / WebGL shaders.
// Each function is self-contained; paste into your shader stage.
// Provenance numbers are in integrations/3dspear-kit/README.md.

// Gaussian approx of e^(-x²/2) — 9 ALU units, MSE 6.4e-4. Low-end friendly.
float gaussianFast(float x) {
    float s = 0.207 * x * x + 1.0;
    return 1.02232 / (s * s * s);
}

// KdV soliton crest — 7 ALU units, water/wave displacement per vertex.
float kdvCrest(float x) {
    return -1.553578 * max(sqrt(abs(6.660595 - max(3.1671, x))), 0.603662) + 2.906087;
}

// Blackbody red channel vs temperature (Kelvin) — PBR light colors.
float blackbodyRed(float tempK) {
    float m = max(tempK, 6608.647704);
    return 12961264.054148 * ((6608.647704 / m) / (m * m)) + 0.703081;
}

// Exact smoothstep — the reference polynomial, 5 ALU units.
float smoothstepExact(float t) {
    return t * t * (3.0 - 2.0 * t);
}
