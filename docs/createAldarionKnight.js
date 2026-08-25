// img2threejs/createAldarionKnight.ts
import * as THREE from "three";
import { RoomEnvironment } from "three/examples/jsm/environments/RoomEnvironment.js";
import { EffectComposer } from "three/examples/jsm/postprocessing/EffectComposer.js";
import { RenderPass } from "three/examples/jsm/postprocessing/RenderPass.js";
import { BokehPass } from "three/examples/jsm/postprocessing/BokehPass.js";
import { UnrealBloomPass } from "three/examples/jsm/postprocessing/UnrealBloomPass.js";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
function buildWatertightCapsule(radius, cylLength, capSegments, radialSegments, heightSegments) {
  const positions = [];
  const indices = [];
  const uvs = [];
  const halfCyl = cylLength / 2;
  const totalSpan = 2 * (Math.PI / 2 * radius) + Math.max(0, cylLength);
  const vOf = (fromBottom) => totalSpan > 0 ? fromBottom / totalSpan : 0;
  const bottomPoleIndex = positions.length / 3;
  positions.push(0, -halfCyl - radius, 0);
  uvs.push(0.5, vOf(0));
  const ringStarts = [];
  const ringV = [];
  for (let ring = 1; ring <= capSegments; ring += 1) {
    const phi = Math.PI / 2 * (ring / capSegments);
    const y = -halfCyl - radius * Math.cos(phi);
    const r = radius * Math.sin(phi);
    const start = positions.length / 3;
    ringStarts.push(start);
    ringV.push(vOf(radius * phi));
    for (let radial = 0; radial < radialSegments; radial += 1) {
      const theta = radial / radialSegments * Math.PI * 2;
      positions.push(r * Math.cos(theta), y, r * Math.sin(theta));
      uvs.push(radial / radialSegments, vOf(radius * phi));
    }
  }
  const cylinderRingStarts = [];
  if (cylLength > 0) {
    for (let step = 1; step <= heightSegments; step += 1) {
      const y = -halfCyl + cylLength * step / heightSegments;
      const start = positions.length / 3;
      cylinderRingStarts.push(start);
      const v = vOf(radius * (Math.PI / 2) + halfCyl + y);
      for (let radial = 0; radial < radialSegments; radial += 1) {
        const theta = radial / radialSegments * Math.PI * 2;
        positions.push(radius * Math.cos(theta), y, radius * Math.sin(theta));
        uvs.push(radial / radialSegments, v);
      }
    }
  }
  const topRingStarts = [];
  for (let ring = capSegments - 1; ring >= 1; ring -= 1) {
    const phi = Math.PI / 2 * (ring / capSegments);
    const y = halfCyl + radius * Math.cos(phi);
    const r = radius * Math.sin(phi);
    const start = positions.length / 3;
    topRingStarts.push(start);
    const v = vOf(radius * (Math.PI / 2) + Math.max(0, cylLength) + radius * (Math.PI / 2 - phi));
    for (let radial = 0; radial < radialSegments; radial += 1) {
      const theta = radial / radialSegments * Math.PI * 2;
      positions.push(r * Math.cos(theta), y, r * Math.sin(theta));
      uvs.push(radial / radialSegments, v);
    }
  }
  const topPoleIndex = positions.length / 3;
  positions.push(0, halfCyl + radius, 0);
  uvs.push(0.5, vOf(totalSpan));
  const firstBottomRing = ringStarts[0];
  for (let radial = 0; radial < radialSegments; radial += 1) {
    const next = (radial + 1) % radialSegments;
    indices.push(bottomPoleIndex, firstBottomRing + radial, firstBottomRing + next);
  }
  const allRings = [...ringStarts, ...cylinderRingStarts, ...topRingStarts];
  for (let i = 0; i < allRings.length - 1; i += 1) {
    const a = allRings[i];
    const b = allRings[i + 1];
    for (let radial = 0; radial < radialSegments; radial += 1) {
      const next = (radial + 1) % radialSegments;
      indices.push(a + radial, a + next, b + next);
      indices.push(a + radial, b + next, b + radial);
    }
  }
  const lastRing = allRings[allRings.length - 1];
  for (let radial = 0; radial < radialSegments; radial += 1) {
    const next = (radial + 1) % radialSegments;
    indices.push(topPoleIndex, lastRing + next, lastRing + radial);
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  geometry.setAttribute("uv", new THREE.Float32BufferAttribute(uvs, 2));
  geometry.setIndex(indices);
  geometry.computeVertexNormals();
  return geometry;
}
function buildExtrudeShape(points, holes) {
  const shape = new THREE.Shape();
  if (points.length > 0) {
    shape.moveTo(points[0][0], points[0][1]);
    for (let i = 1; i < points.length; i += 1) {
      shape.lineTo(points[i][0], points[i][1]);
    }
  }
  for (const loop of holes ?? []) {
    if (loop.length < 3) continue;
    const path = new THREE.Path();
    path.moveTo(loop[0][0], loop[0][1]);
    for (let i = 1; i < loop.length; i += 1) path.lineTo(loop[i][0], loop[i][1]);
    path.closePath();
    shape.holes.push(path);
  }
  return shape;
}
function ovalLoop(cx, cy, rx, ry, seg = 24) {
  const loop = [];
  for (let i = 0; i < seg; i += 1) {
    const a = i / seg * Math.PI * 2;
    loop.push([cx + Math.cos(a) * rx, cy + Math.sin(a) * ry]);
  }
  return loop;
}
function buildExtrudeGeometry(profile) {
  const holes = [...profile.holes ?? [], ...(profile.ovalHoles ?? []).map((o) => ovalLoop(o.cx, o.cy, o.rx, o.ry))];
  const shape = buildExtrudeShape(profile.points, holes);
  return new THREE.ExtrudeGeometry(shape, {
    depth: profile.depth,
    bevelEnabled: false,
    steps: 1
  });
}
function hashString(value) {
  let hash = 2166136261;
  for (let index = 0; index < value.length; index += 1) {
    hash ^= value.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }
  return hash >>> 0;
}
function readLayerNumber(value, keys, fallback) {
  if (typeof value === "number") return value;
  if (value && typeof value === "object") {
    const record = value;
    for (const key of keys) {
      if (typeof record[key] === "number") return record[key];
    }
  }
  return fallback;
}
function hexToRgb(hex) {
  const normalized = /^#[0-9a-f]{3}$/i.test(hex) ? "#" + hex.slice(1).split("").map((part) => part + part).join("") : hex;
  const value = /^#[0-9a-f]{6}$/i.test(normalized) ? Number.parseInt(normalized.slice(1), 16) : 9075295;
  return [clampAlbedoChannel(value >> 16 & 255), clampAlbedoChannel(value >> 8 & 255), clampAlbedoChannel(value & 255)];
}
function materialPalette(spec) {
  const palette = spec.colorVariation?.palette;
  if (Array.isArray(palette) && palette.length > 0) return palette.filter((value) => typeof value === "string");
  const secondary = spec.albedo?.secondary;
  const colors = [spec.baseColor ?? spec.color ?? spec.albedo?.dominant, ...Array.isArray(secondary) ? secondary : []];
  return colors.filter((value) => typeof value === "string" && value.startsWith("#"));
}
function clamp01(value) {
  return Math.max(0, Math.min(1, value));
}
function clampAlbedoChannel(value) {
  return Math.max(30, Math.min(240, Math.round(value)));
}
function clampPbrF0(value) {
  return Math.max(0.02, Math.min(1, value));
}
function clampPbrIor(value) {
  return Math.max(1, Math.min(2.5, value));
}
function clampPbrMetalness(value) {
  return value >= 0.5 ? 1 : 0;
}
function clampedAlbedoColor(spec) {
  const source = typeof spec.baseColor === "string" ? spec.baseColor : "#8A7A5F";
  return new THREE.Color().setStyle(source, THREE.SRGBColorSpace);
}
function smoothCurve(value) {
  return value * value * (3 - 2 * value);
}
function periodicHash(x, y, seed, periodX, periodY) {
  const wrappedX = (x % periodX + periodX) % periodX;
  const wrappedY = (y % periodY + periodY) % periodY;
  let value = Math.imul(wrappedX + seed * 17, 374761393) ^ Math.imul(wrappedY + seed * 31, 668265263);
  value = Math.imul(value ^ value >>> 13, 1274126177);
  return ((value ^ value >>> 16) >>> 0) / 4294967295;
}
function periodicValueNoise(u, v, seed, periodX, periodY) {
  const x = u * periodX;
  const y = v * periodY;
  const x0 = Math.floor(x);
  const y0 = Math.floor(y);
  const tx = smoothCurve(x - x0);
  const ty = smoothCurve(y - y0);
  const a = periodicHash(x0, y0, seed, periodX, periodY);
  const b = periodicHash(x0 + 1, y0, seed, periodX, periodY);
  const c = periodicHash(x0, y0 + 1, seed, periodX, periodY);
  const d = periodicHash(x0 + 1, y0 + 1, seed, periodX, periodY);
  return THREE.MathUtils.lerp(THREE.MathUtils.lerp(a, b, tx), THREE.MathUtils.lerp(c, d, tx), ty);
}
function surfaceBands(spec) {
  const source = Array.isArray(spec.surfaceFrequencyBands) ? spec.surfaceFrequencyBands : [];
  const parsed = source.flatMap((item) => {
    if (!item || typeof item !== "object") return [];
    const band = item;
    const frequency = typeof band.frequency === "number" ? band.frequency : 0;
    const amplitude = typeof band.amplitude === "number" ? band.amplitude : 0;
    if (frequency <= 0 || amplitude <= 0) return [];
    const stretch = Array.isArray(band.stretch) ? band.stretch : [1, 1];
    const description = `${String(band.pattern ?? "")} ${String(band.role ?? "")}`.toLowerCase();
    return [{
      frequency,
      amplitude,
      stretchX: typeof stretch[0] === "number" ? Math.max(0.1, stretch[0]) : 1,
      stretchY: typeof stretch[1] === "number" ? Math.max(0.1, stretch[1]) : 1,
      ridge: /(ridge|groove|grain|fiber|striated|crack)/.test(description)
    }];
  });
  return parsed.length > 0 ? parsed : [
    { frequency: 2, amplitude: 0.42, stretchX: 1, stretchY: 1, ridge: false },
    { frequency: 12, amplitude: 0.22, stretchX: 1, stretchY: 1, ridge: false },
    { frequency: 56, amplitude: 0.08, stretchX: 1, stretchY: 1, ridge: false }
  ];
}
function sampleSurface(u, v, bands, seed) {
  let value = 0;
  let weight = 0;
  for (let index = 0; index < bands.length; index += 1) {
    const band = bands[index];
    const periodX = Math.max(1, Math.round(band.frequency * band.stretchX));
    const periodY = Math.max(1, Math.round(band.frequency * band.stretchY));
    let sample = periodicValueNoise(u, v, seed + index * 1013, periodX, periodY);
    if (band.ridge) sample = 1 - Math.abs(sample * 2 - 1);
    value += sample * band.amplitude;
    weight += band.amplitude;
  }
  return weight > 0 ? clamp01(value / weight) : 0.5;
}
function mixPalette(colors, value) {
  if (colors.length === 1) return colors[0];
  const scaled = clamp01(value) * (colors.length - 1);
  const index = Math.min(colors.length - 2, Math.floor(scaled));
  const mix = scaled - index;
  const a = colors[index];
  const b = colors[index + 1];
  return [
    Math.round(THREE.MathUtils.lerp(a[0], b[0], mix)),
    Math.round(THREE.MathUtils.lerp(a[1], b[1], mix)),
    Math.round(THREE.MathUtils.lerp(a[2], b[2], mix))
  ];
}
function parseRgba(value) {
  const match = /rgba?\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)/.exec(value);
  if (!match) return [138, 122, 95];
  return [clampAlbedoChannel(Number(match[1])), clampAlbedoChannel(Number(match[2])), clampAlbedoChannel(Number(match[3]))];
}
function sampleColorGradient(gradient, u, v) {
  const stops = gradient.stops.length >= 2 ? gradient.stops : [{ offset: 0, color: "rgba(138,122,95,1)" }, { offset: 1, color: "rgba(138,122,95,1)" }];
  let t;
  if (gradient.type === "radial") {
    const [cx, cy] = gradient.axis;
    const dx = u - cx;
    const dy = v - cy;
    const maxRadius = Math.max(1e-3, Math.hypot(Math.max(cx, 1 - cx), Math.max(cy, 1 - cy)));
    t = clamp01(Math.hypot(dx, dy) / maxRadius);
  } else {
    const [ax, ay] = gradient.axis;
    const projection = (u - 0.5) * ax + (v - 0.5) * ay;
    const maxProjection = 0.5 * (Math.abs(ax) + Math.abs(ay)) || 0.5;
    t = clamp01(projection / maxProjection + 0.5);
  }
  const scaled = t * (stops.length - 1);
  const index = Math.min(stops.length - 2, Math.max(0, Math.floor(scaled)));
  const mix = scaled - index;
  const a = parseRgba(stops[index].color);
  const b = parseRgba(stops[index + 1].color);
  return [
    THREE.MathUtils.lerp(a[0], b[0], mix),
    THREE.MathUtils.lerp(a[1], b[1], mix),
    THREE.MathUtils.lerp(a[2], b[2], mix)
  ];
}
function writePixel(data, offset, red, green, blue) {
  data[offset] = Math.max(0, Math.min(255, Math.round(red)));
  data[offset + 1] = Math.max(0, Math.min(255, Math.round(green)));
  data[offset + 2] = Math.max(0, Math.min(255, Math.round(blue)));
  data[offset + 3] = 255;
}
function makeCanvas(size) {
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  return canvas;
}
function createMapTexture(canvas, colorSpace, spec, options) {
  const texture = new THREE.CanvasTexture(canvas);
  const projection = spec.textureProjection && typeof spec.textureProjection === "object" ? spec.textureProjection : {};
  const repeat = Array.isArray(projection.repeat) ? projection.repeat : [2, 2];
  texture.colorSpace = colorSpace;
  texture.wrapS = THREE.RepeatWrapping;
  texture.wrapT = THREE.RepeatWrapping;
  texture.repeat.set(
    typeof repeat[0] === "number" ? repeat[0] : 2,
    typeof repeat[1] === "number" ? repeat[1] : 2
  );
  texture.anisotropy = Math.max(1, Math.round(options.textureAnisotropy ?? projection.anisotropy ?? 8));
  texture.needsUpdate = true;
  return texture;
}
function referenceMapUrl(spec, channel) {
  const reference = spec.referencePbr;
  if (!reference || typeof reference !== "object") return null;
  if (reference.usable === false) return null;
  const confidence = typeof reference.confidence === "number" ? reference.confidence : typeof reference.estimatedFidelity === "number" ? reference.estimatedFidelity : 0;
  const threshold = typeof reference.targetThreshold === "number" ? reference.targetThreshold : 0.7;
  if (confidence < threshold) return null;
  const maps = reference.maps;
  if (!maps || typeof maps !== "object") return null;
  const map = maps[channel];
  if (!map || typeof map !== "object") return null;
  const record = map;
  const url = typeof record.url === "string" && record.url.trim() ? record.url : record.path;
  return typeof url === "string" && url.trim() ? url : null;
}
function createLoadedMapTexture(url, colorSpace, spec, options) {
  const texture = new THREE.TextureLoader().load(url);
  const projection = spec.textureProjection && typeof spec.textureProjection === "object" ? spec.textureProjection : {};
  const repeat = Array.isArray(projection.repeat) ? projection.repeat : [1, 1];
  texture.colorSpace = colorSpace;
  texture.wrapS = THREE.RepeatWrapping;
  texture.wrapT = THREE.RepeatWrapping;
  texture.repeat.set(
    typeof repeat[0] === "number" ? repeat[0] : 1,
    typeof repeat[1] === "number" ? repeat[1] : 1
  );
  texture.anisotropy = Math.max(1, Math.round(options.textureAnisotropy ?? projection.anisotropy ?? 8));
  texture.needsUpdate = true;
  return texture;
}
function makeReferenceTextureSet(spec, options) {
  const albedo = referenceMapUrl(spec, "albedo");
  const roughness = referenceMapUrl(spec, "roughness");
  const height = referenceMapUrl(spec, "height");
  const normal = referenceMapUrl(spec, "normal");
  const ao = referenceMapUrl(spec, "ao");
  if (!albedo || !roughness || !height || !normal || !ao) return null;
  return {
    albedo: createLoadedMapTexture(albedo, THREE.SRGBColorSpace, spec, options),
    roughness: createLoadedMapTexture(roughness, THREE.NoColorSpace, spec, options),
    height: createLoadedMapTexture(height, THREE.NoColorSpace, spec, options),
    normal: createLoadedMapTexture(normal, THREE.NoColorSpace, spec, options),
    ao: createLoadedMapTexture(ao, THREE.NoColorSpace, spec, options),
    source: "reference-pixel-extraction"
  };
}
function makeProceduralTextureSet(id, spec, options) {
  if (typeof document === "undefined") return null;
  const qualityFirst = (options.qualityPriority ?? "reference-fidelity") === "reference-fidelity";
  const requested = options.textureSize ?? spec.textureResolution;
  const requestedSize = typeof requested === "number" && Number.isFinite(requested) ? requested : qualityFirst ? 1024 : 512;
  const size = Math.max(256, Math.min(2048, 2 ** Math.round(Math.log2(requestedSize))));
  const canvases = {
    albedo: makeCanvas(size),
    roughness: makeCanvas(size),
    height: makeCanvas(size),
    normal: makeCanvas(size),
    ao: makeCanvas(size)
  };
  const contexts = {
    albedo: canvases.albedo.getContext("2d"),
    roughness: canvases.roughness.getContext("2d"),
    height: canvases.height.getContext("2d"),
    normal: canvases.normal.getContext("2d"),
    ao: canvases.ao.getContext("2d")
  };
  if (!contexts.albedo || !contexts.roughness || !contexts.height || !contexts.normal || !contexts.ao) return null;
  const images = {
    albedo: contexts.albedo.createImageData(size, size),
    roughness: contexts.roughness.createImageData(size, size),
    height: contexts.height.createImageData(size, size),
    normal: contexts.normal.createImageData(size, size),
    ao: contexts.ao.createImageData(size, size)
  };
  const seed = hashString(id);
  const bands = surfaceBands(spec);
  const heightField = new Float32Array(size * size);
  const roughnessField = new Float32Array(size * size);
  const palette = materialPalette(spec);
  const fallback = typeof spec.baseColor === "string" ? spec.baseColor : "#8A7A5F";
  const colors = (palette.length >= 2 ? palette : [fallback, "#6E614B", "#A08F70"]).map(hexToRgb);
  const baseRoughness = clamp01(readLayerNumber(spec.roughness, ["base"], 0.76));
  const roughnessVariation = clamp01(readLayerNumber(spec.roughness, ["variation"], 0.18));
  const colorAmplitude = clamp01(readLayerNumber(spec.colorVariation, ["amplitude", "variation"], 0.18));
  const heightCorrelation = clamp01(readLayerNumber(spec.colorVariation, ["heightCorrelation"], 0.3));
  const colorGradient = spec.colorGradient;
  for (let y = 0; y < size; y += 1) {
    const v = y / size;
    for (let x = 0; x < size; x += 1) {
      const u = x / size;
      const index = y * size + x;
      const height = sampleSurface(u, v, bands, seed + 101);
      const roughNoise = sampleSurface(u, v, bands, seed + 7001);
      const colorNoise = sampleSurface(u, v, bands, seed + 15013);
      heightField[index] = height;
      roughnessField[index] = clamp01(baseRoughness + (roughNoise - 0.5) * roughnessVariation * 2);
      let color;
      if (colorGradient) {
        color = sampleColorGradient(colorGradient, u, v);
      } else {
        const paletteValue = clamp01(
          0.5 + (colorNoise - 0.5) * colorAmplitude * 2 + (height - 0.5) * heightCorrelation
        );
        color = mixPalette(colors, paletteValue);
      }
      writePixel(images.albedo.data, index * 4, color[0], color[1], color[2]);
    }
  }
  const normalStrength = Math.max(0.05, readLayerNumber(spec.normal, ["strength", "amplitude"], 0.35));
  const aoStrength = clamp01(readLayerNumber(spec.ambientOcclusion, ["cavityStrength", "strength"], 0.35));
  for (let y = 0; y < size; y += 1) {
    const up = (y - 1 + size) % size * size;
    const down = (y + 1) % size * size;
    for (let x = 0; x < size; x += 1) {
      const left = (x - 1 + size) % size;
      const right = (x + 1) % size;
      const index = y * size + x;
      const center = heightField[index];
      const dx = (heightField[y * size + right] - heightField[y * size + left]) * normalStrength * 6;
      const dy = (heightField[down + x] - heightField[up + x]) * normalStrength * 6;
      const inverseLength = 1 / Math.sqrt(dx * dx + dy * dy + 1);
      const normalX = -dx * inverseLength;
      const normalY = -dy * inverseLength;
      const normalZ = inverseLength;
      const neighborAverage = (heightField[y * size + left] + heightField[y * size + right] + heightField[up + x] + heightField[down + x]) * 0.25;
      const cavity = Math.max(0, neighborAverage - center);
      const ao = clamp01(1 - aoStrength * (cavity * 12 + (1 - center) * 0.16));
      const offset = index * 4;
      const heightByte = center * 255;
      const roughnessByte = roughnessField[index] * 255;
      writePixel(images.height.data, offset, heightByte, heightByte, heightByte);
      writePixel(images.roughness.data, offset, roughnessByte, roughnessByte, roughnessByte);
      writePixel(
        images.normal.data,
        offset,
        (normalX * 0.5 + 0.5) * 255,
        (normalY * 0.5 + 0.5) * 255,
        (normalZ * 0.5 + 0.5) * 255
      );
      writePixel(images.ao.data, offset, ao * 255, ao * 255, ao * 255);
    }
  }
  contexts.albedo.putImageData(images.albedo, 0, 0);
  contexts.roughness.putImageData(images.roughness, 0, 0);
  contexts.height.putImageData(images.height, 0, 0);
  contexts.normal.putImageData(images.normal, 0, 0);
  contexts.ao.putImageData(images.ao, 0, 0);
  return {
    albedo: createMapTexture(canvases.albedo, THREE.SRGBColorSpace, spec, options),
    roughness: createMapTexture(canvases.roughness, THREE.NoColorSpace, spec, options),
    height: createMapTexture(canvases.height, THREE.NoColorSpace, spec, options),
    normal: createMapTexture(canvases.normal, THREE.NoColorSpace, spec, options),
    ao: createMapTexture(canvases.ao, THREE.NoColorSpace, spec, options),
    source: "procedural"
  };
}
function createSculptMaterial(id, spec, options, denseComponent = false) {
  const textureless = spec.textureless?.declared === true;
  const textures = textureless ? null : makeReferenceTextureSet(spec, options) ?? makeProceduralTextureSet(id, spec, options);
  const material = new THREE.MeshPhysicalMaterial({
    color: textures ? 16777215 : clampedAlbedoColor(spec),
    roughness: textures ? 1 : clamp01(readLayerNumber(spec.roughness, ["base"], 0.76)),
    metalness: clampPbrMetalness(readLayerNumber(spec.metalness, ["base"], 0)),
    clearcoat: clamp01(readLayerNumber(spec.clearcoat, ["base", "amount"], 0)),
    clearcoatRoughness: clamp01(readLayerNumber(spec.clearcoatRoughness, ["base"], 0.25)),
    transmission: clamp01(readLayerNumber(spec.transmission, ["base", "amount"], 0)),
    ior: clampPbrIor(readLayerNumber(spec.ior, ["base", "value"], 1.5)),
    thickness: Math.max(0, readLayerNumber(spec.thickness, ["base", "amount"], 0)),
    attenuationDistance: Math.max(1e-3, readLayerNumber(spec.attenuationDistance, ["base", "value"], Infinity)),
    attenuationColor: new THREE.Color(typeof spec.attenuationColor === "string" ? spec.attenuationColor : "#ffffff"),
    sheen: clamp01(readLayerNumber(spec.sheen, ["base", "amount"], 0)),
    sheenColor: new THREE.Color(typeof spec.sheenColor === "string" ? spec.sheenColor : "#ffffff"),
    sheenRoughness: clamp01(readLayerNumber(spec.sheenRoughness, ["base"], 1)),
    iridescence: clamp01(readLayerNumber(spec.iridescence, ["base", "amount"], 0)),
    iridescenceIOR: clampPbrIor(readLayerNumber(spec.iridescenceIOR, ["base", "value"], 1.3)),
    anisotropy: clamp01(readLayerNumber(spec.anisotropy, ["base", "amount"], 0)),
    anisotropyRotation: readLayerNumber(spec.anisotropy, ["rotation"], 0),
    specularIntensity: clampPbrF0(readLayerNumber(spec.specularF0 ?? spec.f0 ?? spec.specularIntensity, ["base", "value"], 1)),
    specularColor: new THREE.Color(typeof spec.specularColor === "string" ? spec.specularColor : "#ffffff"),
    emissive: new THREE.Color(typeof spec.emissive === "string" ? spec.emissive : "#000000"),
    emissiveIntensity: Math.max(0, readLayerNumber(spec.emissiveIntensity, ["base"], 1)),
    opacity: clamp01(readLayerNumber(spec.opacity, ["base"], 1)),
    transparent: readLayerNumber(spec.transmission, ["base", "amount"], 0) > 0 || readLayerNumber(spec.opacity, ["base"], 1) < 1,
    alphaTest: Math.max(0, readLayerNumber(spec.alpha, ["cutoff", "alphaTest"], 0)),
    wireframe: options.wireframe ?? false,
    side: spec.doubleSided === true ? THREE.DoubleSide : THREE.FrontSide,
    flatShading: spec.flatShading === true
  });
  if (textures) {
    material.map = textures.albedo;
    material.roughnessMap = textures.roughness;
    material.normalMap = textures.normal;
    material.normalScale.setScalar(Math.max(0.05, readLayerNumber(spec.normal, ["strength", "amplitude"], 0.35)));
    material.aoMap = textures.ao;
    material.aoMap.channel = 0;
    material.aoMapIntensity = readLayerNumber(spec.ambientOcclusion, ["cavityStrength", "strength"], 0.35);
    const denseMesh = denseComponent || spec.denseMesh === true || spec.geometryDensity === "dense" || spec.topologyClass === "dense";
    const bumpScale = Math.max(0, readLayerNumber(spec.bump, ["amplitude", "strength"], 0));
    const effectiveBumpScale = denseMesh ? Math.max(0.05, bumpScale) : bumpScale;
    if (effectiveBumpScale > 0) {
      material.bumpMap = textures.height;
      material.bumpScale = effectiveBumpScale;
    }
    const displacementScale = Math.max(0, readLayerNumber(spec.displacement, ["amplitude", "strength"], 0));
    const effectiveDisplacementScale = denseMesh ? Math.max(5e-3, displacementScale) : displacementScale;
    if (effectiveDisplacementScale > 0) {
      material.displacementMap = textures.height;
      material.displacementScale = effectiveDisplacementScale;
      material.displacementBias = -effectiveDisplacementScale * 0.5;
    }
  }
  material.envMapIntensity = readLayerNumber(spec, ["envMapIntensity"], 0.8);
  material.userData.sculptMaterial = spec;
  material.userData.proceduralMapsIndependent = true;
  material.userData.pbrConstraints = { albedoRange: [30, 240], binaryMetalness: true, f0Range: [0.02, 1], iorRange: [1, 2.5] };
  material.userData.pbrTextureSource = textures?.source ?? "flat-fallback";
  material.userData.referencePbr = spec.referencePbr ?? null;
  material.userData.referenceMaterialId = spec.referenceMaterialId ?? spec.materialReference?.profileId ?? null;
  material.userData.materialEvidence = spec.materialEvidence ?? null;
  material.userData.validationViews = spec.materialReference?.validationViews ?? [];
  material.needsUpdate = true;
  return material;
}
function readVector3(value, fallback) {
  if (Array.isArray(value) && value.length === 3 && value.every((item) => typeof item === "number")) {
    return new THREE.Vector3(value[0], value[1], value[2]);
  }
  return new THREE.Vector3(fallback[0], fallback[1], fallback[2]);
}
function readNumber(value, fallback) {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}
function makeAttachmentEndpoint(attachment) {
  if (!attachment || typeof attachment !== "object") return null;
  const record = attachment;
  const start = readVector3(record.localStart, [0, 0, 0]);
  const end = readVector3(record.localEnd, [0, 1, 0]);
  const delta = end.clone().sub(start);
  const length = delta.length();
  if (length <= 1e-4) return null;
  const direction = delta.clone().normalize();
  const quaternion = new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 1, 0), direction);
  const baseRadius = Math.max(5e-3, readNumber(record.baseRadius, 0.06));
  const endRadius = Math.max(3e-3, readNumber(record.endRadius, baseRadius * 0.55));
  return {
    start,
    midpoint: delta.multiplyScalar(0.5),
    quaternion,
    length,
    baseRadius,
    endRadius
  };
}
function createAldarionKnightModel(options = {}) {
  const root = new THREE.Group();
  root.name = "Aldarion Knight";
  root.userData.reconstructionEvidence = { "itemFamily": null, "subtype": null, "componentAdapter": null, "route": null, "exactnessTier": null, "referenceCamera": { "solved": false, "fovDegrees": 40, "aspect": 1, "orientation": { "yaw": 0, "pitch": 0, "roll": 0 }, "positionHint": [0, 0, 3], "note": "For likeness work, solve the reference camera (forge/stage1_intake/solve_camera_pose.py) so the review render aligns with the photo and the reference can be projected. Confirm by overlay review." }, "approximationNotes": [] };
  root.userData.materialPipeline = {};
  root.userData.materialReferenceRegistry = null;
  const materialMap = {};
  materialMap["steel"] = createSculptMaterial(
    "steel",
    { "id": "steel", "name": "Armour steel", "type": "standard", "shaderModel": "MeshStandardMaterial / PBR", "baseColor": "#8a8f96", "metalness": 1, "roughness": { "base": 0.42, "map": "edge wear lowers roughness on bevels; cavities and rivet lines raise it (visual analysis of sprite-front.png)" }, "albedo": { "dominant": "#8a8f96", "secondary": ["#6e747c", "#a7adb4"], "samplingNotes": "warm-grey plate with battle scuffs" }, "localOverrides": [{ "region": "plate edges", "color": "#b9bec5", "note": "worn highlights on bevels" }], "textureResolution": 2048, "textureProjection": { "mode": "triplanar-procedural", "texelDensity": "2048px/m \u2014 plate seams at 8px/cm equivalent", "note": "front-view crop projection reserved for face + chest emblem per projection-route.md" }, "surfaceFrequency": { "macro": "plate segmentation & bevel wear", "meso": "brushed grinding marks along lames", "micro": "fine scratches + dust in recesses" }, "roughnessMap": { "independent": true, "base": 0.42, "variation": "edge wear lowers roughness; cavities raise it", "source": "visual-analysis of sprite-front.png" }, "aoResponse": { "independent": true, "note": "cavity darkening in recesses, under lames, cape folds" }, "referencePbr": { "source": "agent-vision analysis of sprite-front.png (per-crop)", "metalness": 1, "roughness": 0.42, "confidence": 0.86, "note": "inference from stylized reference, not inverse rendering", "usable": true, "maps": { "albedo": { "url": "img2threejs/pbr-evidence/steel/steel_albedo.png", "note": "extracted from sprite-front.png crop" }, "roughness": { "url": "img2threejs/pbr-evidence/steel/steel_roughness.png", "note": "extracted from sprite-front.png crop" }, "height": { "url": "img2threejs/pbr-evidence/steel/steel_height.png", "note": "extracted from sprite-front.png crop" }, "normal": { "url": "img2threejs/pbr-evidence/steel/steel_normal.png", "note": "extracted from sprite-front.png crop" }, "ao": { "url": "img2threejs/pbr-evidence/steel/steel_ao.png", "note": "extracted from sprite-front.png crop" } } }, "ambientOcclusion": { "response": "cavity darkening under lames and along rivet rives" }, "surfaceFrequencyBands": [{ "id": "macro", "frequency": 1.2, "amplitude": 0.5 }, { "id": "meso", "frequency": 6, "amplitude": 0.3 }, { "id": "micro", "frequency": 24, "amplitude": 0.15 }] },
    options
  );
  materialMap["goldTrim"] = createSculptMaterial(
    "goldTrim",
    { "id": "goldTrim", "name": "Gold filigree & trim", "type": "standard", "shaderModel": "MeshStandardMaterial / PBR", "baseColor": "#c9a04e", "metalness": 1, "roughness": { "base": 0.3, "map": "polished highs on relief tops; engraving recesses rougher" }, "albedo": { "dominant": "#c9a04e", "secondary": ["#e0c27a", "#8a6a2e"], "samplingNotes": "ochre-gold filigree linework" }, "textureResolution": 2048, "textureProjection": { "mode": "triplanar-procedural", "texelDensity": "2048px/m \u2014 plate seams at 8px/cm equivalent", "note": "front-view crop projection reserved for face + chest emblem per projection-route.md" }, "surfaceFrequency": { "macro": "plate segmentation & bevel wear", "meso": "brushed grinding marks along lames", "micro": "filigree engraving lines" }, "roughnessMap": { "independent": true, "base": 0.3, "variation": "edge wear lowers roughness; cavities raise it", "source": "visual-analysis of sprite-front.png" }, "aoResponse": { "independent": true, "note": "cavity darkening in recesses, under lames, cape folds" }, "referencePbr": { "source": "agent-vision analysis of sprite-front.png (per-crop)", "metalness": 1, "roughness": 0.3, "confidence": 0.86, "note": "inference from stylized reference, not inverse rendering", "usable": true, "maps": { "albedo": { "url": "img2threejs/pbr-evidence/goldTrim/goldTrim_albedo.png", "note": "extracted from sprite-front.png crop" }, "roughness": { "url": "img2threejs/pbr-evidence/goldTrim/goldTrim_roughness.png", "note": "extracted from sprite-front.png crop" }, "height": { "url": "img2threejs/pbr-evidence/goldTrim/goldTrim_height.png", "note": "extracted from sprite-front.png crop" }, "normal": { "url": "img2threejs/pbr-evidence/goldTrim/goldTrim_normal.png", "note": "extracted from sprite-front.png crop" }, "ao": { "url": "img2threejs/pbr-evidence/goldTrim/goldTrim_ao.png", "note": "extracted from sprite-front.png crop" } } }, "ambientOcclusion": { "response": "recess darkening inside filigree engraving lines" }, "surfaceFrequencyBands": [{ "id": "macro", "frequency": 2, "amplitude": 0.5 }, { "id": "meso", "frequency": 9, "amplitude": 0.3 }, { "id": "micro", "frequency": 40, "amplitude": 0.15 }] },
    options
  );
  materialMap["clothNavy"] = createSculptMaterial(
    "clothNavy",
    { "id": "clothNavy", "name": "Navy cloth (cape/tabard/cowl)", "type": "standard", "shaderModel": "MeshStandardMaterial / PBR", "baseColor": "#2a3550", "metalness": 0, "roughness": { "base": 0.85, "map": "weave keeps uniform high roughness; faded patches slightly smoother" }, "albedo": { "dominant": "#2a3550", "secondary": ["#1d2540", "#3d4a6b"], "samplingNotes": "dark navy with faded patches" }, "textureResolution": 2048, "textureProjection": { "mode": "triplanar-procedural", "texelDensity": "2048px/m \u2014 plate seams at 8px/cm equivalent", "note": "front-view crop projection reserved for face + chest emblem per projection-route.md" }, "surfaceFrequency": { "macro": "drape folds + tear edges", "meso": "weave grain", "micro": "fine scratches + dust in recesses" }, "roughnessMap": { "independent": true, "base": 0.85, "variation": "edge wear lowers roughness; cavities raise it", "source": "visual-analysis of sprite-front.png" }, "aoResponse": { "independent": true, "note": "cavity darkening in recesses, under lames, cape folds" }, "referencePbr": { "source": "agent-vision analysis of sprite-front.png (per-crop)", "metalness": 0, "roughness": 0.85, "confidence": 0.86, "note": "inference from stylized reference, not inverse rendering", "usable": true, "maps": { "albedo": { "url": "img2threejs/pbr-evidence/clothNavy/clothNavy_albedo.png", "note": "extracted from sprite-front.png crop" }, "roughness": { "url": "img2threejs/pbr-evidence/clothNavy/clothNavy_roughness.png", "note": "extracted from sprite-front.png crop" }, "height": { "url": "img2threejs/pbr-evidence/clothNavy/clothNavy_height.png", "note": "extracted from sprite-front.png crop" }, "normal": { "url": "img2threejs/pbr-evidence/clothNavy/clothNavy_normal.png", "note": "extracted from sprite-front.png crop" }, "ao": { "url": "img2threejs/pbr-evidence/clothNavy/clothNavy_ao.png", "note": "extracted from sprite-front.png crop" } } }, "ambientOcclusion": { "response": "fold-shadow pooling in drape creases and tear edges" }, "surfaceFrequencyBands": [{ "id": "macro", "frequency": 0.8, "amplitude": 0.5 }, { "id": "meso", "frequency": 5, "amplitude": 0.3 }, { "id": "micro", "frequency": 18, "amplitude": 0.15 }], "doubleSided": true },
    options
  );
  materialMap["leather"] = createSculptMaterial(
    "leather",
    { "id": "leather", "name": "Leather belt & straps", "type": "standard", "shaderModel": "MeshStandardMaterial / PBR", "baseColor": "#5a4632", "metalness": 0, "roughness": { "base": 0.7, "map": "creased zones rougher; buckle-adjacent polish" }, "albedo": { "dominant": "#5a4632", "secondary": ["#3f2f20", "#7a5f42"] }, "textureResolution": 2048, "textureProjection": { "mode": "triplanar-procedural", "texelDensity": "2048px/m \u2014 plate seams at 8px/cm equivalent", "note": "front-view crop projection reserved for face + chest emblem per projection-route.md" }, "surfaceFrequency": { "macro": "plate segmentation & bevel wear", "meso": "brushed grinding marks along lames", "micro": "fine scratches + dust in recesses" }, "roughnessMap": { "independent": true, "base": 0.7, "variation": "edge wear lowers roughness; cavities raise it", "source": "visual-analysis of sprite-front.png" }, "aoResponse": { "independent": true, "note": "cavity darkening in recesses, under lames, cape folds" }, "referencePbr": { "source": "agent-vision analysis of sprite-front.png (per-crop)", "metalness": 0, "roughness": 0.7, "confidence": 0.86, "note": "inference from stylized reference, not inverse rendering", "usable": true, "maps": { "albedo": { "url": "img2threejs/pbr-evidence/leather/leather_albedo.png", "note": "extracted from sprite-front.png crop" }, "roughness": { "url": "img2threejs/pbr-evidence/leather/leather_roughness.png", "note": "extracted from sprite-front.png crop" }, "height": { "url": "img2threejs/pbr-evidence/leather/leather_height.png", "note": "extracted from sprite-front.png crop" }, "normal": { "url": "img2threejs/pbr-evidence/leather/leather_normal.png", "note": "extracted from sprite-front.png crop" }, "ao": { "url": "img2threejs/pbr-evidence/leather/leather_ao.png", "note": "extracted from sprite-front.png crop" } } }, "ambientOcclusion": { "response": "grain darkening around buckles and stitch lines" }, "surfaceFrequencyBands": [{ "id": "macro", "frequency": 1, "amplitude": 0.5 }, { "id": "meso", "frequency": 7, "amplitude": 0.3 }, { "id": "micro", "frequency": 22, "amplitude": 0.15 }] },
    options
  );
  materialMap["skin"] = createSculptMaterial(
    "skin",
    { "id": "skin", "name": "Skin", "type": "standard", "shaderModel": "MeshStandardMaterial / PBR (rim-SSS approximation)", "baseColor": "#d8b39a", "metalness": 0, "roughness": { "base": 0.6, "map": "uniform mid roughness; slight sheen on forehead/nose" }, "albedo": { "dominant": "#d8b39a", "secondary": ["#b98f74"], "samplingNotes": "soft rim/backlight term instead of true SSS" }, "textureResolution": 2048, "textureProjection": { "mode": "triplanar-procedural", "texelDensity": "2048px/m \u2014 plate seams at 8px/cm equivalent", "note": "front-view crop projection reserved for face + chest emblem per projection-route.md" }, "surfaceFrequency": { "macro": "plate segmentation & bevel wear", "meso": "brushed grinding marks along lames", "micro": "fine scratches + dust in recesses" }, "roughnessMap": { "independent": true, "base": 0.6, "variation": "edge wear lowers roughness; cavities raise it", "source": "visual-analysis of sprite-front.png" }, "aoResponse": { "independent": true, "note": "cavity darkening in recesses, under lames, cape folds" }, "referencePbr": { "source": "agent-vision analysis of sprite-front.png (per-crop)", "metalness": 0, "roughness": 0.6, "confidence": 0.86, "note": "inference from stylized reference, not inverse rendering", "usable": true, "maps": { "albedo": { "url": "img2threejs/pbr-evidence/skin/skin_albedo.png", "note": "extracted from sprite-front.png crop" }, "roughness": { "url": "img2threejs/pbr-evidence/skin/skin_roughness.png", "note": "extracted from sprite-front.png crop" }, "height": { "url": "img2threejs/pbr-evidence/skin/skin_height.png", "note": "extracted from sprite-front.png crop" }, "normal": { "url": "img2threejs/pbr-evidence/skin/skin_normal.png", "note": "extracted from sprite-front.png crop" }, "ao": { "url": "img2threejs/pbr-evidence/skin/skin_ao.png", "note": "extracted from sprite-front.png crop" } } }, "ambientOcclusion": { "response": "soft ambient falloff at neck, under cowl and pauldrons" }, "surfaceFrequencyBands": [{ "id": "macro", "frequency": 0.6, "amplitude": 0.5 }, { "id": "meso", "frequency": 4, "amplitude": 0.3 }, { "id": "micro", "frequency": 16, "amplitude": 0.15 }] },
    options
  );
  materialMap["hairDark"] = createSculptMaterial(
    "hairDark",
    { "id": "hairDark", "name": "Hair (stylized clump masses)", "type": "standard", "shaderModel": "MeshStandardMaterial / PBR", "baseColor": "#1d1a22", "metalness": 0, "roughness": { "base": 0.55, "map": "clump tops slightly glossier than roots" }, "albedo": { "dominant": "#1d1a22", "secondary": ["#33303c"], "samplingNotes": "5-15 clump masses, no strands" }, "textureResolution": 2048, "textureProjection": { "mode": "triplanar-procedural", "texelDensity": "2048px/m \u2014 plate seams at 8px/cm equivalent", "note": "front-view crop projection reserved for face + chest emblem per projection-route.md" }, "surfaceFrequency": { "macro": "plate segmentation & bevel wear", "meso": "brushed grinding marks along lames", "micro": "fine scratches + dust in recesses" }, "roughnessMap": { "independent": true, "base": 0.55, "variation": "edge wear lowers roughness; cavities raise it", "source": "visual-analysis of sprite-front.png" }, "aoResponse": { "independent": true, "note": "cavity darkening in recesses, under lames, cape folds" }, "referencePbr": { "source": "agent-vision analysis of sprite-front.png (per-crop)", "metalness": 0, "roughness": 0.55, "confidence": 0.86, "note": "inference from stylized reference, not inverse rendering", "usable": true, "maps": { "albedo": { "url": "img2threejs/pbr-evidence/hairDark/hairDark_albedo.png", "note": "extracted from sprite-front.png crop" }, "roughness": { "url": "img2threejs/pbr-evidence/hairDark/hairDark_roughness.png", "note": "extracted from sprite-front.png crop" }, "height": { "url": "img2threejs/pbr-evidence/hairDark/hairDark_height.png", "note": "extracted from sprite-front.png crop" }, "normal": { "url": "img2threejs/pbr-evidence/hairDark/hairDark_normal.png", "note": "extracted from sprite-front.png crop" }, "ao": { "url": "img2threejs/pbr-evidence/hairDark/hairDark_ao.png", "note": "extracted from sprite-front.png crop" } } }, "ambientOcclusion": { "response": "mass-to-mass occlusion between clump layers" }, "surfaceFrequencyBands": [{ "id": "macro", "frequency": 2.5, "amplitude": 0.5 }, { "id": "meso", "frequency": 8, "amplitude": 0.3 }, { "id": "micro", "frequency": 30, "amplitude": 0.15 }] },
    options
  );
  const nodes = { root };
  const meshes = {};
  const sockets = {};
  const colliders = {};
  const destructionGroups = {};
  const endpoint_root_0 = makeAttachmentEndpoint(null);
  const node_root_0 = new THREE.Group();
  node_root_0.name = "Aldarion Knight__pivot";
  node_root_0.scale.set(1, 1, 1);
  if (endpoint_root_0) {
    node_root_0.position.copy(endpoint_root_0.start);
    node_root_0.rotation.set(0, 0, 0);
  } else {
    node_root_0.position.set(0, 0, 0);
    node_root_0.rotation.set(0, 0, 0);
  }
  node_root_0.userData.sculptComponent = { "id": "root", "name": "Aldarion Knight", "level": "macro", "role": "body root / skeleton anchor", "parent": null, "importance": 1, "confidence": 0.9, "primitive": "box", "topologyClass": "assembled-solid", "topologyRationale": "Full character assembly; silhouette defined by cape + pauldron asymmetry", "geometryDescriptor": {}, "materialRefs": [], "localFeatures": [], "colorMaterialRecipe": { "dominantAlbedo": "rgba(138, 143, 150, 1.0)", "secondaryAlbedo": "rgba(42, 53, 80, 1.0)", "materialClass": "metal", "materialClassConfidence": 0.85, "evidenceRef": "sprite-front.png#root" }, "actionProfile": { "animationRole": "assemblage complet" }, "material": "steel" };
  node_root_0.userData.actionProfile = { "animationRole": "assemblage complet" };
  (nodes["root"] ?? root).add(node_root_0);
  nodes["root"] = node_root_0;
  const mesh_root_0Geometry = endpoint_root_0 ? new THREE.CylinderGeometry(endpoint_root_0.endRadius, endpoint_root_0.baseRadius, endpoint_root_0.length, 32, 12) : new THREE.BoxGeometry(1, 1, 1, 12, 12, 12);
  if (!endpoint_root_0) {
    mesh_root_0Geometry.scale(1, 1, 1);
  }
  const mesh_root_0 = new THREE.Mesh(
    mesh_root_0Geometry,
    materialMap["steel"] ?? new THREE.MeshStandardMaterial({ color: 8947848 })
  );
  mesh_root_0.name = "Aldarion Knight";
  if (endpoint_root_0) {
    mesh_root_0.position.copy(endpoint_root_0.midpoint);
    mesh_root_0.quaternion.copy(endpoint_root_0.quaternion);
  }
  mesh_root_0.castShadow = options.castShadow ?? true;
  mesh_root_0.receiveShadow = options.receiveShadow ?? true;
  mesh_root_0.userData.sculptComponent = { "id": "root", "name": "Aldarion Knight", "level": "macro", "role": "body root / skeleton anchor", "parent": null, "importance": 1, "confidence": 0.9, "primitive": "box", "topologyClass": "assembled-solid", "topologyRationale": "Full character assembly; silhouette defined by cape + pauldron asymmetry", "geometryDescriptor": {}, "materialRefs": [], "localFeatures": [], "colorMaterialRecipe": { "dominantAlbedo": "rgba(138, 143, 150, 1.0)", "secondaryAlbedo": "rgba(42, 53, 80, 1.0)", "materialClass": "metal", "materialClassConfidence": 0.85, "evidenceRef": "sprite-front.png#root" }, "actionProfile": { "animationRole": "assemblage complet" }, "material": "steel" };
  node_root_0.add(mesh_root_0);
  meshes["root"] = mesh_root_0;
  colliders["root"] = {};
  const endpoint_head_1 = makeAttachmentEndpoint(null);
  const node_head_1 = new THREE.Group();
  node_head_1.name = "Head & hair__pivot";
  node_head_1.scale.set(1, 1, 1);
  if (endpoint_head_1) {
    node_head_1.position.copy(endpoint_head_1.start);
    node_head_1.rotation.set(0, 0, 0);
  } else {
    node_head_1.position.set(0, 0, 0);
    node_head_1.rotation.set(0, 0, 0);
  }
  node_head_1.userData.sculptComponent = { "id": "head", "name": "Head & hair", "level": "macro", "role": "head unit", "parent": "root", "importance": 1, "confidence": 0.9, "primitive": "sphere", "topologyClass": "continuous-sculpt", "topologyRationale": "Skull volume continuous; hair as 5-15 clump masses per contract", "geometryDescriptor": {}, "materialRefs": ["skin", "hairDark"], "localFeatures": [{ "id": "eyes-glossy", "note": "glossy spheres + iris disc + catchlight quad toward moonlight key" }, { "id": "hair-silhouette", "note": "tousled fringe, side-swept part, nape volume \u2014 identity feature" }], "colorMaterialRecipe": { "dominantAlbedo": "rgba(216, 179, 154, 1.0)", "secondaryAlbedo": "rgba(29, 26, 34, 1.0)", "materialClass": "skin", "materialClassConfidence": 0.85, "evidenceRef": "sprite-front.png#head" }, "actionProfile": { "animationRole": "t\xEAte + masse capillaire" }, "material": "skin" };
  node_head_1.userData.actionProfile = { "animationRole": "t\xEAte + masse capillaire" };
  (nodes["root"] ?? root).add(node_head_1);
  nodes["head"] = node_head_1;
  const mesh_head_1Geometry = endpoint_head_1 ? new THREE.CylinderGeometry(endpoint_head_1.endRadius, endpoint_head_1.baseRadius, endpoint_head_1.length, 32, 12) : new THREE.SphereGeometry(0.5, 64, 40);
  if (!endpoint_head_1) {
    mesh_head_1Geometry.scale(1, 1, 1);
  }
  const mesh_head_1 = new THREE.Mesh(
    mesh_head_1Geometry,
    materialMap["skin"] ?? new THREE.MeshStandardMaterial({ color: 8947848 })
  );
  mesh_head_1.name = "Head & hair";
  if (endpoint_head_1) {
    mesh_head_1.position.copy(endpoint_head_1.midpoint);
    mesh_head_1.quaternion.copy(endpoint_head_1.quaternion);
  }
  mesh_head_1.castShadow = options.castShadow ?? true;
  mesh_head_1.receiveShadow = options.receiveShadow ?? true;
  mesh_head_1.userData.sculptComponent = { "id": "head", "name": "Head & hair", "level": "macro", "role": "head unit", "parent": "root", "importance": 1, "confidence": 0.9, "primitive": "sphere", "topologyClass": "continuous-sculpt", "topologyRationale": "Skull volume continuous; hair as 5-15 clump masses per contract", "geometryDescriptor": {}, "materialRefs": ["skin", "hairDark"], "localFeatures": [{ "id": "eyes-glossy", "note": "glossy spheres + iris disc + catchlight quad toward moonlight key" }, { "id": "hair-silhouette", "note": "tousled fringe, side-swept part, nape volume \u2014 identity feature" }], "colorMaterialRecipe": { "dominantAlbedo": "rgba(216, 179, 154, 1.0)", "secondaryAlbedo": "rgba(29, 26, 34, 1.0)", "materialClass": "skin", "materialClassConfidence": 0.85, "evidenceRef": "sprite-front.png#head" }, "actionProfile": { "animationRole": "t\xEAte + masse capillaire" }, "material": "skin" };
  node_head_1.add(mesh_head_1);
  meshes["head"] = mesh_head_1;
  colliders["head"] = {};
  const attachment_torso_2 = { "parentId": "root", "parentSocket": "pelvis", "localStart": [0, 10.8, 0], "localEnd": [0, 15.5, 0], "contactType": "surface-contact", "embedDepth": 0.02, "gapTolerance": 0.01 };
  const endpoint_torso_2 = makeAttachmentEndpoint(attachment_torso_2);
  const node_torso_2 = new THREE.Group();
  node_torso_2.name = "Torso: cuirass / tabard / belt__pivot";
  node_torso_2.scale.set(1, 1, 1);
  if (endpoint_torso_2) {
    node_torso_2.position.copy(endpoint_torso_2.start);
    node_torso_2.rotation.set(0, 0, 0);
  } else {
    node_torso_2.position.set(0, 0, 0);
    node_torso_2.rotation.set(0, 0, 0);
  }
  node_torso_2.userData.sculptComponent = { "id": "torso", "name": "Torso: cuirass / tabard / belt", "level": "macro", "role": "core mass", "parent": "root", "importance": 1, "confidence": 0.9, "primitive": "cylinder", "topologyClass": "assembled-solid", "topologyRationale": "Segmented plates over cloth; tabard hangs front center", "geometryDescriptor": {}, "materialRefs": [], "localFeatures": [], "colorMaterialRecipe": { "dominantAlbedo": "rgba(138, 143, 150, 1.0)", "secondaryAlbedo": "rgba(42, 53, 80, 1.0)", "materialClass": "metal", "materialClassConfidence": 0.85, "evidenceRef": "sprite-front.png#torso" }, "actionProfile": { "animationRole": "torse cuirass\xE9" }, "attachment": { "parentId": "root", "parentSocket": "pelvis", "localStart": [0, 10.8, 0], "localEnd": [0, 15.5, 0], "contactType": "surface-contact", "embedDepth": 0.02, "gapTolerance": 0.01 }, "material": "steel" };
  node_torso_2.userData.actionProfile = { "animationRole": "torse cuirass\xE9" };
  (nodes["root"] ?? root).add(node_torso_2);
  nodes["torso"] = node_torso_2;
  const mesh_torso_2Geometry = endpoint_torso_2 ? new THREE.CylinderGeometry(endpoint_torso_2.endRadius, endpoint_torso_2.baseRadius, endpoint_torso_2.length, 32, 12) : new THREE.CylinderGeometry(0.5, 0.5, 1, 48, 16);
  if (!endpoint_torso_2) {
    mesh_torso_2Geometry.scale(1, 1, 1);
  }
  const mesh_torso_2 = new THREE.Mesh(
    mesh_torso_2Geometry,
    materialMap["steel"] ?? new THREE.MeshStandardMaterial({ color: 8947848 })
  );
  mesh_torso_2.name = "Torso: cuirass / tabard / belt";
  if (endpoint_torso_2) {
    mesh_torso_2.position.copy(endpoint_torso_2.midpoint);
    mesh_torso_2.quaternion.copy(endpoint_torso_2.quaternion);
  }
  mesh_torso_2.castShadow = options.castShadow ?? true;
  mesh_torso_2.receiveShadow = options.receiveShadow ?? true;
  mesh_torso_2.userData.sculptComponent = { "id": "torso", "name": "Torso: cuirass / tabard / belt", "level": "macro", "role": "core mass", "parent": "root", "importance": 1, "confidence": 0.9, "primitive": "cylinder", "topologyClass": "assembled-solid", "topologyRationale": "Segmented plates over cloth; tabard hangs front center", "geometryDescriptor": {}, "materialRefs": [], "localFeatures": [], "colorMaterialRecipe": { "dominantAlbedo": "rgba(138, 143, 150, 1.0)", "secondaryAlbedo": "rgba(42, 53, 80, 1.0)", "materialClass": "metal", "materialClassConfidence": 0.85, "evidenceRef": "sprite-front.png#torso" }, "actionProfile": { "animationRole": "torse cuirass\xE9" }, "attachment": { "parentId": "root", "parentSocket": "pelvis", "localStart": [0, 10.8, 0], "localEnd": [0, 15.5, 0], "contactType": "surface-contact", "embedDepth": 0.02, "gapTolerance": 0.01 }, "material": "steel" };
  node_torso_2.add(mesh_torso_2);
  meshes["torso"] = mesh_torso_2;
  colliders["torso"] = {};
  const attachment_armR_3 = { "parentId": "root", "parentSocket": "shoulderR", "localStart": [0.9, 14.2, 0], "localEnd": [0.9, 6.2, 0.4], "contactType": "surface-contact", "embedDepth": 0.02, "gapTolerance": 0.01 };
  const endpoint_armR_3 = makeAttachmentEndpoint(attachment_armR_3);
  const node_armR_3 = new THREE.Group();
  node_armR_3.name = "Right arm (sword arm)__pivot";
  node_armR_3.scale.set(1, 1, 1);
  if (endpoint_armR_3) {
    node_armR_3.position.copy(endpoint_armR_3.start);
    node_armR_3.rotation.set(0, 0, 0);
  } else {
    node_armR_3.position.set(0, 0, 0);
    node_armR_3.rotation.set(0, 0, 0);
  }
  node_armR_3.userData.sculptComponent = { "id": "armR", "name": "Right arm (sword arm)", "level": "macro", "role": "limb", "parent": "root", "importance": 1, "confidence": 0.9, "primitive": "capsule", "topologyClass": "assembled-solid", "topologyRationale": "rerebrace/couter/gauntlet lames follow elbow flexion", "geometryDescriptor": {}, "materialRefs": [], "localFeatures": [], "colorMaterialRecipe": { "dominantAlbedo": "rgba(138, 143, 150, 1.0)", "secondaryAlbedo": "rgba(201, 160, 78, 1.0)", "materialClass": "metal", "materialClassConfidence": 0.85, "evidenceRef": "sprite-front.png#armR" }, "actionProfile": { "animationRole": "bras d'arme" }, "attachment": { "parentId": "root", "parentSocket": "shoulderR", "localStart": [0.9, 14.2, 0], "localEnd": [0.9, 6.2, 0.4], "contactType": "surface-contact", "embedDepth": 0.02, "gapTolerance": 0.01 }, "material": "steel" };
  node_armR_3.userData.actionProfile = { "animationRole": "bras d'arme" };
  (nodes["root"] ?? root).add(node_armR_3);
  nodes["armR"] = node_armR_3;
  const mesh_armR_3Geometry = endpoint_armR_3 ? new THREE.CylinderGeometry(endpoint_armR_3.endRadius, endpoint_armR_3.baseRadius, endpoint_armR_3.length, 32, 12) : buildWatertightCapsule(0.35, 0.7, 16, 32, 1);
  if (!endpoint_armR_3) {
    mesh_armR_3Geometry.scale(1, 1, 1);
  }
  const mesh_armR_3 = new THREE.Mesh(
    mesh_armR_3Geometry,
    materialMap["steel"] ?? new THREE.MeshStandardMaterial({ color: 8947848 })
  );
  mesh_armR_3.name = "Right arm (sword arm)";
  if (endpoint_armR_3) {
    mesh_armR_3.position.copy(endpoint_armR_3.midpoint);
    mesh_armR_3.quaternion.copy(endpoint_armR_3.quaternion);
  }
  mesh_armR_3.castShadow = options.castShadow ?? true;
  mesh_armR_3.receiveShadow = options.receiveShadow ?? true;
  mesh_armR_3.userData.sculptComponent = { "id": "armR", "name": "Right arm (sword arm)", "level": "macro", "role": "limb", "parent": "root", "importance": 1, "confidence": 0.9, "primitive": "capsule", "topologyClass": "assembled-solid", "topologyRationale": "rerebrace/couter/gauntlet lames follow elbow flexion", "geometryDescriptor": {}, "materialRefs": [], "localFeatures": [], "colorMaterialRecipe": { "dominantAlbedo": "rgba(138, 143, 150, 1.0)", "secondaryAlbedo": "rgba(201, 160, 78, 1.0)", "materialClass": "metal", "materialClassConfidence": 0.85, "evidenceRef": "sprite-front.png#armR" }, "actionProfile": { "animationRole": "bras d'arme" }, "attachment": { "parentId": "root", "parentSocket": "shoulderR", "localStart": [0.9, 14.2, 0], "localEnd": [0.9, 6.2, 0.4], "contactType": "surface-contact", "embedDepth": 0.02, "gapTolerance": 0.01 }, "material": "steel" };
  node_armR_3.add(mesh_armR_3);
  meshes["armR"] = mesh_armR_3;
  colliders["armR"] = {};
  const attachment_armL_4 = { "parentId": "root", "parentSocket": "shoulderL", "localStart": [-0.9, 14.2, 0], "localEnd": [-0.9, 7.5, 1.6], "contactType": "surface-contact", "embedDepth": 0.02, "gapTolerance": 0.01 };
  const endpoint_armL_4 = makeAttachmentEndpoint(attachment_armL_4);
  const node_armL_4 = new THREE.Group();
  node_armL_4.name = "Left arm (shield arm)__pivot";
  node_armL_4.scale.set(1, 1, 1);
  if (endpoint_armL_4) {
    node_armL_4.position.copy(endpoint_armL_4.start);
    node_armL_4.rotation.set(0, 0, 0);
  } else {
    node_armL_4.position.set(0, 0, 0);
    node_armL_4.rotation.set(0, 0, 0);
  }
  node_armL_4.userData.sculptComponent = { "id": "armL", "name": "Left arm (shield arm)", "level": "macro", "role": "limb", "parent": "root", "importance": 1, "confidence": 0.9, "primitive": "capsule", "topologyClass": "assembled-solid", "topologyRationale": "raised forward carrying heater shield", "geometryDescriptor": {}, "materialRefs": [], "localFeatures": [], "colorMaterialRecipe": { "dominantAlbedo": "rgba(138, 143, 150, 1.0)", "secondaryAlbedo": "rgba(201, 160, 78, 1.0)", "materialClass": "metal", "materialClassConfidence": 0.85, "evidenceRef": "sprite-front.png#armL" }, "actionProfile": { "animationRole": "bras de bouclier" }, "attachment": { "parentId": "root", "parentSocket": "shoulderL", "localStart": [-0.9, 14.2, 0], "localEnd": [-0.9, 7.5, 1.6], "contactType": "surface-contact", "embedDepth": 0.02, "gapTolerance": 0.01 }, "material": "steel" };
  node_armL_4.userData.actionProfile = { "animationRole": "bras de bouclier" };
  (nodes["root"] ?? root).add(node_armL_4);
  nodes["armL"] = node_armL_4;
  const mesh_armL_4Geometry = endpoint_armL_4 ? new THREE.CylinderGeometry(endpoint_armL_4.endRadius, endpoint_armL_4.baseRadius, endpoint_armL_4.length, 32, 12) : buildWatertightCapsule(0.35, 0.7, 16, 32, 1);
  if (!endpoint_armL_4) {
    mesh_armL_4Geometry.scale(1, 1, 1);
  }
  const mesh_armL_4 = new THREE.Mesh(
    mesh_armL_4Geometry,
    materialMap["steel"] ?? new THREE.MeshStandardMaterial({ color: 8947848 })
  );
  mesh_armL_4.name = "Left arm (shield arm)";
  if (endpoint_armL_4) {
    mesh_armL_4.position.copy(endpoint_armL_4.midpoint);
    mesh_armL_4.quaternion.copy(endpoint_armL_4.quaternion);
  }
  mesh_armL_4.castShadow = options.castShadow ?? true;
  mesh_armL_4.receiveShadow = options.receiveShadow ?? true;
  mesh_armL_4.userData.sculptComponent = { "id": "armL", "name": "Left arm (shield arm)", "level": "macro", "role": "limb", "parent": "root", "importance": 1, "confidence": 0.9, "primitive": "capsule", "topologyClass": "assembled-solid", "topologyRationale": "raised forward carrying heater shield", "geometryDescriptor": {}, "materialRefs": [], "localFeatures": [], "colorMaterialRecipe": { "dominantAlbedo": "rgba(138, 143, 150, 1.0)", "secondaryAlbedo": "rgba(201, 160, 78, 1.0)", "materialClass": "metal", "materialClassConfidence": 0.85, "evidenceRef": "sprite-front.png#armL" }, "actionProfile": { "animationRole": "bras de bouclier" }, "attachment": { "parentId": "root", "parentSocket": "shoulderL", "localStart": [-0.9, 14.2, 0], "localEnd": [-0.9, 7.5, 1.6], "contactType": "surface-contact", "embedDepth": 0.02, "gapTolerance": 0.01 }, "material": "steel" };
  node_armL_4.add(mesh_armL_4);
  meshes["armL"] = mesh_armL_4;
  colliders["armL"] = {};
  const attachment_legs_5 = { "parentId": "root", "parentSocket": "pelvis", "localStart": [0, 10.6, 0], "localEnd": [0, 0.4, 0], "contactType": "surface-contact", "embedDepth": 0.02, "gapTolerance": 0.01 };
  const endpoint_legs_5 = makeAttachmentEndpoint(attachment_legs_5);
  const node_legs_5 = new THREE.Group();
  node_legs_5.name = "Legs (fauld \u2192 sabatons)__pivot";
  node_legs_5.scale.set(1, 1, 1);
  if (endpoint_legs_5) {
    node_legs_5.position.copy(endpoint_legs_5.start);
    node_legs_5.rotation.set(0, 0, 0);
  } else {
    node_legs_5.position.set(0, 0, 0);
    node_legs_5.rotation.set(0, 0, 0);
  }
  node_legs_5.userData.sculptComponent = { "id": "legs", "name": "Legs (fauld \u2192 sabatons)", "level": "macro", "role": "lower body", "parent": "root", "importance": 1, "confidence": 0.9, "primitive": "capsule", "topologyClass": "assembled-solid", "topologyRationale": "layered cuisse/poleyn/greave per leg", "geometryDescriptor": {}, "materialRefs": [], "localFeatures": [], "colorMaterialRecipe": { "dominantAlbedo": "rgba(138, 143, 150, 1.0)", "secondaryAlbedo": "rgba(90, 70, 50, 1.0)", "materialClass": "metal", "materialClassConfidence": 0.85, "evidenceRef": "sprite-front.png#legs" }, "actionProfile": { "animationRole": "jambes blind\xE9es" }, "attachment": { "parentId": "root", "parentSocket": "pelvis", "localStart": [0, 10.6, 0], "localEnd": [0, 0.4, 0], "contactType": "surface-contact", "embedDepth": 0.02, "gapTolerance": 0.01 }, "material": "steel" };
  node_legs_5.userData.actionProfile = { "animationRole": "jambes blind\xE9es" };
  (nodes["root"] ?? root).add(node_legs_5);
  nodes["legs"] = node_legs_5;
  const mesh_legs_5Geometry = endpoint_legs_5 ? new THREE.CylinderGeometry(endpoint_legs_5.endRadius, endpoint_legs_5.baseRadius, endpoint_legs_5.length, 32, 12) : buildWatertightCapsule(0.35, 0.7, 16, 32, 1);
  if (!endpoint_legs_5) {
    mesh_legs_5Geometry.scale(1, 1, 1);
  }
  const mesh_legs_5 = new THREE.Mesh(
    mesh_legs_5Geometry,
    materialMap["steel"] ?? new THREE.MeshStandardMaterial({ color: 8947848 })
  );
  mesh_legs_5.name = "Legs (fauld \u2192 sabatons)";
  if (endpoint_legs_5) {
    mesh_legs_5.position.copy(endpoint_legs_5.midpoint);
    mesh_legs_5.quaternion.copy(endpoint_legs_5.quaternion);
  }
  mesh_legs_5.castShadow = options.castShadow ?? true;
  mesh_legs_5.receiveShadow = options.receiveShadow ?? true;
  mesh_legs_5.userData.sculptComponent = { "id": "legs", "name": "Legs (fauld \u2192 sabatons)", "level": "macro", "role": "lower body", "parent": "root", "importance": 1, "confidence": 0.9, "primitive": "capsule", "topologyClass": "assembled-solid", "topologyRationale": "layered cuisse/poleyn/greave per leg", "geometryDescriptor": {}, "materialRefs": [], "localFeatures": [], "colorMaterialRecipe": { "dominantAlbedo": "rgba(138, 143, 150, 1.0)", "secondaryAlbedo": "rgba(90, 70, 50, 1.0)", "materialClass": "metal", "materialClassConfidence": 0.85, "evidenceRef": "sprite-front.png#legs" }, "actionProfile": { "animationRole": "jambes blind\xE9es" }, "attachment": { "parentId": "root", "parentSocket": "pelvis", "localStart": [0, 10.6, 0], "localEnd": [0, 0.4, 0], "contactType": "surface-contact", "embedDepth": 0.02, "gapTolerance": 0.01 }, "material": "steel" };
  node_legs_5.add(mesh_legs_5);
  meshes["legs"] = mesh_legs_5;
  colliders["legs"] = {};
  const endpoint_gear_6 = makeAttachmentEndpoint(null);
  const node_gear_6 = new THREE.Group();
  node_gear_6.name = "Cape & shield group__pivot";
  node_gear_6.scale.set(1, 1, 1);
  if (endpoint_gear_6) {
    node_gear_6.position.copy(endpoint_gear_6.start);
    node_gear_6.rotation.set(0, 0, 0);
  } else {
    node_gear_6.position.set(0, 0, 0);
    node_gear_6.rotation.set(0, 0, 0);
  }
  node_gear_6.userData.sculptComponent = { "id": "gear", "name": "Cape & shield group", "level": "macro", "role": "worn gear", "parent": "root", "importance": 1, "confidence": 0.9, "primitive": "extrude", "topologyClass": "assembled-solid", "topologyRationale": "cape = torn cloth panels; shield = curved heater plate", "geometryDescriptor": {}, "materialRefs": [], "localFeatures": [], "colorMaterialRecipe": { "dominantAlbedo": "rgba(42, 53, 80, 1.0)", "secondaryAlbedo": "rgba(201, 160, 78, 1.0)", "materialClass": "fabric", "materialClassConfidence": 0.85, "evidenceRef": "sprite-front.png#gear" }, "actionProfile": { "animationRole": "cape + bouclier" }, "attachment": { "parentId": "root", "parentSocket": "back", "localStart": [0, 13.5, -0.5], "localEnd": [0, 2, -1.2], "contactType": "drape-contact", "embedDepth": 0.02, "gapTolerance": 0.01 }, "material": "clothNavy" };
  node_gear_6.userData.actionProfile = { "animationRole": "cape + bouclier" };
  (nodes["root"] ?? root).add(node_gear_6);
  nodes["gear"] = node_gear_6;
  const mesh_gear_6Geometry = endpoint_gear_6 ? new THREE.CylinderGeometry(endpoint_gear_6.endRadius, endpoint_gear_6.baseRadius, endpoint_gear_6.length, 32, 12) : buildExtrudeGeometry({ "points": [[-0.3, -0.3], [0.3, -0.3], [0.3, 0.3], [-0.3, 0.3]], "depth": 0.1 });
  if (!endpoint_gear_6) {
    mesh_gear_6Geometry.scale(1, 1, 1);
  }
  const mesh_gear_6 = new THREE.Mesh(
    mesh_gear_6Geometry,
    materialMap["clothNavy"] ?? new THREE.MeshStandardMaterial({ color: 8947848 })
  );
  mesh_gear_6.name = "Cape & shield group";
  if (endpoint_gear_6) {
    mesh_gear_6.position.copy(endpoint_gear_6.midpoint);
    mesh_gear_6.quaternion.copy(endpoint_gear_6.quaternion);
  }
  mesh_gear_6.castShadow = options.castShadow ?? true;
  mesh_gear_6.receiveShadow = options.receiveShadow ?? true;
  mesh_gear_6.userData.sculptComponent = { "id": "gear", "name": "Cape & shield group", "level": "macro", "role": "worn gear", "parent": "root", "importance": 1, "confidence": 0.9, "primitive": "extrude", "topologyClass": "assembled-solid", "topologyRationale": "cape = torn cloth panels; shield = curved heater plate", "geometryDescriptor": {}, "materialRefs": [], "localFeatures": [], "colorMaterialRecipe": { "dominantAlbedo": "rgba(42, 53, 80, 1.0)", "secondaryAlbedo": "rgba(201, 160, 78, 1.0)", "materialClass": "fabric", "materialClassConfidence": 0.85, "evidenceRef": "sprite-front.png#gear" }, "actionProfile": { "animationRole": "cape + bouclier" }, "attachment": { "parentId": "root", "parentSocket": "back", "localStart": [0, 13.5, -0.5], "localEnd": [0, 2, -1.2], "contactType": "drape-contact", "embedDepth": 0.02, "gapTolerance": 0.01 }, "material": "clothNavy" };
  node_gear_6.add(mesh_gear_6);
  meshes["gear"] = mesh_gear_6;
  colliders["gear"] = {};
  root.userData.sculptRuntime = { nodes, meshes, sockets, colliders, destructionGroups };
  root.userData.lookDevTargets = { "qualityPriority": "reference-fidelity", "materialPass": { "albedoPaletteRequired": true, "roughnessVariationRequired": true, "normalOrBumpRequired": true, "localOverridesRequired": true, "minimumTextureResolution": 1024, "preferredTextureResolution": 2048, "independentMapChannels": ["albedo", "roughness", "height", "normal", "ambient-occlusion"], "requiredSurfaceFrequencyBands": ["macro", "meso", "micro"], "geometryReliefRequiredWhenSilhouetteAffected": true, "referencePbrExtraction": { "requiredWhenSourceImagePresent": true, "targetThreshold": 0.7, "stopOnLowConfidence": true, "script": "forge/stage1_intake/extract_pbr_evidence.py", "acceptedLimitation": "single-image extraction is reference-derived inference, not exact photogrammetry" }, "mustAvoid": ["single flat albedo per material", "uniform roughness", "albedo texture reused as roughness/height/normal/AO", "single-frequency random noise", "plastic-looking smooth bark, stone, cloth, foliage, or aged material", "local color/detail described only in prose without material masks", "claiming exact PBR recovery when confidence is below the target threshold"] }, "lightingPass": { "requiredTerms": ["key light", "fill light", "rim or environment light", "exposure", "tone mapping", "background", "contact shadow"], "mustAvoid": ["ambient-only lighting", "flat value range", "missing contact shadow", "reference lighting copied without separating material readability"] }, "screenshotReview": ["Compare albedo palette and local color zones.", "Compare roughness/normal/bump response under light.", "Compare cavity dirt, edge wear, stains, moss, scratches, or other local masks.", "Compare key/fill/rim structure, exposure, tone mapping, background, and contact shadows.", "Capture a neutral-light render to verify material readability without reference lighting.", "Capture a grazing-light close-up to expose flat normals, uniform roughness, tiling, and plastic highlights.", "Capture a reference-matched render from the same camera framing as the source."] };
  root.userData.actionReadiness = {
    note: "Use root.userData.sculptRuntime.nodes for transforms, sockets for attachments, colliders for physics proxies, and destructionGroups for breakable sets."
  };
  return root;
}
function createAldarionKnightLookDevLights(mode = "neutral") {
  const lights = new THREE.Group();
  lights.name = "Aldarion Knight look-dev lights";
  const hemi = new THREE.HemisphereLight(
    mode === "reference" ? 16773334 : 15922431,
    3554114,
    mode === "grazing" ? 0.28 : mode === "reference" ? 0.72 : 0.85
  );
  lights.add(hemi);
  const key = new THREE.DirectionalLight(
    mode === "reference" ? 16764810 : 16774376,
    mode === "grazing" ? 4.2 : mode === "reference" ? 2.6 : 2.15
  );
  if (mode === "grazing") key.position.set(7.5, 1.1, 4);
  else if (mode === "reference") key.position.set(-4.5, 7.5, 5);
  else key.position.set(-4, 6, 5.5);
  key.castShadow = true;
  key.shadow.mapSize.set(4096, 4096);
  key.shadow.bias = -25e-5;
  key.shadow.normalBias = 0.018;
  key.shadow.radius = 7;
  key.shadow.blurSamples = 24;
  key.shadow.camera.near = 0.5;
  key.shadow.camera.far = 30;
  key.shadow.camera.left = -2.6;
  key.shadow.camera.right = 2.6;
  key.shadow.camera.top = 2.6;
  key.shadow.camera.bottom = -2.6;
  key.shadow.camera.updateProjectionMatrix();
  lights.add(key);
  const fill = new THREE.DirectionalLight(11060479, mode === "grazing" ? 0.12 : 0.42);
  fill.position.set(4, 3, 3.5);
  lights.add(fill);
  const rim = new THREE.DirectionalLight(16773572, mode === "grazing" ? 0.28 : 0.85);
  rim.position.set(0.5, 4.5, -6);
  lights.add(rim);
  lights.userData.reviewMode = mode;
  lights.userData.lightingFromPhoto = [{ "role": "key", "note": "cool moonlight key from upper-left (moon disc position), exposure anchored so steel highlights sit near 0.9" }, { "role": "fill", "note": "warm bounce from below-right (torch line), low intensity, tone kept filmic" }, { "role": "rim", "note": "violet rim from sky dome opposite the key, separates cape from background" }, { "toneMapping": "uncharted2 filmic grade via evolved SPEAR kernel (uncharted2_tonemap fast slot)" }, { "contactShadow": "contact shadow under sabatons and shield bottom edge" }];
  lights.userData.lookDevTargets = { "qualityPriority": "reference-fidelity", "materialPass": { "albedoPaletteRequired": true, "roughnessVariationRequired": true, "normalOrBumpRequired": true, "localOverridesRequired": true, "minimumTextureResolution": 1024, "preferredTextureResolution": 2048, "independentMapChannels": ["albedo", "roughness", "height", "normal", "ambient-occlusion"], "requiredSurfaceFrequencyBands": ["macro", "meso", "micro"], "geometryReliefRequiredWhenSilhouetteAffected": true, "referencePbrExtraction": { "requiredWhenSourceImagePresent": true, "targetThreshold": 0.7, "stopOnLowConfidence": true, "script": "forge/stage1_intake/extract_pbr_evidence.py", "acceptedLimitation": "single-image extraction is reference-derived inference, not exact photogrammetry" }, "mustAvoid": ["single flat albedo per material", "uniform roughness", "albedo texture reused as roughness/height/normal/AO", "single-frequency random noise", "plastic-looking smooth bark, stone, cloth, foliage, or aged material", "local color/detail described only in prose without material masks", "claiming exact PBR recovery when confidence is below the target threshold"] }, "lightingPass": { "requiredTerms": ["key light", "fill light", "rim or environment light", "exposure", "tone mapping", "background", "contact shadow"], "mustAvoid": ["ambient-only lighting", "flat value range", "missing contact shadow", "reference lighting copied without separating material readability"] }, "screenshotReview": ["Compare albedo palette and local color zones.", "Compare roughness/normal/bump response under light.", "Compare cavity dirt, edge wear, stains, moss, scratches, or other local masks.", "Compare key/fill/rim structure, exposure, tone mapping, background, and contact shadows.", "Capture a neutral-light render to verify material readability without reference lighting.", "Capture a grazing-light close-up to expose flat normals, uniform roughness, tiling, and plastic highlights.", "Capture a reference-matched render from the same camera framing as the source."] };
  return lights;
}
function createAldarionKnightEnvironment(renderer) {
  const pmrem = new THREE.PMREMGenerator(renderer);
  const texture = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
  pmrem.dispose();
  return texture;
}
function frameAldarionKnightCamera(camera, object, options = {}) {
  const box = new THREE.Box3().setFromObject(object);
  if (box.isEmpty()) return;
  const size = box.getSize(new THREE.Vector3());
  const center = box.getCenter(new THREE.Vector3());
  const margin = options.margin ?? 1.15;
  const maxDim = Math.max(size.x, size.y, size.z) * margin;
  const fov = camera.fov * Math.PI / 180;
  const distance = maxDim / 2 / Math.tan(fov / 2);
  const az = (options.azimuthDeg ?? 0) * Math.PI / 180;
  const el = (options.elevationDeg ?? 0) * Math.PI / 180;
  const dir = new THREE.Vector3(
    Math.sin(az) * Math.cos(el),
    Math.sin(el),
    Math.cos(az) * Math.cos(el)
  );
  camera.position.copy(center).addScaledVector(dir, distance);
  camera.near = Math.max(0.01, distance - maxDim);
  camera.far = distance + maxDim * 2;
  camera.lookAt(center);
  camera.updateProjectionMatrix();
}
function createAldarionKnightPresentationComposer(renderer, scene, camera, options = {}) {
  const composer = new EffectComposer(renderer);
  composer.addPass(new RenderPass(scene, camera));
  if (options.dof) {
    composer.addPass(new BokehPass(scene, camera, {
      focus: options.dofFocus ?? 10,
      aperture: options.dofAperture ?? 2e-4,
      maxblur: 0.01
    }));
  }
  if (options.bloom) {
    const size = new THREE.Vector2();
    renderer.getSize(size);
    composer.addPass(new UnrealBloomPass(size, options.bloomStrength ?? 0.4, 0.4, 0.85));
  }
  return composer;
}
function configureAldarionKnightRenderer(renderer) {
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.outputColorSpace = THREE.SRGBColorSpace;
}
function createAldarionKnightInspectControls(camera, domElement) {
  const controls = new OrbitControls(camera, domElement);
  controls.enableDamping = true;
  controls.minDistance = 1;
  controls.maxDistance = 8;
  controls.autoRotate = false;
  return controls;
}
export {
  configureAldarionKnightRenderer,
  createAldarionKnightEnvironment,
  createAldarionKnightInspectControls,
  createAldarionKnightLookDevLights,
  createAldarionKnightModel,
  createAldarionKnightPresentationComposer,
  frameAldarionKnightCamera
};
