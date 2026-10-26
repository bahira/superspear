// Small, dependency-free helpers shared by the bot's tools.

import { randomUUID } from "node:crypto";

/** A promise plus a way to resolve it once, from anywhere. */
export function deferred() {
  let resolve;
  let reject;
  const promise = new Promise((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

export function newJobId() {
  return `job_${randomUUID().replaceAll("-", "").slice(0, 20)}`;
}

/**
 * A page the bot may open: http or https, no credentials, no fragment. This mirrors what
 * Pomerado itself accepts, so a bad URL fails here with a readable message instead of
 * twenty seconds into a browser session.
 */
export function checkUrl(value, field = "url") {
  if (typeof value !== "string" || value.trim() === "") {
    throw new InputError(`"${field}" is required and must be an http(s) URL.`);
  }
  let url;
  try {
    url = new URL(value.trim());
  } catch {
    throw new InputError(`"${field}" is not a valid URL: ${value}`);
  }
  if (url.protocol !== "http:" && url.protocol !== "https:") {
    throw new InputError(`"${field}" must use http or https.`);
  }
  if (url.username !== "" || url.password !== "") {
    throw new InputError(`"${field}" must not carry credentials; sign in through the site.`);
  }
  if (url.hash !== "") throw new InputError(`"${field}" must not carry a #fragment.`);
  return url.toString();
}

/**
 * An integration folder name derived from a site: lower case letters, digits and
 * underscores, starting with a letter, at most 60 characters. Pomerado enforces the same
 * shape, and a stable slug is what lets a second call to the same site reuse the first
 * integration instead of minting another one.
 */
export function siteSlug(rawUrl) {
  const host = new URL(checkUrl(rawUrl)).hostname.toLowerCase();
  let slug = host
    .replace(/^www\./, "")
    .replace(/[^a-z0-9]+/g, "_")
    .replace(/^_+|_+$/g, "")
    .slice(0, 60)
    .replace(/_+$/g, "");
  if (!/^[a-z]/.test(slug)) slug = `site_${slug}`;
  if (slug === "" || /^[^a-z]/.test(slug)) slug = "site";
  // Names Pomerado's own MCP servers reserve for their helper tools.
  if (["mint", "get_job", "provide_input", "cancel_job"].includes(slug)) slug = `${slug}_site`;
  return slug;
}

/** A name the caller supplied, checked before it reaches the file system. */
export function checkIntegrationName(value) {
  if (typeof value !== "string" || value.trim() === "") {
    throw new InputError('"integration" is required.');
  }
  const name = value.trim();
  if (!/^[a-z][a-z0-9_]{0,59}$/.test(name)) {
    throw new InputError(
      `"integration" must start with a lowercase letter and use only lowercase letters, digits and underscores (got ${name}).`,
    );
  }
  return name;
}

export function readString(value, field, { max = 4000, required = true } = {}) {
  if (value === undefined || value === null || value === "") {
    if (required) throw new InputError(`"${field}" is required.`);
    return undefined;
  }
  if (typeof value !== "string") throw new InputError(`"${field}" must be a string.`);
  const text = value.trim();
  if (text === "" && required) throw new InputError(`"${field}" must not be empty.`);
  if (text.length > max) throw new InputError(`"${field}" is longer than ${max} characters.`);
  return text;
}

export function readBoolean(value, field, fallback = false) {
  if (value === undefined || value === null) return fallback;
  if (typeof value !== "boolean") throw new InputError(`"${field}" must be a boolean.`);
  return value;
}

export function readObject(value, field) {
  if (value === undefined || value === null) return {};
  if (typeof value !== "object" || Array.isArray(value)) {
    throw new InputError(`"${field}" must be a JSON object.`);
  }
  return value;
}

export function readSeconds(value, fallback) {
  if (value === undefined || value === null) return fallback;
  const seconds = Number(value);
  if (!Number.isFinite(seconds) || seconds < 0 || seconds > 30) {
    throw new InputError('"wait_seconds" must be a number between 0 and 30.');
  }
  return Math.round(seconds * 1000);
}

/** A failure caused by what the caller sent, not by the bot or the website. */
export class InputError extends Error {
  constructor(message) {
    super(message);
    this.name = "InputError";
  }
}

/** A failure the agent can act on, with a stable code. */
export class BotError extends Error {
  constructor(code, message, cause) {
    super(message, cause === undefined ? undefined : { cause });
    this.name = "BotError";
    this.code = code;
  }
}

/** Parse the JSON text an MCP tool answered with, preferring structured content. */
export function parseToolResult(result) {
  if (result === undefined || result === null) return undefined;
  if (result.structuredContent !== undefined && result.structuredContent !== null) {
    const structured = result.structuredContent;
    // A single free-form array or scalar arrives wrapped as { result }.
    if (
      typeof structured === "object" &&
      !Array.isArray(structured) &&
      Object.keys(structured).length === 1 &&
      "result" in structured &&
      !("status" in structured) &&
      !("job_id" in structured)
    ) {
      return structured.result;
    }
    return structured;
  }
  const text = Array.isArray(result.content)
    ? result.content.filter((part) => part?.type === "text").map((part) => part.text).join("\n")
    : "";
  if (text.trim() === "") return undefined;
  try {
    return JSON.parse(text);
  } catch {
    return text;
  }
}

/** The first readable message inside an MCP tool result or a thrown error. */
export function toolErrorMessage(result, fallback) {
  if (result?.isError === true) {
    const text = Array.isArray(result.content)
      ? result.content.filter((part) => part?.type === "text").map((part) => part.text).join(" ").trim()
      : "";
    return text === "" ? fallback : text;
  }
  return fallback;
}
