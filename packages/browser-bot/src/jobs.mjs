// Long-running work, exposed the way an MCP client expects it.
//
// A mint can take twenty minutes and a run can wait on a question, so a tool call returns a
// job id instead of blocking. `get_job` observes a job without ever restarting it, which is
// what keeps a write from being sent to a website twice.

import { BotError, InputError, deferred, newJobId } from "./values.mjs";

const TERMINAL_RETENTION_MS = 15 * 60_000;
const MAX_STEPS = 24;
const ACTIVE = new Set(["running", "input_required"]);

function delay(ms) {
  return new Promise((resolve) => {
    const timer = setTimeout(resolve, ms);
    timer.unref?.();
  });
}

export class JobStore {
  #jobs = new Map();
  #maxJobs;

  constructor({ maxJobs = 2 } = {}) {
    if (!Number.isInteger(maxJobs) || maxJobs < 1) throw new Error("maxJobs must be >= 1");
    this.#maxJobs = maxJobs;
  }

  get activeCount() {
    return [...this.#jobs.values()].filter((job) => ACTIVE.has(job.status)).length;
  }

  /**
   * Start a job and return its view at once. `execute` receives the job's context and runs
   * in the background; its return value becomes the job's output.
   */
  start({ kind, label, execute }) {
    this.#sweep();
    if (this.activeCount >= this.#maxJobs) {
      throw new BotError(
        "busy",
        `The bot already runs ${this.#maxJobs} job(s). Follow them with get_job, answer them with provide_input, or cancel one with cancel_job.`,
      );
    }
    const job = {
      id: newJobId(),
      kind,
      label,
      status: "running",
      createdAt: Date.now(),
      updatedAt: Date.now(),
      steps: [],
      pending: undefined,
      output: undefined,
      integration: undefined,
      error: undefined,
      errorCode: undefined,
      child: undefined,
      childJobId: undefined,
      cancelled: false,
      signal: deferred(),
    };
    this.#jobs.set(job.id, job);
    job.run = this.#run(job, execute);
    return view(job);
  }

  async #run(job, execute) {
    try {
      const output = await execute(context(job));
      if (job.cancelled) return;
      job.output = output?.output;
      job.integration = output?.integration;
      job.status = "completed";
    } catch (cause) {
      if (job.cancelled || cause?.code === "cancelled") {
        job.status = "cancelled";
        job.error = undefined;
      } else {
        job.status = "failed";
        job.error = cause?.message ?? String(cause);
        job.errorCode = cause?.code;
      }
    } finally {
      const child = job.child;
      job.child = undefined;
      if (child !== undefined) await child.close().catch(() => {});
      job.updatedAt = Date.now();
      job.pending = undefined;
      this.#signal(job);
    }
  }

  #signal(job) {
    const previous = job.signal;
    job.signal = deferred();
    job.updatedAt = Date.now();
    previous.resolve();
  }

  #require(id) {
    const job = typeof id === "string" ? this.#jobs.get(id) : undefined;
    if (job === undefined) {
      throw new BotError(
        "unknown_job",
        "Unknown or expired job. Jobs only live in this bot's process, and a restart ends them.",
      );
    }
    return job;
  }

  #sweep() {
    const now = Date.now();
    for (const [id, job] of this.#jobs) {
      if (ACTIVE.has(job.status)) continue;
      if (now - job.updatedAt > TERMINAL_RETENTION_MS) this.#jobs.delete(id);
    }
  }

  /** Observe a job, waiting up to `waitMs` for it to change. Never restarts work. */
  async get(id, waitMs = 20_000) {
    this.#sweep();
    const deadline = Date.now() + Math.max(0, waitMs);
    for (;;) {
      const job = this.#require(id);
      if (job.status !== "running" || job.pending !== undefined) return view(job);
      const remaining = deadline - Date.now();
      if (remaining <= 0) return view(job);
      await Promise.race([job.signal.promise, delay(remaining)]);
    }
  }

  /** Answer the questions a job is waiting on, and hand them to the child unchanged. */
  provideInput(id, requestId, answers) {
    const job = this.#require(id);
    if (job.pending === undefined) {
      throw new BotError(
        "stale_input",
        "This job is not waiting for an answer. Use get_job to see its current state.",
      );
    }
    if (job.pending.id !== requestId) {
      throw new BotError(
        "stale_input",
        `This job waits on request ${job.pending.id}, not ${String(requestId)}. Use get_job to read the pending questions.`,
      );
    }
    if (answers === null || typeof answers !== "object" || Array.isArray(answers)) {
      throw new InputError('"answers" must be an object keyed by question id.');
    }
    const pending = job.pending;
    job.pending = undefined;
    job.status = "running";
    this.#signal(job);
    pending.answer.resolve(answers);
    return view(job);
  }

  /** Stop a job, its child process and the browser that child owns. */
  async cancel(id) {
    const job = this.#require(id);
    if (!ACTIVE.has(job.status)) return view(job);
    job.cancelled = true;
    job.status = "cancelled";
    const pending = job.pending;
    job.pending = undefined;
    if (pending !== undefined) {
      const error = new Error("The job was cancelled.");
      error.code = "cancelled";
      pending.answer.reject(error);
    }
    this.#signal(job);
    if (job.child !== undefined && job.childJobId !== undefined && !job.child.closed) {
      await job.child
        .call("cancel_job", { job_id: job.childJobId }, { timeoutMs: 20_000 })
        .catch(() => {});
    }
    const child = job.child;
    job.child = undefined;
    if (child !== undefined) await child.close().catch(() => {});
    return view(job);
  }

  list() {
    this.#sweep();
    return [...this.#jobs.values()].map(view);
  }

  /** Stop every job at process exit, so no browser outlives the bot. */
  async cancelAll() {
    const active = [...this.#jobs.values()].filter((job) => ACTIVE.has(job.status));
    await Promise.all(active.map((job) => this.cancel(job.id).catch(() => {})));
  }
}

/** The object an executor works with. */
function context(job) {
  return {
    job,
    progress(message) {
      if (job.cancelled) return;
      job.steps.push({ at: new Date().toISOString(), message: String(message).slice(0, 400) });
      if (job.steps.length > MAX_STEPS) job.steps.shift();
      job.updatedAt = Date.now();
    },
    attachChild(child) {
      job.child = child;
    },
    /** Hand the current child back, so a flow can close it before starting the next one. */
    detachChild() {
      const child = job.child;
      job.child = undefined;
      job.childJobId = undefined;
      return child;
    },
    setChildJob(childJobId) {
      job.childJobId = childJobId;
    },
    setIntegration(integration) {
      job.integration = integration;
    },
    /**
     * Publish a child's pending questions and wait for the agent's answer. A cancelled job
     * rejects at once, so a cancel never leaves a promise hanging on a closed browser.
     */
    async ask(pendingInput, { timeoutMs = 10 * 60_000 } = {}) {
      if (job.cancelled) {
        const error = new Error("The job was cancelled.");
        error.code = "cancelled";
        throw error;
      }
      const answer = deferred();
      job.pending = { id: pendingInput.id, request: pendingInput, answer };
      job.status = "input_required";
      job.updatedAt = Date.now();
      job.signal.resolve();
      const timer = setTimeout(() => {
        if (job.pending?.answer === answer) {
          job.pending = undefined;
          job.status = "running";
          const error = new Error("The questions expired before they were answered.");
          error.code = "input_expired";
          answer.reject(error);
        }
      }, timeoutMs);
      timer.unref?.();
      try {
        return await answer.promise;
      } finally {
        clearTimeout(timer);
        if (job.pending?.answer === answer) {
          job.pending = undefined;
          if (!job.cancelled) job.status = "running";
          job.updatedAt = Date.now();
          job.signal.resolve();
        }
      }
    },
  };
}

/** The JSON an agent sees for a job: no source, no saved answers, no key. */
export function view(job) {
  const base = {
    job_id: job.id,
    kind: job.kind,
    label: job.label,
    status: job.status,
    created_at: new Date(job.createdAt).toISOString(),
    updated_at: new Date(job.updatedAt).toISOString(),
    elapsed_ms: job.updatedAt - job.createdAt,
    steps: job.steps.map((step) => step.message).slice(-8),
  };
  if (job.status === "input_required" && job.pending !== undefined) {
    return {
      ...base,
      pending_input: { ...job.pending.request, id: job.pending.id },
      next: {
        tool: "provide_input",
        arguments: { job_id: job.id, request_id: job.pending.id },
      },
    };
  }
  if (job.status === "running") {
    return { ...base, next: { tool: "get_job", arguments: { job_id: job.id } } };
  }
  if (job.status === "completed") {
    return {
      ...base,
      output: job.output,
      ...(job.integration === undefined ? {} : { integration: job.integration }),
    };
  }
  return {
    ...base,
    ...(job.error === undefined ? {} : { error: job.error }),
    ...(job.errorCode === undefined ? {} : { code: job.errorCode }),
  };
}
