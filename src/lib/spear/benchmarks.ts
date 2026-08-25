// Compatibility facade — the task registry lives in ./tasks/ now.
//
//   tasks/types.ts      contracts (TaskDef, baselines, milestones)
//   tasks/shared.ts     helpers, EXACT_LAWS, data builders, KV world
//   tasks/factories.ts  buildActivationTask / buildRegressionTask
//   tasks/domains/*     one file per domain — THIS is where you add tasks
//   tasks/index.ts      registry assembly + SPEAR_TASKS filter
export { buildTasks } from "./tasks/index";
export { taskOpProfile } from "./tasks/shared";
export type { TaskBaseline, TaskMilestone, TaskEval, TaskDef } from "./tasks/types";
